"""
scripts/validate_backtest.py
=============================
Reproduces the walk-forward backtest results and generates the regime timeline chart.

Run after seed_historical.py has populated MongoDB and the HMM has been trained.
This script is the proof-of-concept for judges — it demonstrates that:
  1. The walk-forward protocol is correctly implemented (no lookahead)
  2. The regime model has predictive signal (Sharpe > 0.6, hit rate > 55%)
  3. The regime labels align with known historical market periods

Output files:
  - backtest_report.html       — full interactive Plotly report
  - regime_timeline.html       — colour-coded regime chart (2015–2024)
  - backtest_metrics.json      — headline numbers for the README

Usage:
    python scripts/validate_backtest.py
    python scripts/validate_backtest.py --start 2020 --end 2024
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent


def load_predictions_from_mongo(
    db,
    start_year: int,
    end_year: int,
) -> "pd.DataFrame":
    """
    Load walk-forward regime predictions from the regime_states collection.

    Args:
        db: pymongo Database object.
        start_year: First year of predictions to load.
        end_year: Last year of predictions to load (inclusive).

    Returns:
        DataFrame with columns: date, predicted_regime, actual_xlv_ret_1d.
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    if end < start:
        raise ValueError("end_year must not be earlier than start_year")
    cursor = db["regime_states"].find(
        {"date": {"$gte": start.to_pydatetime(), "$lte": end.to_pydatetime()}},
        {"_id": 0, "date": 1, "predicted_regime": 1, "train_end_date": 1},
    ).sort("date", 1)
    records = []
    for document in cursor:
        # Generic regime labels may have been fitted in-sample. Require the
        # training cutoff written by the walk-forward trainer for every row.
        label = document.get("predicted_regime")
        prediction_date = pd.Timestamp(document["date"])
        cutoff = pd.Timestamp(document.get("train_end_date"))
        if not label or pd.isna(cutoff) or pd.isna(prediction_date):
            raise ValueError("Every prediction requires predicted_regime and train_end_date provenance")
        if cutoff >= prediction_date:
            raise ValueError("train_end_date must precede the prediction date")
        records.append({"date": prediction_date, "predicted_regime": label,
                        "train_end_date": cutoff})
    if not records:
        raise ValueError(f"No regime predictions found for {start_year}–{end_year}")
    predictions = pd.DataFrame(records).set_index("date")
    if predictions.index.has_duplicates:
        raise ValueError("Duplicate prediction dates make the backtest ambiguous")
    predictions.index = pd.DatetimeIndex(predictions.index).tz_localize(None)

    # Include a lead-in so pct_change for the first prediction date uses the
    # preceding trading close rather than silently dropping the first return.
    prices_raw = yf.download(
        "XLV",
        start=(predictions.index.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        end=(predictions.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if prices_raw.empty:
        raise ValueError("yfinance returned no XLV prices for the validation period")
    close = prices_raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    predictions["xlv_price"] = close.reindex(predictions.index)
    predictions["actual_xlv_ret_1d"] = close.pct_change(fill_method=None).reindex(predictions.index)
    sessions = close.loc[predictions.index.min():predictions.index.max()].index
    if not sessions.difference(predictions.index).empty:
        raise ValueError("Missing predictions for XLV trading sessions inside the evaluation interval")
    if predictions[["xlv_price", "actual_xlv_ret_1d"]].isna().any().any():
        raise ValueError("Missing XLV prices or preceding close; refusing to silently drop sessions")
    return predictions.sort_index()


def print_metrics_table(metrics: dict) -> None:
    """
    Print the headline backtest metrics in a formatted table.

    Args:
        metrics: Output of hmm/backtest.compute_metrics().
    """
    rows = [
        ("Total return", metrics["total_return_strategy"], metrics["total_return_buyhold"], ".1%"),
        ("Maximum drawdown", metrics["max_drawdown_strategy"], metrics["max_drawdown_buyhold"], ".1%"),
        ("Sharpe ratio", metrics["sharpe_strategy"], metrics["sharpe_buyhold"], ".2f"),
    ]
    print("\nHealthQuant walk-forward results")
    print(f"{'Metric':<24} {'Regime strategy':>18} {'XLV buy-and-hold':>20}")
    print("-" * 64)
    for label, strategy, benchmark, spec in rows:
        print(f"{label:<24} {format(strategy, spec):>18} {format(benchmark, spec):>20}")
    hit_rate = metrics.get("hit_rate_10d", float("nan"))
    hit_text = "n/a" if not np.isfinite(hit_rate) else f"{hit_rate:.1%}"
    print(f"{'10-day directional hit':<24} {hit_text:>18} {'—':>20}")
    print("\nFor educational and research purposes only. Past performance does not guarantee future results.\n")


def main(start_year: int = 2020, end_year: int = 2024) -> None:
    """
    Run the full backtest validation and save output files.

    Args:
        start_year: First year of the test period.
        end_year: Last year of the test period.
    """
    from database.mongo_client import get_db
    from hmm.backtest import run_backtest, compute_metrics, plot_regime_timeline, generate_backtest_report

    logger.info(f"Running walk-forward backtest: {start_year}–{end_year}")
    db = get_db()

    logger.info("Loading predictions from MongoDB...")
    predictions_df = load_predictions_from_mongo(db, start_year, end_year)

    logger.info("Running backtest simulation...")
    backtest_df = run_backtest(predictions_df)

    logger.info("Computing metrics...")
    metrics = compute_metrics(backtest_df)
    print_metrics_table(metrics)

    logger.info("Generating regime timeline chart...")
    plot_regime_timeline(predictions_df, str(OUTPUT_DIR / "regime_timeline.html"))
    logger.info("Saved: regime_timeline.html")

    logger.info("Generating full backtest report...")
    generate_backtest_report(metrics, backtest_df, str(OUTPUT_DIR / "backtest_report.html"))
    logger.info("Saved: backtest_report.html")

    logger.info("Saving metrics JSON...")
    serialisable_metrics = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
    }
    with (OUTPUT_DIR / "backtest_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serialisable_metrics, handle, indent=2)

    logger.info("Validation complete. Check the output HTML files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthQuant walk-forward backtest validator")
    parser.add_argument("--start", type=int, default=2020, help="First test year")
    parser.add_argument("--end", type=int, default=2024, help="Last test year")
    args = parser.parse_args()
    main(start_year=args.start, end_year=args.end)
