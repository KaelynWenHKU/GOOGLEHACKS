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
    # TODO: query regime_states for dates in [start_year-01-01, end_year-12-31]
    # TODO: join with yfinance XLV data to get actual_xlv_ret_1d for each date
    # TODO: return DataFrame sorted by date
    raise NotImplementedError


def print_metrics_table(metrics: dict) -> None:
    """
    Print the headline backtest metrics in a formatted table.

    Args:
        metrics: Output of hmm/backtest.compute_metrics().
    """
    # TODO: format and print a clean table comparing strategy vs buy-and-hold
    # Columns: Metric | Regime Strategy | XLV Buy-and-Hold
    raise NotImplementedError


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
    # TODO: predictions_df = load_predictions_from_mongo(db, start_year, end_year)

    logger.info("Running backtest simulation...")
    # TODO: backtest_df = run_backtest(predictions_df)

    logger.info("Computing metrics...")
    # TODO: metrics = compute_metrics(backtest_df)
    # TODO: print_metrics_table(metrics)

    logger.info("Generating regime timeline chart...")
    # TODO: fig = plot_regime_timeline(predictions_df)
    # TODO: fig.write_html(str(OUTPUT_DIR / "regime_timeline.html"))
    # TODO: logger.info(f"Saved: regime_timeline.html")

    logger.info("Generating full backtest report...")
    # TODO: generate_backtest_report(metrics, backtest_df, str(OUTPUT_DIR / "backtest_report.html"))
    # TODO: logger.info(f"Saved: backtest_report.html")

    logger.info("Saving metrics JSON...")
    # TODO: with open(OUTPUT_DIR / "backtest_metrics.json", "w") as f:
    # TODO:     json.dump(metrics, f, indent=2)

    logger.info("Validation complete. Check the output HTML files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthQuant walk-forward backtest validator")
    parser.add_argument("--start", type=int, default=2020, help="First test year")
    parser.add_argument("--end", type=int, default=2024, help="Last test year")
    args = parser.parse_args()
    main(start_year=args.start, end_year=args.end)
