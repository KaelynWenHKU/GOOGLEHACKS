"""
hmm/backtest.py
================
Walk-forward backtesting engine for the HMM regime model.

Implements the strategy from Section 15:
  - Risk-on regime:      hold 100% XLV
  - Neutral regime:      hold 50% XLV, 50% cash
  - Catalyst-fear regime: hold 0% XLV (cash or short via puts)
  Rebalanced daily based on predicted regime.

Target backtest metrics (Section 15):
  Total return 2020–2024:  to be computed (benchmark: XLV +68%)
  Max drawdown:            target < 25%   (benchmark: ~-35%)
  Sharpe ratio:            target > 0.8   (benchmark: ~0.6)
  Hit rate (10d direction): target > 55%

IMPORTANT: All predictions must use the walk-forward protocol from train.py.
Never predict on in-sample data. Report results with the disclaimer:
"Past performance does not guarantee future results."
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Regime → portfolio weight in XLV (0.0 = full cash, 1.0 = full XLV)
REGIME_WEIGHTS: dict[str, float] = {
    "risk-on": 1.0,
    "neutral": 0.5,
    "catalyst-fear": 0.0,
}

# Risk-free rate assumption for Sharpe ratio computation (annualised)
RISK_FREE_RATE_ANNUAL = 0.04


def run_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """
    Simulate the regime-following portfolio strategy against a buy-and-hold baseline.

    Args:
        predictions_df: DataFrame with columns:
            date (DatetimeIndex), predicted_regime (str), actual_xlv_ret_1d (float).
            Each row is one trading day with an out-of-sample regime prediction.
        initial_capital: Starting portfolio value in USD.

    Returns:
        DataFrame indexed by date with columns:
            strategy_value, buyhold_value, regime_label, daily_ret_strategy,
            daily_ret_buyhold, drawdown_strategy, drawdown_buyhold.
    """
    required = {"predicted_regime", "actual_xlv_ret_1d"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df is missing columns: {sorted(missing)}")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    frame = predictions_df.copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame = frame.sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("predictions_df must be non-empty with unique dates")
    unknown = set(frame["predicted_regime"].dropna()) - set(REGIME_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown regime labels: {sorted(unknown)}")
    returns = pd.to_numeric(frame["actual_xlv_ret_1d"], errors="coerce")
    if returns.isna().any() or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("actual_xlv_ret_1d must contain finite returns greater than -100%")

    signal_weight = frame["predicted_regime"].map(REGIME_WEIGHTS).astype(float)
    # A regime inferred from today's close can only be acted on from the next
    # trading session. Shifting one row is the key anti-lookahead execution rule.
    executed_weight = signal_weight.shift(1).fillna(0.0)
    strategy_returns = executed_weight * returns
    strategy_value = initial_capital * (1.0 + strategy_returns).cumprod()
    buyhold_value = initial_capital * (1.0 + returns).cumprod()

    output = pd.DataFrame(
        {
            "strategy_value": strategy_value,
            "buyhold_value": buyhold_value,
            "regime_label": frame["predicted_regime"],
            "signal_weight": signal_weight,
            "executed_weight": executed_weight,
            "daily_ret_strategy": strategy_returns,
            "daily_ret_buyhold": returns,
        },
        index=frame.index,
    )
    if "xlv_price" in frame:
        output["xlv_price"] = frame["xlv_price"]
    output["drawdown_strategy"] = output["strategy_value"] / output["strategy_value"].cummax() - 1.0
    output["drawdown_buyhold"] = output["buyhold_value"] / output["buyhold_value"].cummax() - 1.0
    output.attrs["initial_capital"] = float(initial_capital)
    return output


def compute_metrics(backtest_df: pd.DataFrame) -> dict:
    """
    Compute headline performance metrics for the backtest.

    Args:
        backtest_df: Output of run_backtest().

    Returns:
        Dict containing:
            total_return_strategy, total_return_buyhold,
            max_drawdown_strategy, max_drawdown_buyhold,
            sharpe_strategy, sharpe_buyhold,
            hit_rate_10d (fraction of 10d windows where direction was correct).
    """
    if backtest_df.empty:
        raise ValueError("backtest_df cannot be empty")
    initial = float(backtest_df.attrs.get("initial_capital", 100_000.0))
    daily_rf = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / 252.0) - 1.0

    def sharpe(returns: pd.Series) -> float:
        volatility = float(returns.std(ddof=1))
        if not np.isfinite(volatility) or volatility == 0:
            return 0.0
        return float((returns.mean() - daily_rf) / volatility * np.sqrt(252))

    return {
        "total_return_strategy": float(backtest_df["strategy_value"].iloc[-1] / initial - 1.0),
        "total_return_buyhold": float(backtest_df["buyhold_value"].iloc[-1] / initial - 1.0),
        "max_drawdown_strategy": float(backtest_df["drawdown_strategy"].min()),
        "max_drawdown_buyhold": float(backtest_df["drawdown_buyhold"].min()),
        "sharpe_strategy": sharpe(backtest_df["daily_ret_strategy"]),
        "sharpe_buyhold": sharpe(backtest_df["daily_ret_buyhold"]),
        "hit_rate_10d": compute_10d_hit_rate(backtest_df),
        "trading_days": int(len(backtest_df)),
    }


def compute_10d_hit_rate(backtest_df: pd.DataFrame) -> float:
    """
    Compute the fraction of 10-day forward windows where the strategy
    correctly predicted the direction of XLV returns.

    Args:
        backtest_df: Output of run_backtest() with daily return columns.

    Returns:
        Hit rate as a float in [0, 1].
    """
    if len(backtest_df) <= 10:
        return float("nan")
    returns = backtest_df["daily_ret_buyhold"].to_numpy(dtype=float)
    regimes = backtest_df["regime_label"].to_numpy()
    hits: list[bool] = []
    for index in range(len(backtest_df) - 10):
        regime = regimes[index]
        if regime == "neutral":
            continue
        forward_return = float(np.prod(1.0 + returns[index + 1 : index + 11]) - 1.0)
        expected_positive = regime == "risk-on"
        hits.append((forward_return > 0) == expected_positive)
    return float(np.mean(hits)) if hits else float("nan")


def plot_regime_timeline(
    predictions_df: pd.DataFrame,
    output_path: str | None = None,
) -> go.Figure:
    """
    Generate the regime timeline chart — the "money shot" for the demo video.

    Plots XLV price with background shading:
      Green  = risk-on
      Yellow = neutral
      Red    = catalyst-fear

    Key historical events are annotated (COVID crash, biotech bear, etc.).

    Args:
        predictions_df: DataFrame with columns: date, predicted_regime, xlv_price.
        output_path: If provided, save the chart to this path as HTML.

    Returns:
        Plotly Figure object (call .show() to display or .write_html() to save).
    """
    if predictions_df.empty or "predicted_regime" not in predictions_df:
        raise ValueError("predictions_df must include predicted_regime rows")
    frame = predictions_df.copy().sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    if "xlv_price" in frame:
        price = pd.to_numeric(frame["xlv_price"])
    elif "actual_xlv_ret_1d" in frame:
        price = 100.0 * (1.0 + pd.to_numeric(frame["actual_xlv_ret_1d"])).cumprod()
        price.name = "XLV indexed value"
    else:
        raise ValueError("Provide xlv_price or actual_xlv_ret_1d for the timeline")

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.index, y=price, mode="lines", name="XLV", line={"color": "#16324F"}))
    colors = {"risk-on": "#2CA58D", "neutral": "#F4B942", "catalyst-fear": "#D1495B"}
    period_id = frame["predicted_regime"].ne(frame["predicted_regime"].shift()).cumsum()
    for _, period in frame.groupby(period_id):
        regime = period["predicted_regime"].iloc[0]
        figure.add_vrect(
            x0=period.index[0],
            x1=period.index[-1],
            fillcolor=colors[regime],
            opacity=0.18,
            layer="below",
            line_width=0,
        )
    annotations = {
        pd.Timestamp("2020-03-16"): "COVID crash",
        pd.Timestamp("2022-09-01"): "2022 biotech drawdown",
    }
    for event_date, label in annotations.items():
        if frame.index.min() <= event_date <= frame.index.max():
            figure.add_vline(x=event_date, line_dash="dot", line_color="#555")
            figure.add_annotation(x=event_date, y=1.0, yref="paper", text=label, showarrow=False, yshift=10)
    figure.update_layout(
        title="HealthQuant Walk-Forward Regime Timeline",
        xaxis_title="Date",
        yaxis_title=price.name or "XLV price",
        template="plotly_white",
        hovermode="x unified",
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(output_path, include_plotlyjs=True)
    return figure


def generate_backtest_report(
    metrics: dict,
    backtest_df: pd.DataFrame,
    output_path: str = "backtest_report.html",
) -> None:
    """
    Write a self-contained HTML backtest report for the GitHub README.

    Includes: headline metrics table, portfolio value chart, regime timeline,
    and the educational disclaimer required by Section 15.

    Args:
        metrics: Output of compute_metrics().
        backtest_df: Output of run_backtest().
        output_path: File path to write the HTML report to.
    """
    required_metrics = [
        "total_return_strategy",
        "total_return_buyhold",
        "max_drawdown_strategy",
        "max_drawdown_buyhold",
        "sharpe_strategy",
        "sharpe_buyhold",
        "hit_rate_10d",
    ]
    missing = [name for name in required_metrics if name not in metrics]
    if missing:
        raise ValueError(f"metrics is missing fields: {missing}")

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    figure.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["strategy_value"], name="Regime strategy"), row=1, col=1)
    figure.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["buyhold_value"], name="XLV buy-and-hold"), row=1, col=1)
    figure.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["drawdown_strategy"], name="Strategy drawdown"), row=2, col=1)
    figure.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["drawdown_buyhold"], name="Buy-and-hold drawdown"), row=2, col=1)
    figure.update_yaxes(title_text="Portfolio value", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_layout(title="HealthQuant Out-of-Sample Walk-Forward Backtest", template="plotly_white", height=750)

    metric_rows = [
        ("Total return", metrics["total_return_strategy"], metrics["total_return_buyhold"], "percent"),
        ("Maximum drawdown", metrics["max_drawdown_strategy"], metrics["max_drawdown_buyhold"], "percent"),
        ("Sharpe ratio", metrics["sharpe_strategy"], metrics["sharpe_buyhold"], "number"),
    ]
    table = ["<table><thead><tr><th>Metric</th><th>Regime strategy</th><th>XLV buy-and-hold</th></tr></thead><tbody>"]
    for label, strategy, benchmark, value_type in metric_rows:
        formatter = (lambda value: f"{value:.1%}") if value_type == "percent" else (lambda value: f"{value:.2f}")
        table.append(f"<tr><td>{label}</td><td>{formatter(strategy)}</td><td>{formatter(benchmark)}</td></tr>")
    table.append(f"<tr><td>10-day directional hit rate</td><td>{metrics['hit_rate_10d']:.1%}</td><td>—</td></tr>")
    table.append("</tbody></table>")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>HealthQuant Backtest</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:24px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #ddd;text-align:right}"
        "th:first-child,td:first-child{text-align:left}.disclaimer{color:#666;font-size:14px}</style></head><body>"
        + "".join(table)
        + pio.to_html(figure, full_html=False, include_plotlyjs=True)
        + "<p class='disclaimer'>For educational and research purposes only. "
        "Past performance does not guarantee future results.</p></body></html>"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
