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
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

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
    # TODO: apply REGIME_WEIGHTS[regime] to get portfolio weight for each day
    # TODO: strategy_daily_ret = weight * xlv_ret_1d  (cash earns 0 for simplicity)
    # TODO: compound daily returns to compute portfolio value
    # TODO: compute buyhold_value as 100% XLV compounded returns
    # TODO: compute drawdown series for both (rolling max then (value - max) / max)
    raise NotImplementedError


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
    # TODO: total_return = (final_value - initial) / initial
    # TODO: max_drawdown = backtest_df["drawdown_strategy"].min()
    # TODO: annualised_ret = (1 + total_return) ** (252 / trading_days) - 1
    # TODO: sharpe = (annualised_ret - RISK_FREE_RATE_ANNUAL) / (daily_rets.std() * sqrt(252))
    # TODO: hit_rate = compute_10d_hit_rate(backtest_df)
    raise NotImplementedError


def compute_10d_hit_rate(backtest_df: pd.DataFrame) -> float:
    """
    Compute the fraction of 10-day forward windows where the strategy
    correctly predicted the direction of XLV returns.

    Args:
        backtest_df: Output of run_backtest() with daily return columns.

    Returns:
        Hit rate as a float in [0, 1].
    """
    # TODO: for each row, compute 10d forward XLV cumulative return
    # TODO: predicted direction: risk-on → long (positive), catalyst-fear → short (negative)
    # TODO: hit = predicted direction matches actual 10d return direction
    # TODO: return mean of hits
    raise NotImplementedError


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
    # TODO: download XLV price history for the prediction date range
    # TODO: create plotly figure with XLV line chart
    # TODO: add vrect shading for each regime period:
    #   fig.add_vrect(x0=start, x1=end, fillcolor="green", opacity=0.15, layer="below")
    # TODO: annotate known events (COVID crash Mar 2020, biotech bear Q3 2022)
    # TODO: add legend and clean layout
    # TODO: optionally save to output_path
    raise NotImplementedError


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
    # TODO: create Plotly subplots: portfolio value + drawdown
    # TODO: add metrics table as a Plotly table trace
    # TODO: include disclaimer text in the HTML footer
    # TODO: write to output_path with fig.write_html()
    raise NotImplementedError
