"""
data/ingestion/market_data.py
================================
Market data feature computation using yfinance (free, no API key).

Computes the 4 market-derived dimensions of the 8-dim feature vector:
  [0] xlv_ret_5d      — XLV 5-day log return
  [1] xlv_vol_20d     — XLV 20-day rolling volatility
  [2] ibb_spy_ratio   — IBB 20d return minus SPY 20d return (relative strength)
  [3] xlv_rsi_14d     — XLV 14-day RSI, normalised to [0, 1]
  [7] xlv_put_call_ratio — XLV options put/call ratio from CBOE daily data

The remaining 3 features (indices 4, 5, 6) are computed from MongoDB
in hmm/features.py, which assembles the full 8-dim vector.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ETF tickers used for feature computation
UNIVERSE_TICKERS = ["XLV", "IBB", "XBI", "SPY"]

# CBOE daily options statistics URL (free, no auth)
CBOE_OPTIONS_URL = "https://www.cboe.com/us/options/market_statistics/daily/"


def get_market_features(target_date: str) -> dict[str, float]:
    """
    Download price history and compute the 4 market-derived feature values
    for a given date. Uses strictly data available BEFORE that date.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Features are computed
                     as of market close on this date.

    Returns:
        Dict with keys: xlv_ret_5d, xlv_vol_20d, ibb_spy_ratio, xlv_rsi_14d.
        All values are raw (not yet standardised — standardisation happens in hmm/features.py).

    Raises:
        ValueError: If insufficient price history is available for the date.
    """
    # TODO: compute start date as target_date - 60 calendar days (buffer for weekends/holidays)
    # TODO: download UNIVERSE_TICKERS with yf.download(auto_adjust=True)
    # TODO: call compute_xlv_ret_5d, compute_xlv_vol_20d, compute_ibb_spy_ratio, compute_rsi
    # TODO: validate that we have at least 25 trading days of data
    raise NotImplementedError


def compute_xlv_ret_5d(closes: pd.DataFrame) -> float:
    """
    Compute the 5-day log return for XLV using the last 6 closing prices.

    Formula: log(P_t / P_{t-5})

    Args:
        closes: DataFrame of adjusted closing prices with ticker columns.

    Returns:
        5-day log return as a float.
    """
    # TODO: xlv = closes["XLV"]; return float(np.log(xlv.iloc[-1] / xlv.iloc[-6]))
    raise NotImplementedError


def compute_xlv_vol_20d(closes: pd.DataFrame) -> float:
    """
    Compute the 20-day rolling standard deviation of XLV daily log returns.

    Args:
        closes: DataFrame of adjusted closing prices.

    Returns:
        Annualised-equivalent daily volatility (not annualised here — raw std).
    """
    # TODO: compute daily log returns for XLV
    # TODO: return std of last 20 daily returns
    raise NotImplementedError


def compute_ibb_spy_ratio(closes: pd.DataFrame) -> float:
    """
    Compute IBB relative strength vs SPY over 20 trading days.

    Formula: log(IBB_t / IBB_{t-20}) - log(SPY_t / SPY_{t-20})
    Positive = healthcare outperforming broad market.

    Args:
        closes: DataFrame of adjusted closing prices.

    Returns:
        20-day relative log return of IBB over SPY.
    """
    # TODO: compute 20d log returns for IBB and SPY separately
    # TODO: return the difference (IBB return - SPY return)
    raise NotImplementedError


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Uses Wilder's smoothing method via simple rolling mean (adequate for HMM feature).

    Args:
        series: Adjusted closing price series (must be longer than `period`).
        period: Lookback window in trading days (default 14).

    Returns:
        RSI series (values 0–100). Same index as input.
    """
    # delta = series.diff()
    # gain = delta.clip(lower=0).rolling(period).mean()
    # loss = (-delta.clip(upper=0)).rolling(period).mean()
    # rs = gain / loss
    # return 100 - (100 / (1 + rs))
    raise NotImplementedError


def get_xlv_put_call_ratio(target_date: str) -> float:
    """
    Fetch the XLV put/call ratio from CBOE daily options statistics.

    The CBOE publishes free daily options data CSV at:
    https://www.cboe.com/us/options/market_statistics/daily/

    Args:
        target_date: ISO date string "YYYY-MM-DD".

    Returns:
        XLV put/call ratio as a float. Returns 1.0 (neutral) on failure
        so that downstream features degrade gracefully rather than crash.
    """
    # TODO: construct the CBOE daily file URL for target_date
    # TODO: download the CSV with requests
    # TODO: parse the DataFrame and locate the XLV row
    # TODO: return put_volume / call_volume
    # TODO: on any exception, log a warning and return 1.0 (neutral fallback)
    raise NotImplementedError


def get_features_for_date_range(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute market features for every trading day in a date range.

    Used during historical data seeding (2015–2024) to batch-populate
    the regime_states collection.

    Args:
        start_date: ISO date string — first day to compute features for.
        end_date: ISO date string — last day to compute features for.

    Returns:
        DataFrame indexed by date, with columns:
        xlv_ret_5d, xlv_vol_20d, ibb_spy_ratio, xlv_rsi_14d, xlv_put_call_ratio.
    """
    # TODO: download all price history in a single yf.download call (efficient)
    # TODO: compute features using vectorised pandas operations (avoid row-by-row loop)
    # TODO: return DataFrame with DatetimeIndex
    raise NotImplementedError
