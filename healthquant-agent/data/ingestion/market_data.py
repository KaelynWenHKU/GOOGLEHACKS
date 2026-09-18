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

Cache strategy: downloaded price data is saved as Parquet in data/cache/
keyed by ticker-range, so repeated calls for nearby dates reuse the same file.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ETF tickers used for feature computation
UNIVERSE_TICKERS = ["XLV", "IBB", "XBI", "SPY"]

# CBOE daily options statistics URL (free, no auth)
CBOE_OPTIONS_URL = "https://www.cboe.com/us/options/market_statistics/daily/"

# Cache directory relative to this file: ../../cache → data/cache/
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# ---------------------------------------------------------------------------
# Internal cache helpers
# ---------------------------------------------------------------------------

def _get_cache_path(start: str, end: str) -> Path:
    """Return path for a cached price Parquet file covering start→end."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Key on start/end dates so stale ranges don't get reused silently
    filename = f"prices_{start}_{end}.parquet"
    return _CACHE_DIR / filename


def _load_prices(start: str, end: str) -> pd.DataFrame:
    """
    Download UNIVERSE_TICKERS adjusted close prices, with disk caching.

    If a Parquet file for this date range already exists it is returned
    immediately, otherwise yfinance is called and the result is persisted.

    Args:
        start: ISO date string for the download start (inclusive).
        end:   ISO date string for the download end (exclusive).

    Returns:
        DataFrame of adjusted closing prices, columns = tickers, DatetimeIndex.

    Raises:
        ValueError: If the download returns an empty DataFrame.
    """
    cache_path = _get_cache_path(start, end)

    # --- Cache hit ---
    if cache_path.exists():
        logger.debug("Cache hit: %s", cache_path.name)
        print(f"  [cache] Loading prices from {cache_path.name}")
        closes = pd.read_parquet(cache_path)
        return closes

    # --- Cache miss: download via yfinance ---
    print(f"  [download] Fetching {UNIVERSE_TICKERS} from {start} to {end} via yfinance...")
    raw = yf.download(
        UNIVERSE_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {start} → {end}")

    # yfinance returns a MultiIndex (metric, ticker) when multiple tickers are given
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].dropna(how="all")
    else:
        # Single-ticker edge case (shouldn't happen here but guard it)
        closes = raw[["Close"]].rename(columns={"Close": UNIVERSE_TICKERS[0]})

    print(f"  [download] Got {len(closes)} trading days, {closes.shape[1]} tickers.")
    closes.to_parquet(cache_path)
    return closes


# ---------------------------------------------------------------------------
# Feature computation helpers (each takes the full closes DataFrame)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Uses Wilder's smoothing method approximated via simple rolling mean
    (adequate for an HMM feature — consistent across all dates).

    Args:
        series: Adjusted closing price series (must be longer than `period`).
        period: Lookback window in trading days (default 14).

    Returns:
        RSI series with values 0–100. Same index as input.
    """
    delta = series.diff()
    # Separate positive and negative price changes
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()

    # rs = inf when loss == 0 (pure uptrend) → RSI approaches 100 (correct)
    # numpy handles inf arithmetic: 100 - 100/(1+inf) == 100
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_xlv_ret_5d(closes: pd.DataFrame) -> float:
    """
    Compute the 5-day log return for XLV using the last 6 closing prices.

    Formula: log(P_t / P_{t-5})

    Args:
        closes: DataFrame of adjusted closing prices with ticker columns.

    Returns:
        5-day log return as a float (typically ±0.05 range).
    """
    xlv = closes["XLV"].dropna()
    if len(xlv) < 6:
        raise ValueError(f"Need ≥6 XLV closes for 5d return, got {len(xlv)}")
    return float(np.log(xlv.iloc[-1] / xlv.iloc[-6]))


def compute_xlv_vol_20d(closes: pd.DataFrame) -> float:
    """
    Compute the 20-day rolling standard deviation of XLV daily log returns.

    This is the raw daily vol (not annualised). Typical range: 0.005 – 0.025.

    Args:
        closes: DataFrame of adjusted closing prices.

    Returns:
        Standard deviation of the last 20 daily log returns (positive float).
    """
    xlv = closes["XLV"].dropna()
    if len(xlv) < 21:
        raise ValueError(f"Need ≥21 XLV closes for 20d vol, got {len(xlv)}")
    # Compute daily log returns, then take std of last 20
    daily_log_rets = np.log(xlv / xlv.shift(1)).dropna()
    return float(daily_log_rets.iloc[-20:].std())


def compute_ibb_spy_ratio(closes: pd.DataFrame) -> float:
    """
    Compute IBB relative strength vs SPY over 20 trading days.

    Formula: log(IBB_t / IBB_{t-20}) - log(SPY_t / SPY_{t-20})
    Positive = healthcare biotech outperforming broad market.

    Args:
        closes: DataFrame of adjusted closing prices.

    Returns:
        20-day relative log return of IBB over SPY (typically ±0.10 range).
    """
    ibb = closes["IBB"].dropna()
    spy = closes["SPY"].dropna()
    if len(ibb) < 21 or len(spy) < 21:
        raise ValueError("Need ≥21 closes for IBB/SPY to compute 20d relative return")
    ibb_ret_20d = float(np.log(ibb.iloc[-1] / ibb.iloc[-21]))
    spy_ret_20d = float(np.log(spy.iloc[-1] / spy.iloc[-21]))
    return ibb_ret_20d - spy_ret_20d


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def get_market_features(target_date: str) -> dict[str, float]:
    """
    Download price history and compute the 4 market-derived feature values
    for a given date. Uses strictly data available BEFORE/ON that date.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Features are computed
                     as of market close on this date.

    Returns:
        Dict with keys: xlv_ret_5d, xlv_vol_20d, ibb_spy_ratio, xlv_rsi_14d.
        All values are raw (not yet standardised — standardisation happens in hmm/features.py).

    Raises:
        ValueError: If insufficient price history is available for the date.
    """
    # We need ~30 trading days of history; 60 calendar days is a safe buffer
    target_ts = pd.Timestamp(target_date)
    start_ts = target_ts - pd.Timedelta(days=60)
    # yfinance end is exclusive, so add 1 day to include target_date
    end_ts = target_ts + pd.Timedelta(days=1)

    start_str = start_ts.strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")

    print(f"\n[market_data] Computing features for {target_date}")
    closes = _load_prices(start_str, end_str)

    # Restrict to data on or before target_date (no lookahead)
    closes = closes[closes.index <= target_ts]

    if len(closes) < 22:
        raise ValueError(
            f"Only {len(closes)} trading days available up to {target_date}; "
            "need at least 22 for all features."
        )

    # Compute RSI on full available XLV series, then take the last value
    xlv_rsi_series = compute_rsi(closes["XLV"].dropna(), period=14)
    xlv_rsi_raw = float(xlv_rsi_series.dropna().iloc[-1])
    # Normalise RSI from [0, 100] → [0, 1] as specified in Section 6
    xlv_rsi_norm = xlv_rsi_raw / 100.0

    features = {
        "xlv_ret_5d": compute_xlv_ret_5d(closes),
        "xlv_vol_20d": compute_xlv_vol_20d(closes),
        "ibb_spy_ratio": compute_ibb_spy_ratio(closes),
        "xlv_rsi_14d": xlv_rsi_norm,
    }

    return features


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
    try:
        # CBOE daily file URL pattern for a given date
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        date_str = dt.strftime("%Y%m%d")
        url = f"https://www.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt={date_str}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # The page contains a link to a CSV with the actual data
        # Pattern: /us/options/market_statistics/daily/csvfile/?dt=YYYYMMDD
        csv_url = f"https://www.cboe.com/us/options/market_statistics/daily/csvfile/?dt={date_str}"
        csv_response = requests.get(csv_url, timeout=10)
        csv_response.raise_for_status()

        # Parse CSV — CBOE format has header rows before the data table
        from io import StringIO
        df = pd.read_csv(StringIO(csv_response.text), skiprows=3)

        # Locate XLV row (Symbol column)
        xlv_row = df[df.iloc[:, 0].astype(str).str.strip().str.upper() == "XLV"]
        if xlv_row.empty:
            logger.warning("XLV not found in CBOE daily data for %s", target_date)
            return 1.0

        # Columns: Symbol, Exchange, Calls, Puts, Total, P/C Ratio
        pc_ratio = float(xlv_row.iloc[0, 5])
        return pc_ratio

    except Exception as exc:
        # Degrade gracefully — put/call ratio is a secondary signal
        logger.warning("Failed to fetch XLV put/call ratio for %s: %s", target_date, exc)
        return 1.0  # 1.0 = neutral (equal puts and calls)


def get_features_for_date_range(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute market features for every trading day in a date range.

    Used during historical data seeding (2015–2024) to batch-populate
    the regime_states collection. Efficiently downloads all price data
    in a single yfinance call rather than one call per date.

    Args:
        start_date: ISO date string — first day to compute features for.
        end_date: ISO date string — last day to compute features for.

    Returns:
        DataFrame indexed by date, with columns:
        xlv_ret_5d, xlv_vol_20d, ibb_spy_ratio, xlv_rsi_14d.
    """
    # Add 60-day lead-in so even the first date has full history
    lead_start = (pd.Timestamp(start_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n[market_data] Batch-computing features from {start_date} to {end_date}")
    all_closes = _load_prices(lead_start, end_exclusive)

    # Get the list of trading days in the requested range
    trading_days = all_closes.index[
        (all_closes.index >= pd.Timestamp(start_date)) &
        (all_closes.index <= pd.Timestamp(end_date))
    ]

    print(f"  [batch] Processing {len(trading_days)} trading days...")
    records: list[dict] = []

    # Vectorised RSI for the full series (compute once, slice per day)
    xlv_rsi_full = compute_rsi(all_closes["XLV"].dropna(), period=14)
    xlv_daily_log_ret = np.log(all_closes["XLV"] / all_closes["XLV"].shift(1))
    ibb_daily_log_ret = np.log(all_closes["IBB"] / all_closes["IBB"].shift(1))
    spy_daily_log_ret = np.log(all_closes["SPY"] / all_closes["SPY"].shift(1))

    for day in trading_days:
        # Slice history up to and including this day (no lookahead)
        xlv_to_day = all_closes["XLV"].loc[:day].dropna()
        ibb_to_day = all_closes["IBB"].loc[:day].dropna()
        spy_to_day = all_closes["SPY"].loc[:day].dropna()

        if len(xlv_to_day) < 22:
            logger.debug("Skipping %s — insufficient history", day.date())
            continue

        try:
            # 5-day log return
            xlv_ret_5d = float(np.log(xlv_to_day.iloc[-1] / xlv_to_day.iloc[-6]))

            # 20-day vol: std of last 20 daily log returns
            xlv_dr = xlv_daily_log_ret.loc[:day].dropna()
            xlv_vol_20d = float(xlv_dr.iloc[-20:].std())

            # IBB vs SPY 20-day relative return
            ibb_ret = float(np.log(ibb_to_day.iloc[-1] / ibb_to_day.iloc[-21]))
            spy_ret = float(np.log(spy_to_day.iloc[-1] / spy_to_day.iloc[-21]))
            ibb_spy_ratio = ibb_ret - spy_ret

            # RSI (already computed for full series — just look up this day)
            rsi_val = xlv_rsi_full.loc[day] if day in xlv_rsi_full.index else np.nan
            xlv_rsi_14d = float(rsi_val) / 100.0 if not np.isnan(rsi_val) else np.nan

            records.append({
                "date": day,
                "xlv_ret_5d": xlv_ret_5d,
                "xlv_vol_20d": xlv_vol_20d,
                "ibb_spy_ratio": ibb_spy_ratio,
                "xlv_rsi_14d": xlv_rsi_14d,
            })

        except Exception as exc:
            logger.warning("Skipping %s due to error: %s", day.date(), exc)

    result = pd.DataFrame(records).set_index("date")
    print(f"  [batch] Done. Computed features for {len(result)} trading days.")
    return result


# ---------------------------------------------------------------------------
# Run-as-main: test output for today + validation on 5 dates in 2023
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("HealthQuant — market_data.py self-test")
    print("=" * 60)

    # ---- Test 1: today's date ----
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n--- Feature values for TODAY ({today}) ---")
    try:
        today_features = get_market_features(today)
        for k, v in today_features.items():
            print(f"  {k:25s} = {v:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ---- Test 2: 5 random dates in 2023 ----
    print("\n--- Validation: 5 random trading dates in 2023 ---")

    # Deterministic seed so results are reproducible
    random.seed(42)
    # Generate random weekdays in 2023 (avoid weekends; market is open ~252 days)
    test_dates: list[str] = []
    while len(test_dates) < 5:
        day_offset = random.randint(0, 364)
        candidate = (datetime(2023, 1, 3) + timedelta(days=day_offset))
        # Skip weekends
        if candidate.weekday() < 5:
            test_dates.append(candidate.strftime("%Y-%m-%d"))

    print(f"\n  Selected dates: {test_dates}\n")

    all_ok = True
    for date_str in test_dates:
        print(f"  Date: {date_str}")
        try:
            feats = get_market_features(date_str)

            # Sanity checks
            ret = feats["xlv_ret_5d"]
            vol = feats["xlv_vol_20d"]
            ratio = feats["ibb_spy_ratio"]
            rsi = feats["xlv_rsi_14d"]

            ret_ok  = -0.20 < ret < 0.20      # 5d returns rarely exceed ±20%
            vol_ok  = vol > 0                   # vol must be positive
            ratio_ok = -0.30 < ratio < 0.30    # relative strength rarely ±30%
            rsi_ok  = 0.0 <= rsi <= 1.0        # normalised to [0, 1]

            status = "✓" if all([ret_ok, vol_ok, ratio_ok, rsi_ok]) else "✗"
            if not all([ret_ok, vol_ok, ratio_ok, rsi_ok]):
                all_ok = False

            print(f"    xlv_ret_5d      = {ret:+.6f}  {'✓' if ret_ok else '✗ OUT OF RANGE'}")
            print(f"    xlv_vol_20d     = {vol:.6f}   {'✓' if vol_ok else '✗ MUST BE POSITIVE'}")
            print(f"    ibb_spy_ratio   = {ratio:+.6f}  {'✓' if ratio_ok else '✗ OUT OF RANGE'}")
            print(f"    xlv_rsi_14d     = {rsi:.6f}   {'✓' if rsi_ok else '✗ MUST BE 0–1'}")
            print(f"    → All checks: {status}\n")

        except Exception as e:
            print(f"    ERROR: {e}\n")
            all_ok = False

    print("=" * 60)
    print(f"Validation result: {'ALL PASSED ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")
    print("=" * 60)
