"""
hmm/features.py
================
Assembles the full 8-dimensional feature vector for the HMM.

Feature index → name → source:
  [0] xlv_ret_5d             yfinance XLV          5-day log return
  [1] xlv_vol_20d            yfinance XLV          20-day rolling std of daily returns
  [2] ibb_spy_ratio          yfinance IBB, SPY     IBB 20d return minus SPY 20d return
  [3] xlv_rsi_14d            yfinance XLV          14-day RSI, normalised to [0, 1]
  [4] pdufa_events_30d       MongoDB pdufa_events  Count of PDUFA dates in next 30d
  [5] phase3_completions_30d MongoDB trial_events  Count of Phase 3 completions in next 30d
  [6] avg_enrollment_active  MongoDB trial_events  Mean enrollment across active Phase 2/3
  [7] xlv_put_call_ratio     CBOE daily data       XLV options put/call ratio

CRITICAL: All features must be computed using data strictly available
before the target date — no lookahead. This is enforced by the
`before_date` parameter passed to all MongoDB queries.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "xlv_ret_5d",
    "xlv_vol_20d",
    "ibb_spy_ratio",
    "xlv_rsi_14d",
    "pdufa_events_30d",
    "phase3_completions_30d",
    "avg_enrollment_active",
    "xlv_put_call_ratio",
]


def build_feature_vector(
    target_date: str,
    db,
) -> np.ndarray:
    """
    Compute the raw (un-standardised) 8-dim feature vector for a single date.

    Args:
        target_date: ISO date string "YYYY-MM-DD". All data is sourced
                     as of market close on this date (no lookahead).
        db: pymongo Database object (healthquant).

    Returns:
        numpy array of shape (8,) containing the raw feature values.

    Raises:
        ValueError: If any feature cannot be computed (missing data).
    """
    # TODO: call market_data.get_market_features(target_date) → features [0:4]
    # TODO: call get_pdufa_count_30d(target_date, db) → feature [4]
    # TODO: call get_phase3_completions_30d(target_date, db) → feature [5]
    # TODO: call get_avg_enrollment_active(db) → feature [6]
    # TODO: call market_data.get_xlv_put_call_ratio(target_date) → feature [7]
    # TODO: assemble into np.array and return
    raise NotImplementedError


def build_feature_matrix(
    date_range: pd.DatetimeIndex,
    db,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Build a (T × 8) feature matrix for a sequence of dates, with standardisation.

    Used by hmm/train.py to prepare training data. Enforces strict no-lookahead:
    features for date[i] only use data available before date[i].

    Args:
        date_range: Sequence of trading dates to compute features for.
        db: pymongo Database object.
        fit_scaler: If True, fit a new StandardScaler on this matrix.
                    If False, the caller is responsible for transforming with a
                    pre-fitted scaler (required for test periods in walk-forward).

    Returns:
        Tuple of (feature_matrix of shape (T, 8), fitted StandardScaler).
        If fit_scaler=False, returns the un-scaled matrix and a dummy scaler.
    """
    # TODO: loop over date_range, call build_feature_vector() for each date
    # TODO: stack into np.array of shape (T, 8)
    # TODO: if fit_scaler=True, fit StandardScaler on the matrix and transform
    # TODO: return (scaled_matrix, scaler)
    raise NotImplementedError


def get_pdufa_count_30d(target_date: str, db) -> int:
    """
    Count PDUFA events in the 30-day window following target_date.

    Uses only MongoDB data — no API call at prediction time.

    Args:
        target_date: ISO date string.
        db: pymongo Database object.

    Returns:
        Integer count of upcoming PDUFA dates in [target_date, target_date + 30d].
    """
    # TODO: parse target_date to datetime
    # TODO: query pdufa_events collection:
    #   {"pdufa_date": {"$gte": target_dt, "$lte": target_dt + timedelta(days=30)}}
    # TODO: return count_documents()
    raise NotImplementedError


def get_phase3_completions_30d(target_date: str, db) -> int:
    """
    Count Phase 3 trials with primary completion dates in the next 30 days.

    Args:
        target_date: ISO date string.
        db: pymongo Database object.

    Returns:
        Integer count of Phase 3 completions due within 30 days of target_date.
    """
    # TODO: query trial_events collection for PHASE3 trials with
    #   primary_completion_date in [target_dt, target_dt + 30d]
    # TODO: return count_documents()
    raise NotImplementedError


def get_avg_enrollment_active(db) -> float:
    """
    Compute mean enrollment count across all currently active Phase 2/3 trials.

    This is a sector-level measure of pipeline density. Computed from the
    trial_events collection; cached for the current day to avoid redundant queries.

    Args:
        db: pymongo Database object.

    Returns:
        Mean enrollment count as a float. Returns 5000.0 as a neutral fallback
        if no active trials are found in the database.
    """
    # TODO: aggregation pipeline:
    #   $match: {phase: {$in: ["PHASE2","PHASE3"]}, status: {$in: ["RECRUITING","ACTIVE_NOT_RECRUITING"]}}
    #   $group: {_id: null, avg_enrollment: {$avg: "$enrollment_count"}}
    # TODO: return result[0]["avg_enrollment"] or 5000.0 fallback
    raise NotImplementedError


def standardise_features(
    raw_matrix: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardise a feature matrix to zero mean and unit variance.

    Args:
        raw_matrix: Array of shape (T, 8) with raw feature values.
        scaler: Pre-fitted StandardScaler to use. If None, fits a new one.

    Returns:
        Tuple of (scaled_matrix, scaler).
    """
    # TODO: if scaler is None, create and fit a new StandardScaler
    # TODO: transform raw_matrix using scaler.transform()
    # TODO: return (scaled_matrix, scaler)
    raise NotImplementedError
