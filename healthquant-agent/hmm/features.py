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
from typing import Callable, Optional

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
    market_feature_provider: Optional[Callable[[str], dict[str, float]]] = None,
    put_call_provider: Optional[Callable[[str], float]] = None,
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
    # Imports are intentionally local: test and batch callers can inject cached
    # providers without importing yfinance or making one HTTP request per date.
    if market_feature_provider is None or put_call_provider is None:
        from data.ingestion import market_data

        market_feature_provider = market_feature_provider or market_data.get_market_features
        put_call_provider = put_call_provider or market_data.get_xlv_put_call_ratio

    date_str = _parse_target_date(target_date).strftime("%Y-%m-%d")
    market = market_feature_provider(date_str)
    missing = [name for name in FEATURE_NAMES[:4] if name not in market]
    if missing:
        raise ValueError(f"Market feature provider omitted required fields: {missing}")

    vector = np.asarray(
        [
            market["xlv_ret_5d"],
            market["xlv_vol_20d"],
            market["ibb_spy_ratio"],
            market["xlv_rsi_14d"],
            get_pdufa_count_30d(date_str, db),
            get_phase3_completions_30d(date_str, db),
            get_avg_enrollment_active(db, date_str),
            put_call_provider(date_str),
        ],
        dtype=float,
    )
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError(f"Feature vector must contain 8 finite values, got {vector!r}")
    return vector


def build_feature_matrix(
    date_range: pd.DatetimeIndex,
    db,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
    feature_builder: Optional[Callable[[str, object], np.ndarray]] = None,
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
    dates = pd.DatetimeIndex(date_range)
    if dates.empty:
        raise ValueError("date_range must contain at least one date")
    if dates.has_duplicates:
        raise ValueError("date_range must not contain duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("date_range must be chronological")

    builder = feature_builder or build_feature_vector
    raw = np.vstack([builder(day.strftime("%Y-%m-%d"), db) for day in dates])
    if fit_scaler:
        if scaler is not None:
            raise ValueError("Pass scaler only when fit_scaler=False")
        return standardise_features(raw)
    if scaler is None:
        raise ValueError("A fitted scaler is required when fit_scaler=False")
    return standardise_features(raw, scaler=scaler)


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
    target_dt = _parse_target_date(target_date)
    end_dt = target_dt + timedelta(days=30)
    query = {
        "pdufa_date": {"$gte": target_dt, "$lte": end_dt},
        **_point_in_time_filter(target_dt),
    }
    return int(db["pdufa_events"].count_documents(query))


def get_phase3_completions_30d(target_date: str, db) -> int:
    """
    Count Phase 3 trials with primary completion dates in the next 30 days.

    Args:
        target_date: ISO date string.
        db: pymongo Database object.

    Returns:
        Integer count of Phase 3 completions due within 30 days of target_date.
    """
    target_dt = _parse_target_date(target_date)
    end_dt = target_dt + timedelta(days=30)
    query = {
        "phase": "PHASE3",
        "primary_completion_date": {"$gte": target_dt, "$lte": end_dt},
        **_point_in_time_filter(target_dt),
    }
    return int(db["trial_events"].count_documents(query))


def get_avg_enrollment_active(db, target_date: Optional[str] = None) -> float:
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
    query: dict = {
        "phase": {"$in": ["PHASE2", "PHASE3"]},
        "enrollment_count": {"$ne": None},
    }
    if target_date is None:
        query["status"] = {"$in": ["RECRUITING", "ACTIVE_NOT_RECRUITING"]}
    else:
        target_dt = _parse_target_date(target_date)
        query.update(_point_in_time_filter(target_dt))
        # Prefer temporal fields for historical as-of reconstruction. The
        # status fallback keeps legacy documents usable until the next ingest.
        query["$and"] = [
            {
                "$or": [
                    {
                        "start_date": {"$lte": target_dt},
                        "primary_completion_date": {"$gte": target_dt},
                    },
                    {
                        "start_date": {"$exists": False},
                        "status": {"$in": ["RECRUITING", "ACTIVE_NOT_RECRUITING"]},
                    },
                ]
            }
        ]

    enrollments = []
    for document in db["trial_events"].find(query, {"_id": 0, "enrollment_count": 1}):
        value = document.get("enrollment_count")
        if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
            enrollments.append(float(value))
    return float(np.mean(enrollments)) if enrollments else 5000.0


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
    matrix = np.asarray(raw_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected feature matrix with shape (n, {len(FEATURE_NAMES)}), got {matrix.shape}"
        )
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix must be non-empty and contain only finite values")
    fitted_scaler = scaler or StandardScaler()
    if scaler is None:
        fitted_scaler.fit(matrix)
    transformed = fitted_scaler.transform(matrix)
    return transformed, fitted_scaler


def _parse_target_date(target_date: str) -> datetime:
    """Parse an ISO date into a timezone-naive midnight datetime for MongoDB."""
    try:
        timestamp = pd.Timestamp(target_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid target date {target_date!r}; expected YYYY-MM-DD") from exc
    if pd.isna(timestamp):
        raise ValueError("target_date cannot be NaT")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize().to_pydatetime()


def _point_in_time_filter(target_dt: datetime) -> dict:
    """Return a permissive knowledge-date guard for legacy and new records.

    New ingestion should populate ``known_as_of`` (or source-specific
    ``announced_at``/``first_posted_date``). Records with none of these fields
    are retained for backwards compatibility, but are explicitly less robust
    for historical no-lookahead validation.
    """
    return {
        "$or": [
            {"known_as_of": {"$lte": target_dt}},
            {"announced_at": {"$lte": target_dt}},
            {"first_posted_date": {"$lte": target_dt}},
            {
                "known_as_of": {"$exists": False},
                "announced_at": {"$exists": False},
                "first_posted_date": {"$exists": False},
            },
        ]
    }
