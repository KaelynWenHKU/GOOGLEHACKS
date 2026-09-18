"""Tests for point-in-time construction of the eight HMM features."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hmm.features import (
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_vector,
    get_avg_enrollment_active,
    get_pdufa_count_30d,
    get_phase3_completions_30d,
    standardise_features,
)


def make_db() -> dict:
    """Return dict-like mocked MongoDB collections."""
    pdufa = MagicMock()
    trials = MagicMock()
    pdufa.count_documents.return_value = 4
    trials.count_documents.return_value = 3
    trials.find.return_value = [
        {"enrollment_count": 100},
        {"enrollment_count": 300},
        {"enrollment_count": None},
    ]
    return {"pdufa_events": pdufa, "trial_events": trials}


def test_point_in_time_counts_include_date_and_knowledge_guards() -> None:
    db = make_db()

    assert get_pdufa_count_30d("2024-01-15", db) == 4
    assert get_phase3_completions_30d("2024-01-15", db) == 3

    pdufa_query = db["pdufa_events"].count_documents.call_args.args[0]
    trial_query = db["trial_events"].count_documents.call_args.args[0]
    assert pdufa_query["pdufa_date"]["$gte"] == pd.Timestamp("2024-01-15")
    assert pdufa_query["pdufa_date"]["$lte"] == pd.Timestamp("2024-02-14")
    assert "$or" in pdufa_query
    assert trial_query["phase"] == "PHASE3"
    assert "$or" in trial_query


def test_average_enrollment_uses_as_of_query_and_fallback() -> None:
    db = make_db()
    assert get_avg_enrollment_active(db, "2024-01-15") == 200.0
    query = db["trial_events"].find.call_args.args[0]
    assert query["phase"] == {"$in": ["PHASE2", "PHASE3"]}
    assert "$and" in query and "$or" in query

    db["trial_events"].find.return_value = []
    assert get_avg_enrollment_active(db, "2024-01-15") == 5000.0


def test_build_feature_vector_preserves_canonical_order() -> None:
    db = make_db()
    market = {
        "xlv_ret_5d": 0.01,
        "xlv_vol_20d": 0.02,
        "ibb_spy_ratio": -0.03,
        "xlv_rsi_14d": 0.6,
    }
    vector = build_feature_vector(
        "2024-01-15",
        db,
        market_feature_provider=lambda _: market,
        put_call_provider=lambda _: 1.25,
    )
    np.testing.assert_allclose(vector, [0.01, 0.02, -0.03, 0.6, 4, 3, 200, 1.25])
    assert vector.shape == (len(FEATURE_NAMES),)


def test_standardise_features_and_reuse_scaler() -> None:
    raw = np.arange(32, dtype=float).reshape(4, 8)
    scaled, scaler = standardise_features(raw)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    reused, returned = standardise_features(raw + 1.0, scaler)
    assert returned is scaler
    np.testing.assert_allclose(reused, scaler.transform(raw + 1.0))


def test_feature_matrix_requires_chronological_dates_and_scaler_reuse() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")

    def builder(date: str, _db) -> np.ndarray:
        day = pd.Timestamp(date).day
        return np.arange(8, dtype=float) + day

    scaled, scaler = build_feature_matrix(dates, {}, feature_builder=builder)
    assert scaled.shape == (4, 8)
    reused, _ = build_feature_matrix(
        dates,
        {},
        fit_scaler=False,
        scaler=scaler,
        feature_builder=builder,
    )
    np.testing.assert_allclose(reused, scaled)

    with pytest.raises(ValueError, match="chronological"):
        build_feature_matrix(dates[::-1], {}, feature_builder=builder)
    with pytest.raises(ValueError, match="fitted scaler"):
        build_feature_matrix(dates, {}, fit_scaler=False, feature_builder=builder)


def test_non_finite_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        standardise_features(np.full((2, 8), np.nan))

