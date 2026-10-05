"""Tests for robust HMM fitting, labeling, persistence, and walk-forward use."""

from types import SimpleNamespace
import pickle
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import hmm.train as train_module
from hmm.features import FEATURE_NAMES
from hmm.train import infer_state_labels, load_model, run_walk_forward_training, save_model, train_hmm
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("history", [[0, 1], [1, 0], [float("nan"), 0], [0]])
def test_iteration_limit_or_falling_likelihood_is_not_convergence(monkeypatch, history):
    """An apparently successful monitor must not admit an unfinished EM fit."""
    class Candidate:
        def __init__(self, **kwargs):
            self.monitor_ = SimpleNamespace(converged=True, history=history, iter=200)

        def fit(self, matrix):
            return self

        def score(self, matrix):
            return 100.0

    monkeypatch.setattr(train_module, "GaussianHMM", Candidate)
    with pytest.raises(RuntimeError, match="All HMM restarts failed"):
        train_hmm(synthetic_features(), n_restarts=2)


def synthetic_features(rows: int = 120, seed: int = 7) -> np.ndarray:
    """Create three well-separated regimes with all eight features varying."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [1.5, -1.0, 1.2, 0.6, -0.7, -0.5, 0.2, -0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.2, 1.5, -0.8, -0.5, 1.4, 0.7, -0.2, 1.3],
        ]
    )
    states = np.repeat(np.arange(3), repeats=np.ceil(rows / 3).astype(int))[:rows]
    return centers[states] + rng.normal(0, 0.18, size=(rows, 8))


def test_train_hmm_fits_valid_transition_model() -> None:
    matrix = synthetic_features()
    model = train_hmm(matrix, n_restarts=2)
    assert model.means_.shape == (3, 8)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)
    assert np.isfinite(model.score(matrix))


def test_infer_state_labels_is_one_to_one_when_scores_overlap() -> None:
    means = np.array(
        [
            [2.0, 2.2, 1.0, 0.5, 1.0, 0.0, 0.0, 1.0],
            [0.2, 1.7, -0.3, 0.0, 2.2, 0.0, 0.0, 2.0],
            [-0.2, -0.5, 0.0, -0.1, -0.5, 0.0, 0.0, -0.4],
        ]
    )
    mapping = infer_state_labels(SimpleNamespace(means_=means))
    assert set(mapping) == {0, 1, 2}
    assert set(mapping.values()) == {"risk-on", "neutral", "catalyst-fear"}


def test_checkpoint_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
    matrix = synthetic_features(90)
    scaler = StandardScaler().fit(matrix)
    model = train_hmm(scaler.transform(matrix), n_restarts=1)
    labels = infer_state_labels(model)

    path = save_model(model, scaler, labels, "2024-01-31")
    loaded_model, loaded_scaler, loaded_labels = load_model("latest")

    assert path.exists()
    assert (tmp_path / "hmm_latest.pkl").exists()
    assert loaded_labels == labels
    np.testing.assert_allclose(loaded_model.means_, model.means_)
    np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)


def test_walk_forward_predictions_never_train_on_prediction_date(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
    dates = pd.bdate_range("2023-09-01", "2024-01-19")
    values = synthetic_features(len(dates), seed=11)
    frame = pd.DataFrame(values, index=dates, columns=FEATURE_NAMES)
    prices = pd.Series(100 * np.exp(np.linspace(0, 0.15, len(dates) + 15)), index=pd.bdate_range(dates[0], periods=len(dates) + 15))

    result = run_walk_forward_training(
        db={},
        start_date="2023-09-01",
        test_years=[2024],
        feature_frame=frame,
        xlv_prices=prices,
        n_restarts=1,
    )[0]
    predictions = result["predictions_df"]

    assert not predictions.empty
    assert predictions.index.min().year == 2024
    assert all(pd.Timestamp(end) < date for date, end in zip(predictions.index, predictions["train_end_date"]))
    assert predictions["actual_xlv_ret_1d"].notna().all()
    assert predictions["actual_xlv_return_10d"].notna().all()


@pytest.mark.parametrize("source", ["features", "prices"])
@pytest.mark.parametrize("defect", ["duplicate", "missing", "intraday"])
def test_walk_forward_rejects_ambiguous_dates_before_fitting(monkeypatch, source, defect):
    dates = pd.bdate_range("2023-09-01", "2024-01-19")
    frame = pd.DataFrame(synthetic_features(len(dates)), index=dates, columns=FEATURE_NAMES)
    prices = pd.Series(100.0, index=dates)
    bad_dates = dates.to_list()
    bad_dates[1] = {"duplicate": dates[0], "missing": pd.NaT,
                    "intraday": dates[1] + pd.Timedelta(hours=12)}[defect]
    if source == "features":
        frame.index = pd.DatetimeIndex(bad_dates)
    else:
        prices.index = pd.DatetimeIndex(bad_dates)
    monkeypatch.setattr(train_module, "train_hmm", lambda *a, **k: pytest.fail("fit started before validation"))
    with pytest.raises(ValueError, match="dates must"):
        run_walk_forward_training({}, "2023-09-01", [2024], frame, prices)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.nan, np.inf])
def test_walk_forward_rejects_invalid_closes_before_fitting(monkeypatch, bad_price):
    dates = pd.bdate_range("2023-09-01", "2024-01-19")
    frame = pd.DataFrame(synthetic_features(len(dates)), index=dates, columns=FEATURE_NAMES)
    prices = pd.Series(100.0, index=dates)
    prices.iloc[-1] = bad_price
    monkeypatch.setattr(train_module, "train_hmm", lambda *a, **k: pytest.fail("fit started before validation"))
    with pytest.raises(ValueError, match="strictly positive"):
        run_walk_forward_training({}, "2023-09-01", [2024], frame, prices)


def test_walk_forward_rejects_empty_years():
    with pytest.raises(ValueError, match="must not be empty"):
        run_walk_forward_training({}, test_years=[])


@pytest.mark.parametrize("defect", ["missing_schema", "wrong_order", "missing_cutoff", "invalid_cutoff", "not_dict"])
def test_checkpoint_requires_explicit_valid_metadata(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
    payload = {"model": None, "scaler": None,
               "state_label_map": {0: "risk-on", 1: "neutral", 2: "catalyst-fear"},
               "feature_names": FEATURE_NAMES.copy(), "train_end_date": "2024-01-31"}
    if defect == "missing_schema":
        del payload["feature_names"]
    elif defect == "wrong_order":
        payload["feature_names"] = FEATURE_NAMES[::-1]
    elif defect == "missing_cutoff":
        del payload["train_end_date"]
    elif defect == "invalid_cutoff":
        payload["train_end_date"] = "NaT"
    else:
        payload = []
    with (tmp_path / "hmm_latest.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError):
        load_model()


@pytest.mark.parametrize("bad_date", [None, "NaT", "2024-02-30", "2024-01-31T12:00:00", "2024-1-1", "../outside"])
def test_checkpoint_dates_are_canonical_before_file_access(tmp_path, monkeypatch, bad_date):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
    labels = {0: "risk-on", 1: "neutral", 2: "catalyst-fear"}
    with pytest.raises(ValueError, match="date"):
        save_model(None, None, labels, bad_date)
    with pytest.raises(ValueError, match="date"):
        load_model(bad_date)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("forward,observed,keep", [
    (.1, pd.Timestamp("2024-01-16"), True),
    (np.nan, pd.Timestamp("2024-01-16"), False),
    (.1, None, False), (np.inf, None, False),
    (.1, pd.NaT, False), (.1, pd.Timestamp("2024-01-02"), False),
])
def test_prediction_update_clears_stale_derived_fields(forward, observed, keep):
    record = {"date": pd.Timestamp("2024-01-02"), "predicted_state": 2,
              "predicted_regime": "risk-on", "actual_xlv_return_10d": forward,
              "actual_xlv_ret_1d": np.nan}
    if observed is not None:
        record["return_observed_at"] = observed
    update = train_module._prediction_update(record)
    assert update["$set"]["regime_label"] == "risk-on"
    assert update["$set"]["regime_id"] == 2
    assert "feature_embedding" in update["$unset"]
    assert "embedding_model" in update["$unset"]
    assert "embedding_created_at" in update["$unset"]
    assert "brief_summary" in update["$unset"]
    assert "xlv_ret_10d_actual" in update["$unset"]
    assert "actual_xlv_ret_1d" in update["$unset"]
    assert ("actual_xlv_return_10d" in update["$set"]) is keep
    assert ("return_observed_at" in update["$set"]) is keep
    assert not set(update["$set"]) & set(update["$unset"])
    assert "regime_label" not in record  # caller's DataFrame record is untouched


def test_persisted_walk_forward_refresh_is_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
    model = SimpleNamespace(transmat_=np.eye(3),
        predict_proba=lambda seq: np.tile([.1, .2, .7], (len(seq), 1)))
    monkeypatch.setattr(train_module, "train_hmm", lambda *a, **k: model)
    monkeypatch.setattr(train_module, "infer_state_labels",
                        lambda _: {0: "neutral", 1: "catalyst-fear", 2: "risk-on"})
    monkeypatch.setattr(train_module, "save_model", lambda *a: tmp_path / "unused.pkl")
    dates = pd.bdate_range("2023-11-01", "2024-01-02")
    frame = pd.DataFrame(synthetic_features(len(dates)), index=dates, columns=FEATURE_NAMES)
    collection = MagicMock()
    run_walk_forward_training({"regime_states": collection}, "2023-11-01", [2024],
                              feature_frame=frame, persist_predictions=True)
    assert collection.update_one.call_count == 2
    for call in collection.update_one.call_args_list:
        query, update = call.args
        assert query["date"] == update["$set"]["date"]
        assert update["$set"]["regime_label"] == "risk-on"
        assert {"feature_embedding", "actual_xlv_return_10d", "return_observed_at"} <= set(update["$unset"])
        assert call.kwargs == {"upsert": True}
        # Apply the operators to an old row: unrelated annotation survives,
        # while the obsolete label, vector and observed outcome do not.
        old = {"regime_label": "catalyst-fear", "feature_embedding": [1.0],
               "return_observed_at": pd.Timestamp("2024-02-01"),
               "actual_xlv_return_10d": .5, "reviewer_note": "keep"}
        old.update(update["$set"])
        for field in update["$unset"]:
            old.pop(field, None)
        assert old["reviewer_note"] == "keep" and old["regime_label"] == "risk-on"
        assert "feature_embedding" not in old and "actual_xlv_return_10d" not in old


def test_prediction_update_retains_finite_daily_return():
    update = train_module._prediction_update({
        "date": pd.Timestamp("2024-01-02"), "predicted_state": 1,
        "predicted_regime": "neutral", "actual_xlv_ret_1d": -.01})
    assert update["$set"]["actual_xlv_ret_1d"] == -.01
    assert "actual_xlv_ret_1d" not in update["$unset"]
