"""Tests for robust HMM fitting, labeling, persistence, and walk-forward use."""

from types import SimpleNamespace

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
