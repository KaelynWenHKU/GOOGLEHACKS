"""Tests for online HMM regime probability and transition calculations."""

from types import SimpleNamespace
from itertools import permutations

import numpy as np
import pytest

from hmm.predict import compute_forward_transition_probs, predict_proba_from_sequence

TEST_MAP = {0: "risk-on", 1: "neutral", 2: "catalyst-fear"}


class PosteriorModel:
    """Small deterministic stand-in for hmmlearn's posterior method."""

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        probabilities = np.tile([0.1, 0.2, 0.7], (len(sequence), 1))
        return probabilities


def test_predict_probability_uses_supplied_semantic_map() -> None:
    mapping = {0: "neutral", 1: "catalyst-fear", 2: "risk-on"}
    regime_id, label, probabilities = predict_proba_from_sequence(
        PosteriorModel(), np.zeros((20, 8)), mapping
    )
    assert regime_id == 2
    assert label == "risk-on"
    assert probabilities == [0.1, 0.2, 0.7]


def test_forward_probabilities_are_mapped_by_semantic_label() -> None:
    model = SimpleNamespace(
        transmat_=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
    )
    mapping = {0: "neutral", 1: "catalyst-fear", 2: "risk-on"}
    result = compute_forward_transition_probs(
        model,
        np.array([1.0, 0.0, 0.0]),
        n_days=1,
        state_label_map=mapping,
    )
    assert result == {
        "to_risk_on": 0.1,
        "to_neutral": 0.8,
        "to_catalyst_fear": 0.1,
    }


def test_invalid_prediction_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        predict_proba_from_sequence(PosteriorModel(), np.zeros((8,)))
    bad_model = SimpleNamespace(transmat_=np.ones((3, 3)))
    with pytest.raises(ValueError, match="sum to one"):
        compute_forward_transition_probs(bad_model, np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("probabilities", [
    [-0.1, 0.4, 0.7], [1.1, -0.1, 0.0], [np.nan, 0.0, 1.0],
    [np.inf, -np.inf, 1.0], [0.2, 0.2, 0.2],
])
def test_invalid_distributions_rejected_by_prediction_and_forecast(probabilities):
    """Even vectors that sum to one must not contain negative probabilities."""
    model = SimpleNamespace(
        transmat_=np.eye(3),
        predict_proba=lambda sequence: np.tile(probabilities, (len(sequence), 1)),
    )
    with pytest.raises(ValueError):
        predict_proba_from_sequence(model, np.zeros((20, 8)), TEST_MAP)
    with pytest.raises(ValueError):
        compute_forward_transition_probs(model, probabilities)
    model.transmat_[0] = probabilities
    with pytest.raises(ValueError):
        compute_forward_transition_probs(model, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("mapping", [
    {}, {0: "risk-on", 1: "risk-on", 2: "catalyst-fear"},
    None, {False: "risk-on", 1: "neutral", 2: "catalyst-fear"},
    {0.0: "risk-on", 1.0: "neutral", 2.0: "catalyst-fear"},
    {0: "risk-on", 1: "neutral", 3: "catalyst-fear"},
    {0: "risk-on", 1: "neutral", 2: "unknown"},
])
def test_invalid_semantic_maps_are_rejected(mapping):
    with pytest.raises(ValueError, match="state_label_map"):
        predict_proba_from_sequence(PosteriorModel(), np.zeros((20, 8)), mapping)
    with pytest.raises(ValueError, match="state_label_map"):
        compute_forward_transition_probs(SimpleNamespace(transmat_=np.eye(3)),
                                         [0.1, 0.2, 0.7], state_label_map=mapping)


@pytest.mark.parametrize("horizon", [-1, 1.5, True, np.nan])
def test_forecast_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="non-negative integer"):
        compute_forward_transition_probs(SimpleNamespace(transmat_=np.eye(3)),
                                         [0.1, 0.2, 0.7], n_days=horizon)


def test_zero_day_forecast_preserves_current_probabilities():
    result = compute_forward_transition_probs(SimpleNamespace(transmat_=np.eye(3)),
                                              [0.1, 0.2, 0.7], n_days=0, state_label_map=TEST_MAP)
    assert result == {"to_risk_on": 0.1, "to_neutral": 0.2, "to_catalyst_fear": 0.7}


@pytest.mark.parametrize("shape", [(3,), (0, 3), (19, 3), (20, 4)])
def test_model_posterior_shape_is_checked(shape):
    model = SimpleNamespace(predict_proba=lambda sequence: np.zeros(shape))
    with pytest.raises(ValueError, match="posterior.*shape"):
        predict_proba_from_sequence(model, np.zeros((20, 8)), TEST_MAP)


@pytest.mark.parametrize("labels", list(permutations(TEST_MAP.values())))
def test_all_label_permutations_preserve_state_probabilities(labels):
    """A retraining permutation must change names, never the underlying masses."""
    mapping = dict(enumerate(labels))
    state, label, probabilities = predict_proba_from_sequence(
        PosteriorModel(), np.zeros((20, 8)), mapping)
    assert state == 2 and label == mapping[2]
    result = compute_forward_transition_probs(SimpleNamespace(transmat_=np.eye(3)),
                                              probabilities, n_days=0, state_label_map=mapping)
    assert result == {f"to_{name.replace('-', '_')}": probabilities[i]
                      for i, name in mapping.items()}
