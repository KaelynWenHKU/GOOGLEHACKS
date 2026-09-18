"""Tests for online HMM regime probability and transition calculations."""

from types import SimpleNamespace

import numpy as np
import pytest

from hmm.predict import compute_forward_transition_probs, predict_proba_from_sequence


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

