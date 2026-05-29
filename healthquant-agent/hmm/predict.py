"""
hmm/predict.py
===============
Regime classification using a trained GaussianHMM.

Given today's 8-dim feature vector, classifies the current healthcare
market regime and computes 10-day forward transition probabilities.

Typical usage:
    from hmm.predict import classify_current_regime
    result = classify_current_regime(db)
"""

import logging
from datetime import datetime, date

import numpy as np

from hmm.train import load_model
from hmm.features import build_feature_vector, FEATURE_NAMES

logger = logging.getLogger(__name__)


def classify_current_regime(db) -> dict:
    """
    Classify today's healthcare market regime using the latest trained HMM.

    Steps:
      1. Load the latest model checkpoint from disk
      2. Build today's 8-dim feature vector from MongoDB + yfinance
      3. Run the Viterbi algorithm to get state probabilities
      4. Compute 10-day forward transition probabilities via matrix power
      5. Return structured result dict

    Args:
        db: pymongo Database object (healthquant).

    Returns:
        Dict matching the get_current_regime tool output schema (Section 9):
        {
            regime_label, regime_id, state_probs, transition_10d,
            feature_summary, top_pdufa_events, active_phase3_count, date
        }
    """
    # TODO: load model, scaler, state_label_map via load_model("latest")
    # TODO: compute today's raw feature vector via build_feature_vector(today, db)
    # TODO: scale using scaler.transform()
    # TODO: call predict_proba_from_sequence() for state probabilities
    # TODO: call compute_forward_transition_probs() for 10-day outlook
    # TODO: call get_top_pdufa_events() and get_active_phase3_count()
    # TODO: return structured dict
    raise NotImplementedError


def predict_proba_from_sequence(
    model,
    feature_sequence: np.ndarray,
) -> tuple[int, str, list[float]]:
    """
    Run the HMM forward algorithm on a feature sequence and return the
    current state probabilities.

    Args:
        model: Fitted GaussianHMM.
        feature_sequence: Array of shape (T, 8) — standardised feature history.
                          Must be long enough for reliable state inference
                          (minimum ~20 trading days recommended).

    Returns:
        Tuple of (most_likely_state_id, regime_label, state_probs_list).
        state_probs_list is a list of 3 floats summing to 1.0.
    """
    # TODO: model.predict_proba(feature_sequence) → shape (T, 3)
    # TODO: today_probs = result[-1]  (last row = current state distribution)
    # TODO: regime_id = int(np.argmax(today_probs))
    # TODO: return (regime_id, state_label_map[regime_id], today_probs.tolist())
    raise NotImplementedError


def compute_forward_transition_probs(
    model,
    current_state_probs: np.ndarray,
    n_days: int = 10,
) -> dict[str, float]:
    """
    Compute the N-day forward regime probability distribution via matrix power.

    Formula (from Section 7.4):
        forward_probs = current_state_probs @ transmat^n_days

    Args:
        model: Fitted GaussianHMM (provides transmat_).
        current_state_probs: Array of shape (3,) — today's state distribution.
        n_days: Forward horizon in trading days (default 10).

    Returns:
        Dict with keys "to_risk_on", "to_neutral", "to_catalyst_fear",
        each mapping to a probability float in [0, 1].
    """
    # TODO: transmat = model.transmat_  (shape 3×3)
    # TODO: forward = current_state_probs @ np.linalg.matrix_power(transmat, n_days)
    # TODO: return {"to_risk_on": forward[0], "to_neutral": forward[1], "to_catalyst_fear": forward[2]}
    raise NotImplementedError


def get_top_pdufa_events(db, top_k: int = 3) -> list[dict]:
    """
    Retrieve the next K upcoming PDUFA events from MongoDB.

    Args:
        db: pymongo Database object.
        top_k: Number of events to return.

    Returns:
        List of dicts with keys: date, company, ticker, drug, indication, review_type.
    """
    # TODO: query pdufa_events where pdufa_date >= today, sort ascending, limit top_k
    # TODO: format each document as a clean dict (no ObjectId)
    raise NotImplementedError


def get_active_phase3_count(db, days: int = 30) -> int:
    """
    Count Phase 3 trials with results expected within the next N days.

    Args:
        db: pymongo Database object.
        days: Look-ahead window in calendar days.

    Returns:
        Integer count of Phase 3 trials due within the window.
    """
    # TODO: query trial_events: phase=PHASE3, status in [RECRUITING, ACTIVE_NOT_RECRUITING]
    # TODO: primary_completion_date in [today, today + days]
    # TODO: return count
    raise NotImplementedError
