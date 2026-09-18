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
from datetime import datetime, date, timedelta

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
    model, scaler, state_label_map = load_model("latest")
    today = date.today().isoformat()
    today_dt = datetime.combine(date.today(), datetime.min.time())
    raw_today = build_feature_vector(today, db)

    historical = list(
        db["regime_states"]
        .find(
            {"date": {"$lt": today_dt}, "feature_vector": {"$exists": True}},
            {"_id": 0, "feature_vector": 1},
        )
        .sort("date", -1)
        .limit(60)
    )
    historical_vectors = [item["feature_vector"] for item in reversed(historical)]
    raw_sequence = np.asarray([*historical_vectors, raw_today.tolist()], dtype=float)
    if raw_sequence.ndim != 2 or raw_sequence.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Stored historical feature vectors do not match the 8-feature schema")
    scaled_sequence = scaler.transform(raw_sequence)
    regime_id, regime_label, state_probs = predict_proba_from_sequence(
        model, scaled_sequence, state_label_map
    )
    transition = compute_forward_transition_probs(
        model, np.asarray(state_probs), state_label_map=state_label_map
    )

    return {
        "regime_label": regime_label,
        "regime_id": regime_id,
        "state_probs": state_probs,
        "transition_10d": transition,
        "feature_vector": raw_today.tolist(),
        "feature_summary": dict(zip(FEATURE_NAMES, raw_today.tolist())),
        "top_pdufa_events": get_top_pdufa_events(db),
        "active_phase3_count": get_active_phase3_count(db),
        "date": today,
    }


def predict_proba_from_sequence(
    model,
    feature_sequence: np.ndarray,
    state_label_map: dict[int, str] | None = None,
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
    sequence = np.asarray(feature_sequence, dtype=float)
    if sequence.ndim != 2 or sequence.shape[1] != len(FEATURE_NAMES) or len(sequence) == 0:
        raise ValueError(f"feature_sequence must have shape (n, {len(FEATURE_NAMES)})")
    if not np.isfinite(sequence).all():
        raise ValueError("feature_sequence contains non-finite values")
    mapping = state_label_map or {0: "risk-on", 1: "neutral", 2: "catalyst-fear"}
    today_probs = np.asarray(model.predict_proba(sequence)[-1], dtype=float)
    if today_probs.shape != (3,) or not np.isclose(today_probs.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Model returned invalid state probabilities: {today_probs}")
    regime_id = int(np.argmax(today_probs))
    return regime_id, mapping[regime_id], today_probs.tolist()


def compute_forward_transition_probs(
    model,
    current_state_probs: np.ndarray,
    n_days: int = 10,
    state_label_map: dict[int, str] | None = None,
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
    transmat = np.asarray(model.transmat_, dtype=float)
    probabilities = np.asarray(current_state_probs, dtype=float)
    if transmat.shape != (3, 3) or probabilities.shape != (3,):
        raise ValueError("Expected a 3-state transition matrix and probability vector")
    if n_days < 0:
        raise ValueError("n_days cannot be negative")
    if not np.allclose(transmat.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Transition matrix rows must sum to one")
    forward = probabilities @ np.linalg.matrix_power(transmat, n_days)
    mapping = state_label_map or {0: "risk-on", 1: "neutral", 2: "catalyst-fear"}
    by_label = {label: float(forward[state]) for state, label in mapping.items()}
    return {
        "to_risk_on": by_label["risk-on"],
        "to_neutral": by_label["neutral"],
        "to_catalyst_fear": by_label["catalyst-fear"],
    }


def get_top_pdufa_events(db, top_k: int = 3) -> list[dict]:
    """
    Retrieve the next K upcoming PDUFA events from MongoDB.

    Args:
        db: pymongo Database object.
        top_k: Number of events to return.

    Returns:
        List of dicts with keys: date, company, ticker, drug, indication, review_type.
    """
    if top_k < 1:
        return []
    today_dt = datetime.combine(date.today(), datetime.min.time())
    cursor = (
        db["pdufa_events"]
        .find({"pdufa_date": {"$gte": today_dt}})
        .sort("pdufa_date", 1)
        .limit(top_k)
    )
    events = []
    for item in cursor:
        event_date = item.get("pdufa_date")
        events.append(
            {
                "date": event_date.date().isoformat() if isinstance(event_date, datetime) else str(event_date),
                "company": item.get("company_name"),
                "ticker": item.get("ticker"),
                "drug": item.get("drug_name"),
                "indication": item.get("indication"),
                "review_type": item.get("review_type", "standard"),
            }
        )
    return events


def get_active_phase3_count(db, days: int = 30) -> int:
    """
    Count Phase 3 trials with results expected within the next N days.

    Args:
        db: pymongo Database object.
        days: Look-ahead window in calendar days.

    Returns:
        Integer count of Phase 3 trials due within the window.
    """
    if days < 0:
        raise ValueError("days cannot be negative")
    today_dt = datetime.combine(date.today(), datetime.min.time())
    return int(
        db["trial_events"].count_documents(
            {
                "phase": "PHASE3",
                "status": {"$in": ["RECRUITING", "ACTIVE_NOT_RECRUITING"]},
                "primary_completion_date": {
                    "$gte": today_dt,
                    "$lte": today_dt + timedelta(days=days),
                },
            }
        )
    )
