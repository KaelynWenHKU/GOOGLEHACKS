"""
agent/tools.py
===============
Google ADK tool definitions for the HealthQuant agent.

Three tools are registered with the ADK agent (Section 9):
  1. get_current_regime       — HMM regime classification for today
  2. find_historical_analogues — Atlas Vector Search for similar past periods
  3. get_upcoming_catalysts    — PDUFA + Phase 3 events in the next N days

Each function follows the ADK tool contract:
  - Returns a JSON-serialisable dict
  - All exceptions are caught and returned as {"error": str} so the
    agent can gracefully handle data availability issues

Usage:
    from agent.tools import get_current_regime, find_historical_analogues, get_upcoming_catalysts
    # Pass these callables to the ADK agent's tools list
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from database.mongo_client import get_db
from database.vector_search import find_analogues_from_feature_vector
from hmm.predict import classify_current_regime, get_top_pdufa_events, get_active_phase3_count

logger = logging.getLogger(__name__)


def get_current_regime() -> dict[str, Any]:
    """
    Fetch today's feature vector, run it through the HMM, and return the
    current healthcare market regime classification.

    This is Tool 1 of the HealthQuant agent. It should always be called first
    in the investment brief workflow to establish the current regime context.

    Returns:
        {
            "regime_label": str,         # "risk-on" | "neutral" | "catalyst-fear"
            "regime_id": int,            # 0 | 1 | 2
            "state_probs": list[float],  # probability over all 3 states
            "transition_10d": dict,      # 10-day forward transition probabilities
            "feature_summary": dict,     # human-readable feature values with labels
            "feature_vector": list[float], # raw 8-dim vector (for Tool 2 input)
            "top_pdufa_events": list,    # next 3 PDUFA dates in the universe
            "active_phase3_count": int,  # Phase 3 trials with results due in 30d
            "date": str                  # ISO date string (today)
        }
        On error: {"error": str, "date": str}
    """
    # TODO: db = get_db()
    # TODO: call classify_current_regime(db) from hmm/predict.py
    # TODO: format feature_summary as {feature_name: formatted_value} for readability
    # TODO: wrap in try/except and return {"error": str(e)} on failure
    raise NotImplementedError


def find_historical_analogues(
    feature_vector: list[float],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Embed the current feature vector and search MongoDB Atlas Vector Search
    for the most similar historical market periods.

    This is Tool 2. Call it with the feature_vector returned by get_current_regime().

    Args:
        feature_vector: 8-dimensional list of raw feature values from Tool 1.
        top_k: Number of analogues to return (default 3).

    Returns:
        {
            "analogues": [
                {
                    "date": str,
                    "regime_label": str,
                    "similarity_score": float,  # cosine similarity in [0, 1]
                    "description": str,         # brief_summary from the document
                    "transition_probs_10d": dict,
                    "key_events": list[str]     # notable events in that period
                }
            ]
        }
        On error: {"error": str, "analogues": []}
    """
    # TODO: db = get_db()
    # TODO: call vector_search.find_analogues_from_feature_vector(db, feature_vector, ...)
    # TODO: format each analogue: convert datetime to ISO string, rename score field
    # TODO: wrap in try/except
    raise NotImplementedError


def get_upcoming_catalysts(days: int = 30) -> dict[str, Any]:
    """
    Query MongoDB for PDUFA events and Phase 3 trial completions within
    the next N calendar days, sorted by estimated market impact.

    This is Tool 3. Use it to identify near-term event risks that should
    inform the Sector Positioning and Risk Flag sections of the brief.

    Args:
        days: Look-ahead window in calendar days (default 30).

    Returns:
        {
            "pdufa_events": [
                {
                    "date": str,
                    "company": str,
                    "ticker": str,
                    "drug": str,
                    "indication": str,
                    "review_type": str,           # "priority" | "standard"
                    "historical_reaction": dict   # avg stock move for past events
                }
            ],
            "phase3_completions": [
                {
                    "expected_date": str,
                    "company": str,
                    "ticker": str,
                    "trial_id": str,
                    "condition": str,
                    "enrollment": int
                }
            ],
            "total_catalyst_density_score": float,  # normalised 0–1 pressure score
            "window_days": int
        }
        On error: {"error": str, "pdufa_events": [], "phase3_completions": []}
    """
    # TODO: db = get_db()
    # TODO: query pdufa_events: pdufa_date in [now, now + days], sorted ascending
    # TODO: query trial_events: PHASE3 + ACTIVE, primary_completion_date in window
    # TODO: for each PDUFA event, look up historical_reaction from trial_events
    # TODO: compute total_catalyst_density_score:
    #   score = min(1.0, pdufa_count / 10) * 0.7 + min(1.0, phase3_count / 5) * 0.3
    # TODO: wrap in try/except
    raise NotImplementedError
