"""Fictional, fixed fixtures for exploring the UI without external services."""
from copy import deepcopy
from hmm.features import FEATURE_NAMES


def sample_snapshot() -> dict:
    """Return a fresh copy so one browser session cannot modify another."""
    vector = [0.012, 0.009, 0.004, 0.57, 2, 1, 420, 0.88]
    return deepcopy({
        "mode": "Sample data", "loaded_at": "2026-09-21T00:00:00+00:00",
        "regime": {"status": "ok", "date": "2026-09-21T00:00:00+00:00",
            "regime_label": "neutral", "state_probs": [0.22, 0.63, 0.15],
            "state_label_map": {"0": "risk-on", "1": "neutral", "2": "catalyst-fear"},
            "transition_10d": {"to_risk_on": 0.28, "to_neutral": 0.53, "to_catalyst_fear": 0.19},
            "feature_vector": vector, "feature_summary": dict(zip(FEATURE_NAMES, vector)),
            "source": "Fictional demonstration fixture"},
        "catalysts": {"status": "ok", "window_days": 30,
            "coverage_note": "Fictional companies and events for interface demonstration only.",
            "pdufa_events": [
                {"date": "2026-09-28", "company": "Example Therapeutics A", "ticker": "DEMO-A", "drug": "Candidate A", "review_type": "priority"},
                {"date": "2026-10-09", "company": "Example Therapeutics B", "ticker": "DEMO-B", "drug": "Candidate B", "review_type": "standard"}],
            "phase3_completions": [{"expected_date": "2026-10-15", "company": "Example Research C", "ticker": "DEMO-C", "trial_id": "SAMPLE-003", "condition": "Example indication"}]},
        "analogues": {"status": "ok", "analogues": [
            {"date": "2024-03-18", "regime_label": "neutral", "similarity_score": 0.91, "xlv_ret_10d_actual": 0.008, "description": "Fictional comparison with moderate volatility."},
            {"date": "2023-08-07", "regime_label": "risk-on", "similarity_score": 0.87, "xlv_ret_10d_actual": 0.019, "description": "Fictional comparison with stronger sector momentum."},
            {"date": "2022-11-14", "regime_label": "neutral", "similarity_score": 0.82, "xlv_ret_10d_actual": -0.006, "description": "Fictional comparison with mixed price signals."}]},
    })


SAMPLE_BRIEF = """## Current Regime
This fictional example shows a neutral regime with a 63% state probability.
These values illustrate the interface; they were not produced by a trained model.

## Historical Analogues
The three example cards demonstrate similarity scores and forward-return displays.
All dates and return values here are synthetic examples, not measured performance.

## Upcoming Catalysts (Next 30 Days)
Two fictional FDA dates and one fictional Phase 3 completion demonstrate the calendar.

## Sector Positioning
No investment conclusion can be drawn from sample data. Switch to Live data and
connect verified evidence before requesting a Gemini brief.
"""
