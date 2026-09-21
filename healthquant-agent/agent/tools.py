"""Read-only evidence tools shared by Streamlit and the Gemini ADK agent.

Failures expose actionable messages, never provider exceptions or credentials.
Absent evidence remains absent rather than being treated as zero.
"""
from datetime import datetime, timezone, timedelta
import logging
import math

from database.mongo_client import get_db
from hmm.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _failure(operation: str, exc: Exception, message: str) -> dict:
    """Log only exception type; provider text may include credentials."""
    logger.warning("%s failed (%s)", operation, type(exc).__name__)
    return {"status": "unavailable", "error": message}


def get_current_regime() -> dict:
    """Read the latest persisted HMM prediction, preserving its as-of date.

    This read does not retrain the model. The data pipeline must publish
    regime_states first. Predictions older than four days are marked stale.
    """
    try:
        doc = get_db()["regime_states"].find_one(
            {"date": {"$lte": datetime.now(timezone.utc).replace(tzinfo=None)}},
            sort=[("date", -1)], projection={"_id": 0, "feature_embedding": 0})
        if not doc:
            return {"status": "unavailable", "error": "No stored HMM predictions. Run the training/data pipeline first."}
        vector = doc.get("feature_vector", [])
        if len(vector) != 8 or not all(math.isfinite(v) for v in vector):
            raise ValueError("Missing features")
        label = doc.get("regime_label", doc.get("predicted_regime"))
        if label not in {"risk-on", "neutral", "catalyst-fear"}:
            raise ValueError("Invalid label")
        timestamp = doc["date"]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        probabilities = doc.get("state_probs", [])
        if len(probabilities) != 3 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
            raise ValueError("Invalid state probabilities")
        return {"status": "ok", "date": timestamp.isoformat(),
                "stale": (datetime.now(timezone.utc) - timestamp).total_seconds() > 4 * 86400,
                "regime_label": label, "state_probs": probabilities,
                "state_label_map": {str(k): v for k, v in doc.get("state_label_map", {}).items()},
                "feature_vector": vector, "feature_summary": dict(zip(FEATURE_NAMES, vector)),
                "transition_10d": doc.get("transition_probs_10d", doc.get("transition_10d", {})),
                "source": "MongoDB / persisted HMM prediction"}
    except Exception as exc:
        return _failure("Regime", exc, "Regime data unavailable. Check MongoDB configuration and stored HMM prediction fields.")


def find_historical_analogues(feature_vector: list[float], top_k: int = 3, as_of: str = "") -> dict:
    """Retrieve up to ten historical analogues for eight raw feature values."""
    try:
        from database.vector_search import find_analogues_from_feature_vector
        if len(feature_vector) != 8 or not all(math.isfinite(x) for x in feature_vector) or not 1 <= top_k <= 10:
            return {"status": "unavailable", "error": "Provide eight finite features and top_k from 1 to 10."}
        cutoff = datetime.fromisoformat(as_of) if as_of else datetime.now(timezone.utc)
        if cutoff.tzinfo:
            cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
        matches = find_analogues_from_feature_vector(get_db(), feature_vector, FEATURE_NAMES,
                                                    "unclassified", top_k, cutoff)
        analogues = []
        for doc in matches:
            # Realized returns need a recorded availability date to be shown.
            observed = doc.get("return_observed_at")
            if observed and observed.tzinfo:
                observed = observed.astimezone(timezone.utc).replace(tzinfo=None)
            actual = doc.get("xlv_ret_10d_actual", doc.get("actual_xlv_return_10d"))
            if not observed or observed > cutoff:
                actual = None
            analogues.append({"date": doc["date"].isoformat(), "regime_label": doc.get("regime_label"),
                "similarity_score": doc.get("score"), "description": doc.get("brief_summary", ""),
                "xlv_ret_10d_actual": actual, "key_events": doc.get("key_events", [])})
        return {"status": "ok", "analogues": analogues, "source": "MongoDB Atlas Vector Search / Voyage"}
    except Exception as exc:
        return _failure("Analogues", exc, "Analogue search unavailable. Check Voyage credentials, stored embeddings and the Atlas vector index.")


def get_upcoming_catalysts(days: int = 30) -> dict:
    """Read stored PDUFA dates and active Phase 3 completions in a date window."""
    if not 1 <= days <= 90:
        return {"status": "unavailable", "error": "Choose a catalyst window between 1 and 90 days."}
    try:
        db = get_db()
        today = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        window = {"$gte": today, "$lte": today + timedelta(days=days)}
        pdufa = list(db["pdufa_events"].find({"pdufa_date": window}, {"_id": 0}).sort("pdufa_date", 1).limit(100))
        trials = list(db["trial_events"].find({"phase": "PHASE3", "status": {"$in": ["RECRUITING", "ACTIVE_NOT_RECRUITING"]},
                     "primary_completion_date": window}, {"_id": 0}).sort("primary_completion_date", 1).limit(100))
        return {"status": "ok", "window_days": days, "source": "MongoDB / stored catalyst calendar",
            "coverage_note": "Stored records only; completeness and source freshness are not verified. At most 100 events per type.",
            "pdufa_events": [{"date": d["pdufa_date"].isoformat(), "company": d.get("company_name"),
                "ticker": d.get("ticker"), "drug": d.get("drug_name"), "indication": d.get("indication"),
                "review_type": d.get("review_type", "standard")} for d in pdufa],
            "phase3_completions": [{"expected_date": d["primary_completion_date"].isoformat(),
                "company": d.get("company_name"), "ticker": d.get("ticker"), "trial_id": d.get("nct_id"),
                "condition": d.get("condition"), "enrollment": d.get("enrollment_count")} for d in trials]}
    except Exception as exc:
        return _failure("Catalysts", exc, "Catalyst calendar unavailable. Check MongoDB credentials and connectivity.")


def collect_evidence() -> dict:
    """Collect one snapshot; no Gemini request is made here."""
    regime = get_current_regime()
    analogues = (find_historical_analogues(regime["feature_vector"], as_of=regime["date"]) if regime.get("status") == "ok"
                 else {"status": "unavailable", "error": "A valid regime feature vector is needed for analogue search."})
    return {"regime": regime, "analogues": analogues, "catalysts": get_upcoming_catalysts(),
            "loaded_at": datetime.now(timezone.utc).isoformat(), "mode": "Live data"}
