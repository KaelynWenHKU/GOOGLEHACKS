"""
database/seed_historical.py
============================
One-time script to populate MongoDB Atlas with historical data (2015–2024).

IMPLEMENTATION STATUS: The full seed pipeline below remains a scaffold.
Only embed_stored_predictions is implemented: it enriches existing, audited
walk-forward predictions; it does not run the latest checkpoint over history.

Run order:
  1. setup_atlas_indexes.py — create indexes and vector search index first
  2. seed_historical.py     — this file

What it populates:
  - company_ticker_map:  from data/ticker_map.json (IBB holdings)
  - trial_events:        completed Phase 3 trials 2015–2024 from ClinicalTrials.gov
  - pdufa_events:        historical FDA decisions 2015–2024 from SEC EDGAR
  - regime_states:       one document per trading day 2015–2024, including:
                         feature vectors, HMM predictions, and Voyage AI embeddings

Expected run time: ~2–4 hours (dominated by Voyage AI embedding calls).
Progress is logged to stdout and checkpointed to MongoDB so the script
can be safely interrupted and resumed.

Usage:
    python -m scripts.embed_historical_regimes --start 2020-01-01 --end 2024-12-31
    (dry-run preview of the implemented embedding workflow)
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def embed_stored_predictions(db, start_date: str, end_date: str,
                             max_documents: int = 25, write: bool = False) -> dict:
    """Prepare missing analogue embeddings from persisted walk-forward rows.

    This does NOT reconstruct history or prove upstream feature provenance.
    Callers must first audit the historical inputs and persist out-of-sample
    predictions with the walk-forward trainer. Dry run reads MongoDB only.
    Explicit write mode calls Voyage (possibly billable), then updates at most
    100 existing rows. Partial writes can be resumed: existing embeddings are
    never replaced. Concurrent source changes are counted as conflicts.
    """
    import math
    from datetime import date, timezone
    from hmm.features import FEATURE_NAMES
    from database import vector_search

    def boundary(value):
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError("Date boundaries must use YYYY-MM-DD")
        return datetime.combine(date.fromisoformat(value), datetime.min.time())

    start, end = boundary(start_date), boundary(end_date)
    if end < start:
        raise ValueError("End date must not precede start date")
    if type(max_documents) is not int or not 1 <= max_documents <= 100:
        raise ValueError("max_documents must be an integer in [1, 100]")
    if type(write) is not bool:
        raise ValueError("write must be a boolean")

    collection = db["regime_states"]
    fields = ["date", "train_end_date", "predicted_regime", "regime_label",
              "feature_names", "feature_vector", "state_probs"]
    rows = list(collection.find(
        {"date": {"$gte": start, "$lte": end}, "feature_embedding": {"$exists": False}},
        {name: 1 for name in fields},
    ).sort("date", 1).limit(max_documents))
    texts = []
    seen_dates = set()
    # Validate the entire bounded batch before any possibly billable API call.
    for row in rows:
        if "_id" not in row or row.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Stored prediction requires identity and canonical feature schema")
        predicted = pd.Timestamp(row.get("date"))
        cutoff = pd.Timestamp(row.get("train_end_date"))
        if pd.isna(predicted) or pd.isna(cutoff):
            raise ValueError("Stored prediction requires date and training cutoff")
        if predicted.tzinfo:
            predicted = predicted.tz_convert("UTC").tz_localize(None)
        if cutoff.tzinfo:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        if (predicted != predicted.normalize() or not start <= predicted <= end
                or cutoff >= predicted or predicted in seen_dates):
            raise ValueError("Prediction dates must be unique daily dates after training cutoff")
        seen_dates.add(predicted)
        label = row.get("predicted_regime")
        if label not in {"risk-on", "neutral", "catalyst-fear"}:
            raise ValueError("Stored prediction requires canonical predicted_regime")
        if "regime_label" in row and row["regime_label"] != label:
            raise ValueError("Conflicting stored regime labels")
        probs = row.get("state_probs", [])
        if (len(probs) != 3 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
                or not math.isclose(sum(probs), 1, abs_tol=1e-6)):
            raise ValueError("Stored prediction requires valid state probabilities")
        # Deliberately exclude forward returns, summaries and post-event fields.
        texts.append(vector_search.build_regime_text({
            "date": predicted.date().isoformat(), "regime_label": label,
            "feature_names": row["feature_names"], "feature_vector": row["feature_vector"],
        }))
    result = {"selected": len(rows), "written": 0, "conflicts": 0, "dry_run": not write}
    if not write or not rows:
        return result
    vectors = vector_search.embed_batch(texts, "document")
    if len(vectors) != len(rows) or any(
        len(v) != vector_search.EMBEDDING_DIMS or not all(math.isfinite(x) for x in v)
        or not any(x != 0 for x in v) for v in vectors
    ):
        raise ValueError("Provider returned invalid embedding vectors")
    for row, vector in zip(rows, vectors):
        # Match source values too: never attach stale vectors after a concurrent
        # refit, or replace an embedding produced by another worker.
        guard = {"_id": row["_id"], "feature_embedding": {"$exists": False}}
        guard.update({name: row[name] if name in row else {"$exists": False} for name in fields})
        updated = collection.update_one(guard, {"$set": {
            "feature_embedding": vector, "embedding_model": vector_search.VOYAGE_MODEL,
            "embedding_created_at": datetime.now(timezone.utc),
            "regime_label": row["predicted_regime"],
        }}, upsert=False)
        result["written" if updated.matched_count else "conflicts"] += 1
    return result

# Historical range to seed
SEED_START_DATE = "2015-01-01"
SEED_END_DATE = "2024-12-31"


def seed_ticker_map(db) -> int:
    """
    Load company→ticker mapping from data/ticker_map.json into MongoDB.

    Args:
        db: pymongo Database object.

    Returns:
        Number of companies upserted.
    """
    # TODO: import company_ticker_map.build_ticker_map_collection
    # TODO: call build_ticker_map_collection(db) and return count
    raise NotImplementedError


def seed_trial_events(db, start_date: str, end_date: str) -> int:
    """
    Fetch and store completed Phase 3 trials from ClinicalTrials.gov for
    the historical date range.

    Args:
        db: pymongo Database object.
        start_date: ISO date string — earliest primary_completion_date to fetch.
        end_date: ISO date string — latest primary_completion_date to fetch.

    Returns:
        Number of trial event documents upserted.
    """
    # TODO: call clinical_trials.fetch_completed_trials(start_date, end_date, "PHASE3")
    # TODO: for each trial, call company_ticker_map.lookup_ticker_from_db() to resolve ticker
    # TODO: bulk_upsert to trial_events collection (key: nct_id)
    # TODO: log progress every 100 records
    raise NotImplementedError


def seed_pdufa_events(db, start_date: str, end_date: str) -> int:
    """
    Fetch and store historical PDUFA events from SEC EDGAR for the date range.

    Args:
        db: pymongo Database object.
        start_date: ISO date string.
        end_date: ISO date string.

    Returns:
        Number of PDUFA event documents upserted.
    """
    # TODO: call pdufa_events.run_ingestion(start_date, end_date, db)
    raise NotImplementedError


def seed_regime_states(
    db,
    start_date: str,
    end_date: str,
    model_checkpoint: str = "latest",
) -> int:
    """
    Compute and store regime state documents for every trading day in the range.

    For each day this function:
      1. Builds the 8-dim feature vector (market + MongoDB features)
      2. Classifies the regime using the trained HMM
      3. Generates a text description of the regime
      4. Calls Voyage AI to embed the description (1024 dims)
      5. Upserts the full document to regime_states collection

    Checkpointing: skips days where a regime_states document already exists.
    This allows safe resume after interruption.

    Args:
        db: pymongo Database object.
        start_date: ISO date string.
        end_date: ISO date string.
        model_checkpoint: HMM checkpoint to use for classification.

    Returns:
        Number of new regime_state documents upserted.
    """
    # TODO: generate list of trading dates using pd.bdate_range(start_date, end_date)
    # TODO: for each date, check if regime_states doc already exists (skip if yes)
    # TODO: call features.build_feature_vector(date, db)
    # TODO: load HMM model, classify regime via predict.predict_proba_from_sequence()
    # TODO: call vector_search.build_regime_text() and embed_regime_document()
    # TODO: assemble full document matching REGIME_STATE_SCHEMA
    # TODO: upsert to regime_states (filter: {date: date})
    # TODO: log progress every 50 dates
    raise NotImplementedError


def main() -> None:
    """
    Run the full historical data seeding pipeline.

    Populates all four collections in the correct order:
      1. company_ticker_map (fast — from local JSON)
      2. trial_events       (medium — ClinicalTrials.gov API)
      3. pdufa_events       (medium — SEC EDGAR)
      4. regime_states      (slow — HMM + Voyage AI embeddings)
    """
    from database.mongo_client import get_db

    print("Starting historical data seeding pipeline...")
    print(f"Date range: {SEED_START_DATE} → {SEED_END_DATE}")
    db = get_db()

    print("\n[1/4] Seeding company ticker map...")
    n = seed_ticker_map(db)
    print(f"      → {n} companies upserted")

    print("\n[2/4] Seeding trial events...")
    n = seed_trial_events(db, SEED_START_DATE, SEED_END_DATE)
    print(f"      → {n} trial events upserted")

    print("\n[3/4] Seeding PDUFA events...")
    n = seed_pdufa_events(db, SEED_START_DATE, SEED_END_DATE)
    print(f"      → {n} PDUFA events upserted")

    print("\n[4/4] Seeding regime states (this will take a while)...")
    n = seed_regime_states(db, SEED_START_DATE, SEED_END_DATE)
    print(f"      → {n} regime state documents upserted")

    print("\nSeeding complete!")


if __name__ == "__main__":
    main()
