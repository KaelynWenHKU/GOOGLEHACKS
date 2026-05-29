"""
database/schema.py
===================
MongoDB collection schema definitions and index creation helpers.

Defines the four collections used by HealthQuant Agent (Section 5):
  - regime_states:     one document per trading day
  - trial_events:      one document per material clinical trial event
  - pdufa_events:      one document per FDA decision date
  - company_ticker_map: sponsor name → ticker lookup

Run setup_all_indexes(db) once after cluster creation to create
standard (non-vector) indexes. The Atlas Vector Search index must
be created via the Atlas UI or scripts/setup_atlas_indexes.py.
"""

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database


# ---------------------------------------------------------------------------
# Collection name constants — use these everywhere, never hard-code strings
# ---------------------------------------------------------------------------
COLLECTION_REGIME_STATES = "regime_states"
COLLECTION_TRIAL_EVENTS = "trial_events"
COLLECTION_PDUFA_EVENTS = "pdufa_events"
COLLECTION_TICKER_MAP = "company_ticker_map"


# ---------------------------------------------------------------------------
# Schema reference dicts — used as templates and for documentation
# These are NOT enforced by MongoDB (schemaless), but serve as the
# authoritative field reference for all ingestion and query code.
# ---------------------------------------------------------------------------

REGIME_STATE_SCHEMA: dict[str, Any] = {
    # Primary key — one document per trading day
    "date": datetime,                   # ISODate
    "regime_label": str,                # "risk-on" | "neutral" | "catalyst-fear"
    "regime_id": int,                   # 0 | 1 | 2
    "state_probs": list,                # [float, float, float] summing to 1.0
    "feature_vector": list,             # [float × 8] — raw un-standardised values
    "feature_names": list,              # list of 8 strings matching FEATURE_NAMES
    "feature_embedding": list,          # [float × 1024] — Voyage AI embedding
    "transition_probs_10d": dict,       # {"to_risk_on": float, ...}
    "brief_summary": str,               # Gemini-generated summary text
    "upcoming_events_count": int,       # PDUFA + Phase 3 events in next 30d
    "created_at": datetime,             # ISODate — document creation timestamp
}

TRIAL_EVENT_SCHEMA: dict[str, Any] = {
    "nct_id": str,                      # ClinicalTrials.gov identifier
    "company_name": str,
    "ticker": str,                      # resolved stock ticker
    "drug_name": str,
    "phase": str,                       # "PHASE2" | "PHASE3" | "PHASE4"
    "condition": str,                   # primary disease/condition
    "therapeutic_area": str,            # normalised area (e.g. "oncology")
    "primary_completion_date": datetime,
    "enrollment_count": int,
    "enrollment_velocity": float,       # patients enrolled per month
    "status": str,                      # "RECRUITING" | "ACTIVE_NOT_RECRUITING" | "COMPLETED"
    "outcome": str,                     # "approval" | "crl" | "failed" | null (pending)
    "stock_reaction": dict,             # {"t_plus_1_pct": float, ...}
    "regime_at_event": str,             # regime label at the time of completion
    "is_historical": bool,
    "data_source": str,                 # "clinicaltrials_gov"
    "last_updated": datetime,
}

PDUFA_EVENT_SCHEMA: dict[str, Any] = {
    "pdufa_date": datetime,
    "company_name": str,
    "ticker": str,
    "drug_name": str,
    "indication": str,
    "therapeutic_area": str,
    "review_type": str,                 # "standard" | "priority"
    "priority_review": bool,
    "application_type": str,            # "NDA" | "BLA" | "sNDA"
    "outcome": str,                     # null until FDA decision date passes
    "data_source": str,                 # "sec_8k" | "rttnews"
    "sec_filing_url": str,
    "created_at": datetime,
}

TICKER_MAP_SCHEMA: dict[str, Any] = {
    "canonical_name": str,              # e.g. "Biogen Inc."
    "ticker": str,                      # e.g. "BIIB"
    "exchange": str,                    # "NASDAQ" | "NYSE"
    "aliases": list,                    # list of known name variants
    "therapeutic_areas": list,          # list of strings
    "in_ibb": bool,                     # is this ticker in the IBB ETF?
    "in_xlv": bool,                     # is this ticker in the XLV ETF?
    "market_cap_tier": str,             # "large" | "mid" | "small"
}


def setup_all_indexes(db: Database) -> None:
    """
    Create all standard MongoDB indexes for the healthquant database.

    These are regular (non-vector) indexes. The Atlas Vector Search index
    on regime_states.feature_embedding must be created separately via:
      - Atlas UI (recommended for first setup), OR
      - scripts/setup_atlas_indexes.py (programmatic)

    Args:
        db: pymongo Database object (healthquant).
    """
    print("Creating indexes for regime_states...")
    # TODO: db.regime_states.create_index([("date", DESCENDING)], unique=True)

    print("Creating indexes for trial_events...")
    # TODO: db.trial_events.create_index([("primary_completion_date", ASCENDING)])
    # TODO: db.trial_events.create_index([("ticker", ASCENDING)])
    # TODO: db.trial_events.create_index([("phase", ASCENDING), ("status", ASCENDING)])
    # TODO: db.trial_events.create_index([("nct_id", ASCENDING)], unique=True)

    print("Creating indexes for pdufa_events...")
    # TODO: db.pdufa_events.create_index([("pdufa_date", ASCENDING)])
    # TODO: db.pdufa_events.create_index([("ticker", ASCENDING)])

    print("Creating indexes for company_ticker_map...")
    # TODO: db.company_ticker_map.create_index([("ticker", ASCENDING)], unique=True)
    # TODO: db.company_ticker_map.create_index([("canonical_name", ASCENDING)])

    print("All standard indexes created.")


def validate_document(document: dict, schema: dict) -> list[str]:
    """
    Check a document against a schema reference dict and report missing fields.

    This is a lightweight validation helper — not a full JSON Schema validator.
    Used in ingestion pipelines to catch data quality issues early.

    Args:
        document: The document dict to validate.
        schema: Reference schema dict (e.g. REGIME_STATE_SCHEMA).

    Returns:
        List of field names that are present in the schema but missing from
        the document. Empty list means validation passed.
    """
    # TODO: return [field for field in schema if field not in document]
    raise NotImplementedError
