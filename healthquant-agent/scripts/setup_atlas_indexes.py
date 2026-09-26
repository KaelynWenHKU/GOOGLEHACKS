"""Idempotent Atlas index setup; never silently replace an existing definition.

Run from healthquant-agent/: python -m scripts.setup_atlas_indexes
Search indexes are asynchronous; success requires READY and queryable.
"""
import time
from pymongo.operations import SearchIndexModel


VECTOR_DEFINITION = {"fields": [
    {"type": "vector", "path": "feature_embedding", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "date"},
    {"type": "filter", "path": "regime_label"},
]}


def setup_standard_indexes(db) -> None:
    """Create identity and calendar indexes without deleting existing data."""
    from database.schema import setup_all_indexes
    setup_all_indexes(db)


def _ensure_index(collection, name, definition, kind, timeout_seconds, poll_seconds):
    """Reuse matching definitions, reject conflicts, and bound readiness polling."""
    if timeout_seconds < 0 or not 0 <= poll_seconds <= 30:
        raise ValueError("Invalid polling limits")
    existing = list(collection.list_search_indexes(name))
    if existing:
        item = existing[0]
        if item.get("type", "search") != kind or item.get("latestDefinition") != definition:
            raise ValueError(f"Existing {name} definition differs; review it manually")
    else:
        collection.create_search_index(model=SearchIndexModel(
            name=name, definition=definition, type=kind))
    deadline = time.monotonic() + timeout_seconds
    while True:
        items = list(collection.list_search_indexes(name))
        if items:
            item = items[0]
            if item.get("status") == "FAILED":
                raise RuntimeError(f"Atlas index {name} failed to build")
            if item.get("status") == "READY" and item.get("queryable") is True:
                return name
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Atlas index {name} is not ready; rerun to check later")
        time.sleep(poll_seconds)


def setup_vector_search_index(collection, timeout_seconds=300, poll_seconds=5) -> str:
    """Create/reuse a 1024-dimensional Voyage index with historical date filters."""
    return _ensure_index(collection, "regime_vector_index", VECTOR_DEFINITION,
                         "vectorSearch", timeout_seconds, poll_seconds)


def setup_text_search_index(collection, timeout_seconds=300, poll_seconds=5) -> str:
    """Create/reuse the trial text search index."""
    return _ensure_index(collection, "trial_text_index", {"mappings": {"dynamic": True}},
                         "search", timeout_seconds, poll_seconds)


def main() -> None:
    """Configure only the database selected by local environment settings."""
    from database.mongo_client import get_db
    try:
        db = get_db()
        setup_standard_indexes(db)
        for setup, name in [(setup_vector_search_index, "regime_states"),
                            (setup_text_search_index, "trial_events")]:
            print(f"Waiting for {name} search index...", flush=True)
            print(f"Ready: {setup(db[name])}", flush=True)
    except Exception as exc:
        # Provider errors may include credentials or full topology details.
        print(f"Index setup incomplete ({type(exc).__name__}); inspect Atlas status/configuration.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
