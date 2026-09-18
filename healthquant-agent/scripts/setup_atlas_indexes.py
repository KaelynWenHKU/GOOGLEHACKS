"""
scripts/setup_atlas_indexes.py
================================
One-time setup script to create all MongoDB Atlas indexes for HealthQuant.

Run this BEFORE seed_historical.py.

Creates:
  1. Standard (non-vector) indexes via pymongo — fast, runs immediately
  2. Atlas Vector Search index on regime_states.feature_embedding — takes 1–5 minutes

Vector Search index config:
  Collection: healthquant.regime_states
  Field: feature_embedding
  Dimensions: 1024 (Voyage AI voyage-3-large)
  Similarity: cosine

Usage:
    python scripts/setup_atlas_indexes.py

After running, verify in Atlas UI:
  Your cluster → Atlas Search → regime_vector_index should show "Active"
"""

import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()


def setup_standard_indexes(db) -> None:
    """
    Create all standard (non-vector) MongoDB indexes.

    Args:
        db: pymongo Database object (healthquant).
    """
    from database.schema import setup_all_indexes
    setup_all_indexes(db)


def setup_vector_search_index(collection) -> str:
    """
    Create the Atlas Vector Search index on regime_states.feature_embedding.

    Uses the pymongo SearchIndexModel API (requires pymongo >= 4.7.0).
    The index takes 1–5 minutes to become queryable on Atlas free tier.

    Args:
        collection: pymongo Collection object for regime_states.

    Returns:
        Name of the created index.
    """
    # TODO: construct SearchIndexModel with:
    #   type="vectorSearch"
    #   name="regime_vector_index"
    #   definition: numDimensions=1024, similarity="cosine", path="feature_embedding"
    #   filter fields: date, regime_label
    # TODO: collection.create_search_index(model=search_index_model)
    # TODO: poll collection.list_search_indexes() until status == "READY"
    # TODO: return index name
    raise NotImplementedError


def setup_text_search_index(collection) -> str:
    """
    Create an Atlas Search (Lucene) full-text index on trial_events.

    Enables free-text search on condition, drug_name, and company_name fields.
    Used by the agent when looking up trials by disease area.

    Args:
        collection: pymongo Collection object for trial_events.

    Returns:
        Name of the created index.
    """
    # TODO: create Atlas Search index with dynamic mapping on trial_events
    # TODO: index name: "trial_text_index"
    raise NotImplementedError


def main() -> None:
    """Run the complete Atlas index setup."""
    from database.mongo_client import get_db

    print("Setting up MongoDB Atlas indexes for HealthQuant...")
    db = get_db()

    print("\n[1/3] Creating standard indexes...")
    setup_standard_indexes(db)
    print("      ✓ Standard indexes created")

    print("\n[2/3] Creating Atlas Vector Search index on regime_states...")
    print("      (This takes 1–5 minutes on Atlas free tier)")
    # TODO: index_name = setup_vector_search_index(db["regime_states"])
    # TODO: print(f"      ✓ Vector Search index '{index_name}' is ACTIVE")
    print("      [TODO: implement setup_vector_search_index]")

    print("\n[3/3] Creating Atlas Search (full-text) index on trial_events...")
    # TODO: index_name = setup_text_search_index(db["trial_events"])
    # TODO: print(f"      ✓ Text index '{index_name}' is ACTIVE")
    print("      [TODO: implement setup_text_search_index]")

    print("\n✅ Atlas index setup complete. You can now run seed_historical.py.")


if __name__ == "__main__":
    main()
