"""
data/ingestion/company_ticker_map.py
=====================================
Maps free-text clinical trial sponsor names to stock tickers.

ClinicalTrials.gov uses inconsistent free-text company names (e.g.
"BIOGEN MA INC", "Biogen Inc.", "Biogen"). This module resolves them
to canonical tickers using:

  1. A hand-curated seed list of ~150 biotech companies from IBB holdings
  2. rapidfuzz fuzzy string matching (score threshold: 85)
  3. Manual override map for known edge cases (subsidiaries, ADRs)
  4. MongoDB company_ticker_map collection as persistent cache

Build order: populate ticker_map.json from IBB holdings CSV first,
then call build_ticker_map_collection() once to load it into MongoDB.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from rapidfuzz import process as fuzz_process

logger = logging.getLogger(__name__)

# Fuzzy match confidence threshold — below this, flag for manual review
FUZZY_SCORE_THRESHOLD = 85

# Path to the seed JSON file (repo root: data/ticker_map.json)
SEED_FILE = Path(__file__).parent.parent / "ticker_map.json"

# Hard-coded overrides for known edge cases (subsidiaries, ADRs, rebrands)
# Format: { "raw_sponsor_name_substring": "TICKER" }
MANUAL_OVERRIDES: dict[str, str] = {
    "BIOGEN MA": "BIIB",
    "GENENTECH": "RHHBY",
    "MEDIMMUNE": "AZN",
    "LILLY": "LLY",
    "ELI LILLY": "LLY",
    "ABBVIE": "ABBV",
    "BRISTOL MYERS": "BMY",
    "BRISTOL-MYERS": "BMY",
    "REGENERON": "REGN",
    "ASTRAZENECA": "AZN",
}


def resolve_ticker(
    sponsor_name: str,
    canonical_names: list[str],
    name_to_ticker: dict[str, str],
) -> tuple[Optional[str], float]:
    """
    Resolve a raw ClinicalTrials.gov sponsor name to a stock ticker.

    Resolution order:
      1. Check MANUAL_OVERRIDES for substring matches
      2. rapidfuzz fuzzy match against canonical_names list
      3. Return None if best score < FUZZY_SCORE_THRESHOLD

    Args:
        sponsor_name: Raw sponsor name string from ClinicalTrials.gov.
        canonical_names: List of canonical company names from the seed file.
        name_to_ticker: Dict mapping canonical_name → ticker.

    Returns:
        Tuple of (ticker_or_None, confidence_score).
        confidence_score is 100.0 for manual overrides, 0.0 when unmatched.
    """
    # TODO: check MANUAL_OVERRIDES (upper-case substring match)
    # TODO: call fuzz_process.extractOne(sponsor_name, canonical_names)
    # TODO: if score >= FUZZY_SCORE_THRESHOLD, return (ticker, score)
    # TODO: else log a warning and return (None, score)
    raise NotImplementedError


def load_seed_file() -> tuple[list[str], dict[str, str]]:
    """
    Load the canonical company→ticker mapping from data/ticker_map.json.

    Returns:
        Tuple of (canonical_names list, name_to_ticker dict).

    Raises:
        FileNotFoundError: If ticker_map.json has not been built yet.
    """
    # TODO: load SEED_FILE with json.load
    # TODO: extract canonical_name and ticker fields from each entry
    # TODO: return (list of names, {name: ticker} dict)
    raise NotImplementedError


def build_ticker_map_collection(db) -> int:
    """
    Load the seed ticker map JSON into the MongoDB company_ticker_map collection.

    Should be run once during initial setup. Uses upsert to avoid duplicates.

    Args:
        db: pymongo Database object (healthquant).

    Returns:
        Number of documents upserted.
    """
    # TODO: call load_seed_file()
    # TODO: for each entry in seed data, upsert to company_ticker_map
    #   filter: { "ticker": ticker }, update: full document
    # TODO: create unique index on ticker if not already present
    # TODO: return count of upserted documents
    raise NotImplementedError


def lookup_ticker_from_db(sponsor_name: str, db) -> Optional[str]:
    """
    Look up a ticker from MongoDB, falling back to fuzzy matching if not found.

    This is the main entry point called by the data ingestion pipeline.

    Args:
        sponsor_name: Raw sponsor name string.
        db: pymongo Database object.

    Returns:
        Ticker string if resolved, None if unresolvable.
    """
    # TODO: first try exact match in company_ticker_map.aliases array
    # TODO: if not found, load all canonical names from DB and call resolve_ticker()
    # TODO: if resolved, store the alias in MongoDB for future fast lookup
    raise NotImplementedError
