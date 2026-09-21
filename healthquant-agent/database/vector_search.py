"""
database/vector_search.py
==========================
MongoDB Atlas Vector Search wrapper for regime analogue retrieval.

Uses Voyage AI voyage-3-large (1024 dims) for embedding and
$vectorSearch aggregation stage for approximate nearest neighbour search.

Vector Search index name: "regime_vector_index"
Collection: regime_states
Similarity metric: cosine
Dimensions: 1024 (voyage-3-large)

See Section 19.4 of the spec for the complete query pattern.
"""

import logging
import os
import math
from datetime import datetime
from typing import Optional

import voyageai
from dotenv import load_dotenv
from pymongo.database import Database

load_dotenv()
logger = logging.getLogger(__name__)

# Voyage AI model used for all embeddings in this project
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3-large")
EMBEDDING_DIMS = 1024  # matches voyage-3-large output dimensions

# Atlas Vector Search index configuration (matches scripts/setup_atlas_indexes.py)
VECTOR_INDEX_NAME = "regime_vector_index"
EMBEDDING_FIELD = "feature_embedding"
NUM_CANDIDATES_MULTIPLIER = 50  # numCandidates = top_k * this value


def get_voyage_client() -> voyageai.Client:
    """
    Return a Voyage AI client, reading VOYAGE_API_KEY from environment.

    Returns:
        voyageai.Client instance.

    Raises:
        ValueError: If VOYAGE_API_KEY is not set.
    """
    if not os.getenv("VOYAGE_API_KEY") or "<" in os.getenv("VOYAGE_API_KEY", ""):
        raise ValueError("Configure VOYAGE_API_KEY for historical analogue search")
    return voyageai.Client(timeout=15, max_retries=1)


def embed_regime_document(text_description: str) -> list[float]:
    """
    Embed a regime state text description for storage in MongoDB.

    Uses input_type="document" — correct for embedding documents at index time.
    See Section 19.5 for the distinction between "document" and "query" types.

    Args:
        text_description: Human-readable regime description string built by
                          build_regime_text().

    Returns:
        List of 1024 floats (the Voyage AI embedding).
    """
    return embed_batch([text_description], "document")[0]


def embed_query(query_text: str) -> list[float]:
    """
    Embed a query string for vector search (at retrieval time).

    Uses input_type="query" — this prompt-prefixing significantly improves
    retrieval accuracy over using "document" for queries.

    Args:
        query_text: Text description of the current regime state.

    Returns:
        List of 1024 floats.
    """
    return embed_batch([query_text], "query")[0]


def embed_batch(
    texts: list[str],
    input_type: str = "document",
) -> list[list[float]]:
    """
    Batch-embed multiple texts in a single Voyage AI API call.

    More efficient than calling embed_regime_document() in a loop.
    Used by seed_historical.py to embed 10 years of daily regime descriptions.

    Args:
        texts: List of text strings to embed.
        input_type: "document" for storage, "query" for search.

    Returns:
        List of embedding lists, one per input text.
    """
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be document or query")
    if not texts:
        return []
    result = get_voyage_client().embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    if len(result.embeddings) != len(texts):
        raise ValueError("Embedding count does not match input count")
    for vector in result.embeddings:
        if len(vector) != EMBEDDING_DIMS or not all(math.isfinite(x) for x in vector):
            raise ValueError("Embedding must match the 1024-dimensional Atlas index")
    return result.embeddings


def build_regime_text(regime_doc: dict) -> str:
    """
    Convert a regime state document into rich text for embedding.

    Rather than embedding the raw 8-float vector (too sparse for semantic meaning),
    we embed a human-readable description. This significantly improves retrieval
    quality for the analogue search.

    Args:
        regime_doc: A regime_states document dict (with feature_vector and feature_names).

    Returns:
        Text string describing the regime state in natural language.
    """
    from hmm.features import FEATURE_NAMES
    values = regime_doc["feature_vector"]
    names = regime_doc.get("feature_names", FEATURE_NAMES)
    if names != FEATURE_NAMES or len(values) != 8 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected the canonical eight finite raw features")
    features = ", ".join(f"{name}={value:.6g}" for name, value in zip(names, values))
    return (f"Healthcare market regime: {regime_doc.get('regime_label', 'unclassified')}. "
            f"Date: {regime_doc.get('date', 'unknown')}. Feature values: {features}.")


def find_historical_analogues(
    db: Database,
    query_embedding: list[float],
    top_k: int = 3,
    before_date: Optional[datetime] = None,
) -> list[dict]:
    """
    Run Atlas Vector Search to find historically similar regime periods.

    Uses the $vectorSearch aggregation stage (not the deprecated $search knnBeta).
    Filters to only return dates strictly before before_date to prevent lookahead.

    Args:
        db: pymongo Database object.
        query_embedding: 1024-dim Voyage AI embedding of the current regime text.
        top_k: Number of similar periods to return.
        before_date: Only search historical dates before this datetime.
                     Should always be set to prevent lookahead in backtesting.

    Returns:
        List of dicts with keys:
            date, regime_label, brief_summary, transition_probs_10d,
            feature_vector, score (cosine similarity in [0, 1]).
    """
    if before_date is None:
        raise ValueError("before_date is required for historical search")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    if len(query_embedding) != EMBEDDING_DIMS or not all(math.isfinite(v) for v in query_embedding):
        raise ValueError("Invalid query embedding")
    pipeline = [{"$vectorSearch": {
        "index": VECTOR_INDEX_NAME, "path": EMBEDDING_FIELD,
        "queryVector": query_embedding, "numCandidates": top_k * NUM_CANDIDATES_MULTIPLIER,
        "limit": top_k, "filter": {"date": {"$lt": before_date}},
    }}, {"$project": {
        "_id": 0, "date": 1, "regime_label": 1, "brief_summary": 1,
        "actual_xlv_return_10d": 1, "xlv_ret_10d_actual": 1,
        "return_observed_at": 1, "key_events": 1,
        "score": {"$meta": "vectorSearchScore"},
    }}]
    return list(db["regime_states"].aggregate(pipeline, maxTimeMS=10000))


def find_analogues_from_feature_vector(
    db: Database,
    raw_feature_vector: list[float],
    feature_names: list[str],
    regime_label: str,
    top_k: int = 3,
    before_date: Optional[datetime] = None,
) -> list[dict]:
    """
    High-level wrapper: converts a raw feature vector to text, embeds it,
    and retrieves analogues via find_historical_analogues().

    This is the function called by the ADK tool find_historical_analogues
    (see agent/tools.py).

    Args:
        db: pymongo Database object.
        raw_feature_vector: 8-dim list of raw feature values.
        feature_names: List of 8 feature name strings.
        regime_label: Current regime label (included in the query text).
        top_k: Number of analogues to return.
        before_date: Cutoff date for analogue search (no lookahead).

    Returns:
        Same format as find_historical_analogues().
    """
    if before_date is None or not 1 <= top_k <= 10:
        raise ValueError("Provide a cutoff and top_k between 1 and 10")
    description = build_regime_text({"feature_vector": raw_feature_vector,
        "feature_names": feature_names, "regime_label": regime_label, "date": before_date.isoformat()})
    return find_historical_analogues(db, embed_query(description), top_k, before_date)
