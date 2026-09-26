"""
database/mongo_client.py
=========================
MongoDB Atlas connection management and common query helpers.

All other modules import `get_db()` from here — never instantiate
MongoClient directly elsewhere. This ensures a single connection pool
is reused across the application.

Connection string is read from the MONGODB_URI environment variable.
"""

import logging
import os
from functools import lru_cache
from typing import Optional
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)

# Default database name — overridden by MONGODB_DB_NAME env var
DEFAULT_DB_NAME = "healthquant"


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """
    Return a singleton MongoClient connected to MongoDB Atlas.

    Uses lru_cache so the connection is reused across calls.
    The client is configured with a 5-second server selection timeout
    to fail fast during development if Atlas is unreachable.

    Returns:
        MongoClient connected to the URI in MONGODB_URI env var.

    Raises:
        ValueError: If MONGODB_URI is not set.
        ConnectionFailure: If Atlas is unreachable.
    """
    uri = os.getenv("MONGODB_URI", "")
    if not uri or "<" in uri:
        raise ValueError("Configure MONGODB_URI in the local .env file")
    # Atlas uses public TLS certificates. Supply a trusted CA bundle on Python
    # installations without a configured system trust store; never disable TLS
    # verification. Non-SRV/private deployments keep their URI configuration.
    tls_options = {"tlsCAFile": certifi.where()} if uri.startswith("mongodb+srv://") else {}
    client = MongoClient(uri, **tls_options, serverSelectionTimeoutMS=5000,
                         connectTimeoutMS=5000, socketTimeoutMS=10000)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        raise ConnectionError("MongoDB is unavailable; check credentials and Atlas network access") from None
    return client


def get_db(db_name: Optional[str] = None) -> Database:
    """
    Return the healthquant MongoDB database object.

    Args:
        db_name: Database name. Defaults to MONGODB_DB_NAME env var
                 or "healthquant" if not set.

    Returns:
        pymongo Database object.
    """
    return get_client()[db_name or os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME)]


def upsert_document(
    collection_name: str,
    filter_query: dict,
    document: dict,
    db: Optional[Database] = None,
) -> str:
    """
    Upsert a single document into a MongoDB collection.

    Args:
        collection_name: Name of the target collection.
        filter_query: Query dict used to find an existing document.
        document: Full document to insert or replace.
        db: pymongo Database object. Defaults to get_db().

    Returns:
        The upserted document's _id as a string.
    """
    database = get_db() if db is None else db
    collection = database[collection_name]
    payload = {key: value for key, value in document.items() if key != "_id"}
    result = collection.update_one(filter_query, {"$set": payload}, upsert=True)
    return str(result.upserted_id or collection.find_one(filter_query)["_id"])


def bulk_upsert(
    collection_name: str,
    documents: list[dict],
    key_fields: list[str],
    db: Optional[Database] = None,
) -> dict:
    """
    Bulk upsert a list of documents using the specified key fields as the filter.

    Uses bulk_write with UpdateOne(upsert=True) for efficiency.

    Args:
        collection_name: Target collection name.
        documents: List of document dicts to upsert.
        key_fields: Field names to use as the upsert filter (e.g. ["ticker", "date"]).
        db: pymongo Database object. Defaults to get_db().

    Returns:
        Dict with counts: {inserted, modified, matched}.
    """
    if not key_fields:
        raise ValueError("key_fields cannot be empty")
    if not documents:
        return {"inserted": 0, "modified": 0, "matched": 0}
    operations = [UpdateOne({key: doc[key] for key in key_fields},
                           {"$set": {k: v for k, v in doc.items() if k != "_id"}},
                           upsert=True) for doc in documents]
    database = get_db() if db is None else db
    result = database[collection_name].bulk_write(operations, ordered=False)
    return {"inserted": result.upserted_count, "modified": result.modified_count,
            "matched": result.matched_count}


def health_check() -> dict:
    """
    Verify MongoDB Atlas connectivity and return cluster status info.

    Returns:
        Dict with keys: connected (bool), db_name (str), collection_counts (dict).
    """
    try:
        db = get_db()
        return {"connected": True, "db_name": db.name, "collection_counts": {
            name: db[name].estimated_document_count() for name in db.list_collection_names()}}
    except Exception:
        return {"connected": False, "error": "MongoDB unavailable; check connection configuration"}
