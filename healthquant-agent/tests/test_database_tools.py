"""Validate read-only service contracts without connecting to external APIs."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from database import mongo_client, vector_search
from agent import tools


def test_mongo_failure_is_sanitized_and_client_closed(monkeypatch):
    mongo_client.get_client.cache_clear()
    client = MagicMock()
    client.admin.command.side_effect = RuntimeError("secret-uri")
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setattr(mongo_client, "MongoClient", lambda *args, **kwargs: client)
    with pytest.raises(ConnectionError) as error:
        mongo_client.get_client()
    assert "secret-uri" not in str(error.value)
    client.close.assert_called_once()
    mongo_client.get_client.cache_clear()


def test_embeddings_have_query_document_types_and_dimensions(monkeypatch):
    client = MagicMock()
    client.embed.return_value = SimpleNamespace(embeddings=[[0.1] * 1024])
    monkeypatch.setattr(vector_search, "get_voyage_client", lambda: client)
    vector_search.embed_query("query")
    assert client.embed.call_args.kwargs["input_type"] == "query"
    vector_search.embed_regime_document("document")
    assert client.embed.call_args.kwargs["input_type"] == "document"
    client.embed.return_value = SimpleNamespace(embeddings=[[0.1] * 768])
    with pytest.raises(ValueError, match="1024"):
        vector_search.embed_query("query")


def test_vector_search_requires_cutoff_and_projects_return_provenance():
    db = {"regime_states": MagicMock()}
    db["regime_states"].aggregate.return_value = []
    cutoff = datetime(2024, 1, 1)
    vector_search.find_historical_analogues(db, [0.1] * 1024, before_date=cutoff)
    pipeline = db["regime_states"].aggregate.call_args.args[0]
    assert pipeline[0]["$vectorSearch"]["filter"] == {"date": {"$lt": cutoff}}
    assert pipeline[1]["$project"]["return_observed_at"] == 1
    with pytest.raises(ValueError, match="before_date"):
        vector_search.find_historical_analogues(db, [0.1] * 1024)


def test_analogue_hides_returns_not_observed_at_snapshot_date(monkeypatch):
    monkeypatch.setattr(tools, "get_db", lambda: {})
    retrieve = MagicMock(return_value=[{
        "date": datetime(2023, 12, 29), "score": .9,
        "actual_xlv_return_10d": .5, "return_observed_at": datetime(2024, 1, 15)}])
    monkeypatch.setattr(vector_search, "find_analogues_from_feature_vector", retrieve)
    result = tools.find_historical_analogues([0.1] * 8, as_of="2024-01-01T00:00:00+00:00")
    assert result["analogues"][0]["xlv_ret_10d_actual"] is None
    assert retrieve.call_args.args[-1] == datetime(2024, 1, 1)


def test_empty_catalyst_calendar_preserves_coverage_warning(monkeypatch):
    collection = MagicMock()
    collection.find.return_value.sort.return_value.limit.return_value = []
    monkeypatch.setattr(tools, "get_db", lambda: {"pdufa_events": collection, "trial_events": collection})
    result = tools.get_upcoming_catalysts()
    assert result["status"] == "ok" and not result["pdufa_events"]
    assert "not verified" in result["coverage_note"]
