"""Offline coverage for bounded, provenance-checked historical embedding."""
from copy import deepcopy
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from database.seed_historical import embed_stored_predictions
from database import vector_search
from hmm.features import FEATURE_NAMES


def record():
    return {"_id": "test-row", "date": datetime(2024, 1, 2),
            "train_end_date": "2023-12-29", "predicted_regime": "risk-on",
            "feature_names": FEATURE_NAMES.copy(), "feature_vector": [0.1] * 8,
            "state_probs": [0.8, 0.1, 0.1]}


def setup_db(monkeypatch, rows):
    collection = MagicMock()
    collection.find.return_value.sort.return_value.limit.return_value = rows
    collection.update_one.return_value.matched_count = 1
    embed = MagicMock(return_value=[[.01] * 1024 for _ in rows])
    monkeypatch.setattr(vector_search, "embed_batch", embed)
    return {"regime_states": collection}, embed


def test_dry_run_validates_without_embeddings_or_writes(monkeypatch):
    db, embed = setup_db(monkeypatch, [record()])
    result = embed_stored_predictions(db, "2024-01-01", "2024-01-31", max_documents=7)
    assert result == {"selected": 1, "written": 0, "conflicts": 0, "dry_run": True}
    embed.assert_not_called()
    db["regime_states"].update_one.assert_not_called()
    query = db["regime_states"].find.call_args.args[0]
    assert query["feature_embedding"] == {"$exists": False}
    db["regime_states"].find.return_value.sort.return_value.limit.assert_called_once_with(7)


@pytest.mark.parametrize("changes", [
    {"train_end_date": "2024-01-03"}, {"train_end_date": None},
    {"feature_names": FEATURE_NAMES[::-1]}, {"predicted_regime": "unknown"},
    {"feature_vector": [float("nan")] * 8}, {"state_probs": [.8, .8, .1]},
    {"regime_label": "catalyst-fear"},
])
def test_entire_batch_is_validated_before_paid_calls(monkeypatch, changes):
    bad = {**record(), "_id": "second-row", "date": datetime(2024, 1, 3), **changes}
    db, embed = setup_db(monkeypatch, [record(), bad])
    with pytest.raises(ValueError):
        embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    embed.assert_not_called()
    db["regime_states"].update_one.assert_not_called()


def test_write_excludes_future_outcomes_and_guards_source_changes(monkeypatch):
    row = {**record(), "actual_xlv_return_10d": 987.654, "brief_summary": "future result"}
    original = deepcopy(row)
    db, embed = setup_db(monkeypatch, [row])
    result = embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    assert result["written"] == 1
    text = embed.call_args.args[0][0]
    assert "987.654" not in text and "future result" not in text
    assert embed.call_args.args[1] == "document"
    query, update = db["regime_states"].update_one.call_args.args
    assert query["_id"] == row["_id"]
    assert query["feature_vector"] == row["feature_vector"]
    assert query["train_end_date"] == row["train_end_date"]
    assert query["feature_embedding"] == {"$exists": False}
    assert "actual_xlv_return_10d" not in update["$set"]
    assert update["$set"]["regime_label"] == "risk-on"
    assert row == original


def test_concurrent_source_change_is_reported_not_upserted(monkeypatch):
    db, _ = setup_db(monkeypatch, [record()])
    db["regime_states"].update_one.return_value.matched_count = 0
    result = embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    assert result["written"] == 0 and result["conflicts"] == 1
    assert db["regime_states"].update_one.call_args.kwargs["upsert"] is False


@pytest.mark.parametrize("vectors", [[], [[0.1] * 8], [[float("inf")] * 1024], [[0.0] * 1024]])
def test_invalid_provider_output_never_written(monkeypatch, vectors):
    db, embed = setup_db(monkeypatch, [record()])
    embed.return_value = vectors
    with pytest.raises(ValueError):
        embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    db["regime_states"].update_one.assert_not_called()


def test_empty_batch_does_not_call_voyage(monkeypatch):
    db, embed = setup_db(monkeypatch, [])
    assert embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)["selected"] == 0
    embed.assert_not_called()


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_invalid_limit_rejected_before_database_access(limit):
    with pytest.raises(ValueError):
        embed_stored_predictions({}, "2024-01-01", "2024-01-31", max_documents=limit)


def test_duplicate_dates_rejected_before_embedding(monkeypatch):
    db, embed = setup_db(monkeypatch, [record(), {**record(), "_id": "duplicate"}])
    with pytest.raises(ValueError, match="unique"):
        embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    embed.assert_not_called()


def test_retry_skips_rows_with_existing_embeddings(monkeypatch):
    db, embed = setup_db(monkeypatch, [record()])
    embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    # Mongo's missing-field filter excludes the now-embedded row on retry.
    db["regime_states"].find.return_value.sort.return_value.limit.return_value = []
    result = embed_stored_predictions(db, "2024-01-01", "2024-01-31", write=True)
    assert result["selected"] == 0
    assert embed.call_count == 1


def test_cli_defaults_to_dry_run_and_sanitizes_errors(monkeypatch, capsys):
    import database.mongo_client as mongo
    import scripts.embed_historical_regimes as cli
    db, embed = setup_db(monkeypatch, [record()])
    monkeypatch.setattr(mongo, "get_db", lambda: db)
    args = ["--start", "2024-01-01", "--end", "2024-01-31"]
    cli.main(args)
    assert '"dry_run": true' in capsys.readouterr().out
    embed.assert_not_called()
    embed.side_effect = RuntimeError("private-provider-token")
    with pytest.raises(SystemExit):
        cli.main([*args, "--write"])
    output = capsys.readouterr().out
    assert "RuntimeError" in output and "private-provider-token" not in output
