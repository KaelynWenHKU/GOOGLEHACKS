"""Offline contracts for live bootstrap; fixtures contain no credentials."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.setup_atlas_indexes import setup_vector_search_index
from data.ingestion.clinical_trials import parse_study


class SearchCollection:
    def __init__(self, existing=None, status="READY"):
        self.items = [] if existing is None else [existing]
        self.created = 0
        self.status = status

    def list_search_indexes(self, name):
        return self.items

    def create_search_index(self, model):
        self.created += 1
        self.items = [{"name": model.document["name"], "type": model.document["type"],
                       "latestDefinition": model.document["definition"],
                       "status": self.status, "queryable": self.status == "READY"}]


def test_index_creation_and_rerun_are_idempotent():
    collection = SearchCollection()
    assert setup_vector_search_index(collection, timeout_seconds=1, poll_seconds=0) == "regime_vector_index"
    setup_vector_search_index(collection, timeout_seconds=1, poll_seconds=0)
    assert collection.created == 1
    assert collection.items[0]["latestDefinition"]["fields"][0]["numDimensions"] == 1024


def test_conflicting_index_is_not_overwritten():
    collection = SearchCollection({"type": "vectorSearch", "latestDefinition": {"fields": []}})
    with pytest.raises(ValueError, match="definition"):
        setup_vector_search_index(collection)
    assert collection.created == 0


@pytest.mark.parametrize("status,exception", [("FAILED", RuntimeError), ("BUILDING", TimeoutError)])
def test_unready_index_is_not_reported_as_success(status, exception):
    with pytest.raises(exception):
        setup_vector_search_index(SearchCollection(status=status), timeout_seconds=0, poll_seconds=0)


def study(completion="2026-10"):
    return {"protocolSection": {
        "identificationModule": {"nctId": "NCT12345678"},
        "statusModule": {"overallStatus": "RECRUITING", "primaryCompletionDateStruct": {"date": completion, "type": "ESTIMATED"},
                         "studyFirstPostDateStruct": {"date": "2018-01-01"}},
        "designModule": {"phases": ["PHASE2", "PHASE3"], "enrollmentInfo": {"count": 150}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example sponsor"}},
        "conditionsModule": {"conditions": ["Example condition"]}}}


def test_partial_dates_are_not_fabricated_and_provenance_is_current():
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    doc = parse_study(study())
    assert doc["primary_completion_date"] is None
    assert doc["primary_completion_date_raw"] == "2026-10"
    assert doc["primary_completion_date_precision"] == "month"
    assert doc["phase"] == "PHASE3"
    assert doc["known_as_of"] >= before
    assert "first_posted_date" not in doc  # no old date bypass of knowledge guard
    assert doc["is_historical"] is False
    assert "ticker" not in doc and "outcome" not in doc


def test_exact_date_and_missing_enrollment():
    raw = study("2026-10-15")
    raw["protocolSection"]["designModule"].pop("enrollmentInfo")
    doc = parse_study(raw)
    assert doc["primary_completion_date"] == datetime(2026, 10, 15)
    assert doc["enrollment_count"] is None


def test_malformed_trial_id_rejected():
    with pytest.raises(ValueError):
        parse_study({"protocolSection": {}})


def test_bounded_snapshot_discloses_partial_coverage(monkeypatch):
    from data.ingestion import clinical_trials as ct
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "studies": [study("2026-10-15")], "totalCount": 10, "nextPageToken": "next"})
    monkeypatch.setattr(ct.requests, "get", get)
    snapshot = ct.fetch_current_snapshot(page_size=1, max_pages=1, sleep_seconds=0)
    assert snapshot["coverage"]["truncated"] is True
    assert snapshot["coverage"]["total_available"] == 10
    assert len(snapshot["documents"]) == 1
    assert "AREA[Phase]PHASE3" in calls[0]["params"]["filter.advanced"]


def test_snapshot_upserts_keep_research_annotations():
    from scripts.import_current_trials import import_snapshot
    calls = []
    collection = SimpleNamespace(update_one=lambda query, update, **kwargs: calls.append((query, update, kwargs)))
    db = {"trial_events": collection, "ingestion_runs": collection}
    snapshot = {"documents": [parse_study(study())], "coverage": {"truncated": True}, "run_id": "test-run"}
    import_snapshot(db, snapshot)
    import_snapshot(db, snapshot)
    assert calls[0][0] == calls[2][0] == {"nct_id": "NCT12345678"}
    assert "outcome" not in calls[0][1]["$set"]
    assert "ticker" not in calls[0][1]["$set"]


def test_pagination_uses_next_token_and_reports_completion(monkeypatch):
    from data.ingestion import clinical_trials as ct
    calls = []
    second = study("2026-10-20")
    second["protocolSection"]["identificationModule"]["nctId"] = "NCT87654321"
    pages = iter([{"studies": [study()], "totalCount": 2, "nextPageToken": "page-2"},
                  {"studies": [second]}])
    def get(url, **kwargs):
        calls.append(kwargs["params"])
        payload = next(pages)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(ct.requests, "get", get)
    result = ct.fetch_current_snapshot(page_size=1, max_pages=2, sleep_seconds=0)
    assert calls[1]["pageToken"] == "page-2"
    assert result["coverage"]["truncated"] is False
    assert len(result["documents"]) == 2


@pytest.mark.parametrize("raw", ["2026-13", "2026-02-30", "tomorrow"])
def test_invalid_dates_are_rejected(raw):
    with pytest.raises(ValueError, match="registry date"):
        parse_study(study(raw))


def test_standard_identity_indexes_are_unique():
    from database.schema import setup_all_indexes
    from unittest.mock import MagicMock
    db = {name: MagicMock() for name in ["trial_events", "regime_states", "pdufa_events", "company_ticker_map"]}
    setup_all_indexes(db)
    db["trial_events"].create_index.assert_any_call([("nct_id", 1)], unique=True)
    db["regime_states"].create_index.assert_any_call([("date", -1)], unique=True)


def test_http_failure_never_returns_partial_snapshot(monkeypatch):
    from data.ingestion import clinical_trials as ct
    import requests
    def fail():
        raise requests.HTTPError("unavailable")
    monkeypatch.setattr(ct.requests, "get", lambda *a, **k: SimpleNamespace(raise_for_status=fail))
    with pytest.raises(requests.HTTPError):
        ct.fetch_current_snapshot()
