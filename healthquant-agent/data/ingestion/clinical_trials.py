"""
data/ingestion/clinical_trials.py
==================================
ClinicalTrials.gov REST API v2 client.

Fetches Phase 2/3 clinical trial data for biotech/pharma companies.
No authentication required. API base: https://clinicaltrials.gov/api/v2/studies

Rate limit: polite use — add sleep(0.5) between paginated requests.
Results are cached to MongoDB to avoid redundant API calls.
"""

import time
import logging
from datetime import datetime, date, timezone
import re
import uuid
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Base URL for the ClinicalTrials.gov v2 API
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields we request from the API to keep response payloads lean
REQUESTED_FIELDS = (
    "NCTId,OfficialTitle,Phase,OverallStatus,"
    "PrimaryCompletionDate,EnrollmentCount,EnrollmentType,"
    "LeadSponsorName,Condition,InterventionType,"
    "StartDate,CompletionDate,BriefSummary"
)


def fetch_trials_by_phase(
    phase: str,
    status: str = "RECRUITING",
    page_size: int = 200,
    max_pages: int = 10,
    sleep_seconds: float = 0.5,
) -> list[dict]:
    """
    Fetch clinical trial records from ClinicalTrials.gov for a given phase and status.

    Args:
        phase: Trial phase filter — "PHASE2", "PHASE3", or "PHASE4".
        status: Overall trial status — e.g. "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED".
        page_size: Records per page (API max is 1000).
        max_pages: Safety limit on the number of pages fetched.
        sleep_seconds: Delay between paginated requests to respect rate limits.

    Returns:
        List of raw trial dicts as returned by the API (protocolSection data).
    """
    # TODO: implement pagination loop using pageToken
    # TODO: parse each study's protocolSection into a flat dict
    # TODO: log progress every 500 records
    raise NotImplementedError


def fetch_trials_for_ticker(
    ticker: str,
    sponsor_name: str,
    phases: list[str] | None = None,
) -> list[dict]:
    """
    Fetch all active Phase 2/3 trials for a specific company by sponsor name.

    Args:
        ticker: Stock ticker symbol (used for logging and storage).
        sponsor_name: Free-text sponsor name as it appears in ClinicalTrials.gov.
        phases: List of phases to query; defaults to ["PHASE2", "PHASE3"].

    Returns:
        List of normalised trial dicts ready for insertion into MongoDB trial_events.
    """
    # TODO: loop over phases
    # TODO: call fetch_trials_by_phase with query.term=sponsor_name
    # TODO: normalise raw API response to MongoDB schema (see Section 5.2)
    raise NotImplementedError


def parse_study(raw_study: dict) -> dict:
    """
    Normalise a raw ClinicalTrials.gov study record into the MongoDB trial_events schema.

    Args:
        raw_study: A single study dict from the API's protocolSection.

    Returns:
        Dict matching the trial_events collection schema (Section 5.2 of spec).
        stock_reaction and outcome fields are left null — filled in post-event.
    """
    protocol = raw_study.get("protocolSection", raw_study)
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise ValueError("Study has an invalid NCT identifier")
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    phases = design.get("phases", [])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    doc = {
        "nct_id": nct_id,
        "company_name": protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        "phase": next((p for p in ["PHASE4", "PHASE3", "PHASE2", "PHASE1"] if p in phases), None),
        "phases": phases,
        "status": status.get("overallStatus"),
        "condition": "; ".join(protocol.get("conditionsModule", {}).get("conditions", [])),
        "enrollment_count": design.get("enrollmentInfo", {}).get("count"),
        "enrollment_type": design.get("enrollmentInfo", {}).get("type"),
        "data_source": "clinicaltrials_gov",
        "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "known_as_of": now, "last_updated": now, "is_historical": False,
        "universe_scope": "current_registry_snapshot_not_ticker_screened",
    }
    for field, source in [("primary_completion_date", "primaryCompletionDateStruct"),
                          ("start_date", "startDateStruct")]:
        value = status.get(source, {})
        raw = value.get("date")
        exact, precision = _parse_registry_date(raw)
        doc.update({field: exact, field + "_raw": raw,
                    field + "_precision": precision, field + "_type": value.get("type")})
    # Source publication dates are audit metadata only. They must not be used
    # to backdate availability of the current, possibly revised record.
    doc["source_first_posted_raw"] = status.get("studyFirstPostDateStruct", {}).get("date")
    doc["source_last_update_raw"] = status.get("lastUpdatePostDateStruct", {}).get("date")
    return doc


def _parse_registry_date(raw):
    """Preserve missing/month/year precision without inventing calendar days."""
    if not raw:
        return None, "missing"
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return datetime.strptime(raw, "%Y-%m-%d"), "day"
        if re.fullmatch(r"\d{4}-\d{2}", raw):
            datetime.strptime(raw, "%Y-%m")
            return None, "month"
        if re.fullmatch(r"\d{4}", raw):
            datetime.strptime(raw, "%Y")
            return None, "year"
    except ValueError:
        pass
    raise ValueError("Invalid registry date")


def fetch_current_snapshot(page_size=100, max_pages=1, sleep_seconds=0.5) -> dict:
    """Fetch a bounded current active Phase 3 snapshot with explicit coverage.

    This is not an archived point-in-time dataset, and it is not restricted to
    public companies. Truncation is reported, never silently called complete.
    """
    if not 1 <= page_size <= 1000 or not 1 <= max_pages <= 10 or not 0 <= sleep_seconds <= 30:
        raise ValueError("Invalid fetch limits")
    params = {"format": "json", "pageSize": page_size, "countTotal": "true",
              "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
              "filter.advanced": "AREA[Phase]PHASE3", "sort": "LastUpdatePostDate:desc"}
    documents = {}
    seen_tokens = set()
    next_token = None
    total = None
    for page in range(max_pages):
        response = requests.get(BASE_URL, params=params.copy(), timeout=(10, 30))
        response.raise_for_status()
        payload = response.json()
        if "studies" not in payload or not isinstance(payload["studies"], list):
            raise ValueError("Unexpected ClinicalTrials.gov response")
        if page == 0:
            total = payload.get("totalCount")
        for raw in payload["studies"]:
            doc = parse_study(raw)
            if "PHASE3" not in doc["phases"] or doc["status"] not in {"RECRUITING", "ACTIVE_NOT_RECRUITING"}:
                raise ValueError("API returned a study outside the requested scope")
            documents[doc["nct_id"]] = doc
        next_token = payload.get("nextPageToken")
        if not next_token:
            break
        if next_token in seen_tokens:
            raise ValueError("Repeated pagination token")
        seen_tokens.add(next_token)
        params["pageToken"] = next_token
        if page + 1 < max_pages:
            time.sleep(sleep_seconds)
    return {"run_id": str(uuid.uuid4()), "documents": list(documents.values()), "coverage": {
        "source": BASE_URL, "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "total_available": total, "fetched": len(documents), "truncated": bool(next_token),
        "scope": "current active Phase 3, newest registry updates first; not ticker-screened",
        "historical_point_in_time": False}}


def fetch_completed_trials(
    start_date: str,
    end_date: str,
    phase: str = "PHASE3",
) -> list[dict]:
    """
    Fetch COMPLETED Phase 3 trials within a historical date range.

    Used during historical data seeding (2015–2024) to populate the
    trial_events collection with past events and their known outcomes.

    Args:
        start_date: ISO date string "YYYY-MM-DD" — primary completion date start.
        end_date: ISO date string "YYYY-MM-DD" — primary completion date end.
        phase: Trial phase to query (default "PHASE3").

    Returns:
        List of normalised trial dicts with is_historical=True.
    """
    # TODO: query API with filter.phase=phase, filter.overallStatus=COMPLETED
    # TODO: apply date range filter on PrimaryCompletionDate
    # TODO: call parse_study on each result and set is_historical=True
    raise NotImplementedError
