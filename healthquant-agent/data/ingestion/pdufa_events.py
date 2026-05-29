"""
data/ingestion/pdufa_events.py
================================
FDA PDUFA date extractor using two complementary sources:

  Source A — SEC EDGAR full-text search (free, no auth):
    Parses 8-K filings mentioning "PDUFA" and "target action date".
    URL: https://efts.sec.gov/LATEST/search-index?q="PDUFA"+"target+action+date"&forms=8-K

  Source B — RTTNews FDA calendar (public HTML):
    URL: https://www.rttnews.com/corpinfo/fdacalendar.aspx
    Scraped with requests + BeautifulSoup. Respects robots.txt.

Results from both sources are deduplicated by (ticker, pdufa_date) before
being stored in the pdufa_events MongoDB collection.
"""

import logging
import re
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# SEC EDGAR full-text search endpoint
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# RTTNews public FDA calendar
RTTNEWS_FDA_URL = "https://www.rttnews.com/corpinfo/fdacalendar.aspx"


def fetch_pdufa_from_edgar(
    start_date: str,
    end_date: str,
    max_results: int = 200,
) -> list[dict]:
    """
    Search SEC EDGAR for 8-K filings that mention PDUFA target action dates.

    Args:
        start_date: ISO date "YYYY-MM-DD" — earliest filing date to search.
        end_date: ISO date "YYYY-MM-DD" — latest filing date to search.
        max_results: Maximum number of filings to retrieve.

    Returns:
        List of raw pdufa_event dicts extracted from SEC filings.
        Each dict contains: company_name, pdufa_date (datetime), drug_name,
        indication, sec_filing_url, data_source="sec_8k".
    """
    # TODO: build EDGAR search URL with q="PDUFA"+"target action date"&forms=8-K&dateRange=custom
    # TODO: paginate through results up to max_results
    # TODO: for each hit, fetch the filing text and call parse_edgar_filing()
    # TODO: deduplicate by (company_name, pdufa_date)
    raise NotImplementedError


def parse_edgar_filing(filing_url: str, company_name: str) -> Optional[dict]:
    """
    Download and parse a single SEC 8-K filing to extract PDUFA date details.

    Args:
        filing_url: Direct URL to the filing document on SEC.gov.
        company_name: Company name from the EDGAR search result.

    Returns:
        Dict with PDUFA event details, or None if no PDUFA date found.
        Keys: pdufa_date, drug_name, indication, review_type, application_type.
    """
    # TODO: fetch filing text with requests.get
    # TODO: use regex to find "PDUFA date" / "target action date" patterns
    # TODO: extract drug name and indication from surrounding text
    # TODO: detect "priority review" keyword → set priority_review=True
    raise NotImplementedError


def fetch_pdufa_from_rttnews() -> list[dict]:
    """
    Scrape the RTTNews FDA calendar for upcoming PDUFA dates.

    Returns:
        List of pdufa_event dicts with: pdufa_date, company_name, drug_name,
        indication, data_source="rttnews".

    Note:
        RTTNews is a public page — be respectful, add User-Agent header,
        and cache results to avoid hammering the server.
    """
    # TODO: fetch the RTTNews FDA calendar page with requests
    # TODO: parse the HTML table with BeautifulSoup
    # TODO: extract date, drug, company, indication from each table row
    # TODO: parse date strings into datetime objects
    raise NotImplementedError


def merge_and_deduplicate(
    edgar_events: list[dict],
    rttnews_events: list[dict],
) -> list[dict]:
    """
    Combine EDGAR and RTTNews PDUFA events, deduplicating by (ticker, pdufa_date).

    When both sources provide the same event, the SEC filing record takes
    precedence (more authoritative) and the RTTNews record is merged in
    to fill any missing fields.

    Args:
        edgar_events: PDUFA events sourced from SEC EDGAR.
        rttnews_events: PDUFA events sourced from RTTNews.

    Returns:
        Deduplicated list of pdufa_event dicts, sorted by pdufa_date ascending.
    """
    # TODO: build a dict keyed by (ticker, pdufa_date.date()) for edgar events
    # TODO: for each rttnews event, check if key exists; if not, add it
    # TODO: sort final list by pdufa_date ascending
    raise NotImplementedError


def run_ingestion(start_date: str, end_date: str, db) -> int:
    """
    Full PDUFA ingestion pipeline: fetch, deduplicate, and upsert to MongoDB.

    Args:
        start_date: ISO date string — start of the search window.
        end_date: ISO date string — end of the search window.
        db: pymongo Database object (healthquant).

    Returns:
        Number of new documents inserted into pdufa_events collection.
    """
    # TODO: call fetch_pdufa_from_edgar(start_date, end_date)
    # TODO: call fetch_pdufa_from_rttnews()
    # TODO: call merge_and_deduplicate()
    # TODO: upsert each event to pdufa_events collection (filter: ticker + pdufa_date)
    # TODO: return count of new insertions
    raise NotImplementedError
