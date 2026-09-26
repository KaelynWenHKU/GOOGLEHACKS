# Live Atlas Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create verified Atlas indexes and import a bounded, clearly labeled current ClinicalTrials.gov snapshot without fabricating historical regimes.

**Architecture:** Idempotent index setup refuses conflicting definitions rather than replacing them. A read-only public API client normalizes administrative trial metadata, retaining date precision and collection-time provenance. An explicit CLI write flag imports a bounded snapshot into trial_events; it does not train models or embed invented history.

**Tech Stack:** Python, requests, PyMongo, pytest, ClinicalTrials.gov v2.

---

### Task 1: Index setup
Files: `healthquant-agent/database/schema.py`, `healthquant-agent/scripts/setup_atlas_indexes.py`, `healthquant-agent/tests/test_live_bootstrap.py`.
1. Add failing tests for create/reuse/conflict/readiness/failure behavior.
2. Implement standard unique identity indexes and bounded search-index polling using SearchIndexModel.
3. Run `.venv/bin/python -m pytest healthquant-agent/tests/test_live_bootstrap.py -q`.

### Task 2: Current trial snapshot
Files: `healthquant-agent/data/ingestion/clinical_trials.py`, `healthquant-agent/scripts/import_current_trials.py`, same tests.
1. Add failing tests for date precision, missing dates, provenance, pagination and explicit partial coverage.
2. Parse only administrative fields; missing enrollment/dates remain null. Partial dates retain raw text and are not invented as exact dates.
3. Add a dry-run-default CLI; `--write` upserts by NCT ID and records coverage metadata. Limit first import to 100 current active Phase 3 records, not the full market or ticker universe.
4. Test twice-run upsert identity and ensure historical outcome fields are not overwritten.

### Task 3: Live verification and handoff
1. Run full pytest and diff/secret checks.
2. Execute index setup and import with the user's local credentials; do not print provider exceptions or secrets.
3. Query readiness, record counts and representative date coverage. Do not create regime predictions without audited historical inputs.
4. Update README with commands and limitations, commit only verified code, push to origin.

The referenced superpowers skill is not installed; execute directly in this session per the user's request. Existing project checkout is clean. No paid tier upgrades, fabricated ticker associations or historical backfills are authorized by this plan.
