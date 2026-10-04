# Historical Prediction Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect persisted walk-forward predictions to historical vector retrieval without refitting the latest model on history.

**Architecture:** Add a bounded dry-run-first function in `database/seed_historical.py` and a small CLI. Validate every selected record before any embedding request. Embed only date, semantic label and raw features, never realized returns. Write only missing embeddings with an optimistic source-field guard; skip changed records instead of overwriting them. Existing historical ingestion stubs remain explicitly incomplete.

**Tech Stack:** Python, PyMongo-compatible collections, existing Voyage wrapper, pytest.

---

### Task 1: Test the bounded workflow

Create `healthquant-agent/tests/test_historical_embeddings.py`. Mock MongoDB and Voyage. Test dry-run/no-op behavior, invalid cutoff/schema/label rejection before provider calls, bounded queries, embedding dimensions and finiteness, writes preserving annotations, optimistic concurrency conflicts, and exclusion of future-return fields from embedding text.

Run `.venv/bin/python -m pytest healthquant-agent/tests/test_historical_embeddings.py -q` before implementation.

### Task 2: Implement helper and CLI

Modify `healthquant-agent/database/seed_historical.py`: add `embed_stored_predictions(db, start_date, end_date, max_documents=25, write=False)`. Enforce canonical dates, a 1–100 batch limit, strict earlier training cutoff, explicit feature schema, canonical label and valid state probabilities. Validate all rows before calling existing `embed_batch(texts, 'document')`; validate all vectors before writes. Store embedding model/time and source label. Use no upserts.

Create `healthquant-agent/scripts/embed_historical_regimes.py`: explicit date range and `--write`; dry run by default; print counts only and sanitize provider failures. Document paid-call possibility and partial-write resume behavior in `healthquant-agent/README.md`.

### Task 3: Verify and publish

Run focused/full tests, compile, whitespace checks and a staged secret scan. Commit only these named files and this plan; push verified code. Do not execute a live write or claim historical data/backtest completion.

The required superpowers execution skill is unavailable; continue locally with test-first verification under existing authorization.

## Verification

The initial tests failed to import the unimplemented helper. All 22 focused embedding tests and all 163 repository tests now pass, including CLI redaction, retry/no-op behavior, batch preflight, future-outcome exclusion and optimistic update conflicts. Compile and whitespace checks passed. One existing joblib physical-core warning remains. No live Voyage call or MongoDB write was executed.
