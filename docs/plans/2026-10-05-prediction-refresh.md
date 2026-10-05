# Prediction Refresh Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent persisted retraining results from retaining stale embeddings, labels and evaluation fields.

**Architecture:** Build a single MongoDB update from each new walk-forward record. Synchronize schema aliases, invalidate generated embedding/summary fields, and remove unavailable returns and their availability date. Keep unrelated document annotations intact.

**Tech Stack:** Python, NumPy, pytest, PyMongo update operators.

---

### Task 1: Regression tests

Modify `healthquant-agent/tests/test_train.py` with a fake training model and MongoDB collection. Verify `persist_predictions=True` emits one atomic `$set`/`$unset` update, synchronized labels, no stale embedding/summary, and removal of missing evaluation values. Test valid versus incomplete forward-return provenance through a small update-builder helper. First run `.venv/bin/python -m pytest healthquant-agent/tests/test_train.py -q` and observe failures.

### Task 2: Implement refresh semantics

Modify `healthquant-agent/hmm/train.py`: `_prediction_update(record)` copies its input, sets `regime_label`/`regime_id` from prediction fields, clears generated cache fields on every persisted rerun, retains only finite evaluation values, and retains forward return only together with its observed-at timestamp. Clear the legacy return alias to prevent reader precedence bugs. Use the helper in the existing upsert.

### Task 3: Verify and document

Document re-embedding after reruns in `healthquant-agent/README.md`. Run focused and full tests, compile/diff checks and staged secret scan. Commit/push only verified files. No live database mutations. The referenced superpowers execution skill is unavailable; execute the authorized plan directly with test-first checks.

## Verification

Five new cases failed before implementation. After the fix and supplemental timestamp/annotation tests, all 60 focused training/embedding tests and all 171 repository tests passed. Compile and whitespace checks passed. One existing joblib physical-core warning remains. All persistence was mocked; no real database records were changed.
