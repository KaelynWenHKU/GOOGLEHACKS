# Checkpoint Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reject checkpoints whose feature order or training cutoff cannot be verified.

**Architecture:** Validate canonical ISO dates at save/load boundaries. Require explicit feature names and training cutoff in loaded dictionary payloads instead of assuming the current schema. Keep the existing model/scaler/map return contract.

**Tech Stack:** Python, pandas, pickle (trusted local files only), pytest.

---

### Task 1: Regression tests

Modify `healthquant-agent/tests/test_train.py`: create trusted temporary pickle fixtures, omit required metadata, reorder feature names, use invalid dates and nondictionary payloads; expect `ValueError`. Verify invalid save dates create no files and invalid checkpoint selectors cannot become file paths. Run `.venv/bin/python -m pytest healthquant-agent/tests/test_train.py -q` before implementation.

### Task 2: Boundary checks

Modify `healthquant-agent/hmm/train.py`: a shared date validator accepts only canonical YYYY-MM-DD values representing real dates. Save uses that validator; load validates a non-latest selector before opening, requires dictionary payloads plus explicit `feature_names` and `train_end_date`, verifies schema order and validates the stored cutoff.

### Task 3: Documentation and release

Update `healthquant-agent/README.md` with the compatibility and trusted-pickle limitation. Run focused/full tests, compile and diff checks, scan staged content for secrets and push only verified changes. Preserve user-owned untracked skill directories. No live model deployment or performance claim.

The referenced superpowers execution skill is unavailable; use direct test-first implementation under the user's continuation authorization.

## Verification

Ten new regression cases failed before implementation; the reordered-schema case already passed. After the fix, all 30 training tests and all 141 repository tests passed. Compile and diff checks passed; one existing joblib physical-core detection warning remains. No live model was trained or deployed and no performance result was generated.
