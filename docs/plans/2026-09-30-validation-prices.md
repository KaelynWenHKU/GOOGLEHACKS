# Validation Price Integrity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reject malformed market-price inputs before calculating validation returns.

**Architecture:** Validate the downloaded close series before alignment and percent changes. Require unique, nonmissing timestamps and finite positive numeric prices; sort valid observations chronologically so provider row order cannot change returns.

**Tech Stack:** pandas, NumPy, pytest, existing validation CLI.

---

### Task 1: Regression tests

Modify `healthquant-agent/tests/test_validate_backtest.py`. Inject prices with zero, negative, NaN, infinity and nonnumeric closes; assert `ValueError` mentioning prices. Test duplicate dates and NaT, including duplicate dates after timezone removal. Verify reverse-ordered valid prices produce the same returns as sorted prices.

Run `.venv/bin/python -m pytest healthquant-agent/tests/test_validate_backtest.py -q` and verify the new tests fail before implementation.

### Task 2: Fail-closed price validation

Modify `healthquant-agent/scripts/validate_backtest.py` before `pct_change`: normalize the price index, reject missing/duplicate dates, coerce closes with `pd.to_numeric(errors='coerce')`, reject nonfinite or nonpositive values, and sort the index. Keep the existing missing-session and prediction-provenance guards.

### Task 3: Verify and publish

Document requirements in `healthquant-agent/README.md`. Run focused tests, `.venv/bin/python -m pytest -q`, compile checks and `git diff --check`. Scan staged additions for secrets, commit only the four named files and push when safe. No external data writes or performance claims.

The referenced superpowers execution skill is unavailable; execute directly using the existing test-first workflow under the scheduled implementation authorization.

## Verification

Eight new cases failed before the fix; the existing missing-data guard already rejected NaN. After implementation and the additional timezone-collision case, all 18 validation tests and all 121 repository tests passed. One existing joblib core-count warning remains. Compile and diff whitespace checks passed. Tests use offline fixtures; no historical performance was measured.
