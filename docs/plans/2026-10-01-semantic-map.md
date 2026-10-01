# Explicit Semantic Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent arbitrary HMM state IDs from being silently assigned semantic labels.

**Architecture:** Both prediction and transition forecasting must consume an explicit fitted-checkpoint state map. Missing maps fail closed; all six valid label permutations remain supported. Production callers already provide the fitted map.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: Tests

Modify `healthquant-agent/tests/test_predict.py`: verify missing maps raise `ValueError`, explicitly supply maps in unrelated probability/shape tests, reject boolean/float state keys, and test all six label permutations in both inference and zero-day forecasts. Run `.venv/bin/python -m pytest healthquant-agent/tests/test_predict.py -q` before implementation to demonstrate failures.

### Task 2: Implementation

Modify `healthquant-agent/hmm/predict.py`: replace the fallback dictionary with a descriptive `ValueError`; require integer keys (excluding booleans) and distinct canonical labels. Document the checkpoint-map requirement on both public helpers. Preserve existing probability and horizon validation.

### Task 3: Verify and publish

Update `healthquant-agent/README.md`, run focused and full tests, compile checks, diff review and secret scan. Commit only these files and this plan; push the verified commit. No live database changes or performance claims.

The superpowers execution skill is unavailable; proceed with local test-first implementation under the existing automation authorization.

## Verification

Three new invalid-map cases failed before implementation. After the fix, 30 focused prediction tests and all 130 repository tests passed, including all six label permutations and the existing monthly walk-forward tests. Compile and whitespace checks passed. One existing joblib core-count warning remains. No real performance result was generated.
