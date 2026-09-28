# Backtest Trading Costs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make trading-cost assumptions explicit and reproducible without claiming measured performance.

**Architecture:** Extend the existing lagged portfolio simulator with one-way basis-point costs on traded portfolio weight, including drift-driven rebalancing. Charge buy-and-hold its initial entry cost, retain raw market returns for directional scoring, and expose assumptions in JSON, HTML and CLI output. Default zero preserves existing calculations.

**Tech Stack:** Python, pandas, NumPy, pytest, existing Plotly report generator.

---

### Task 1: Test and implement cost accounting

Files: `healthquant-agent/hmm/backtest.py`, `healthquant-agent/tests/test_backtest.py`.

1. Add deterministic tests for entry, exit, half-weight drift, benchmark entry, zero-cost compatibility, invalid basis points and unaffected hit rates.
2. Run `.venv/bin/python -m pytest healthquant-agent/tests/test_backtest.py -q`; new tests should fail before implementation.
3. Compute pretrade weights from the previous holding and market return, then `turnover = abs(target - pretrade)`. Charge `cost = turnover * bps / 10000`; net return is `(1 - cost) * (1 + target * market_return) - 1`.
4. Preserve raw market returns; use separate net benchmark returns for Sharpe. Expose rate and total turnover in metrics.
5. Rerun focused tests.

### Task 2: Reproducible CLI and documentation

Files: `healthquant-agent/scripts/validate_backtest.py`, `healthquant-agent/tests/test_validate_backtest.py`, `healthquant-agent/README.md`.

1. Add `--transaction-cost-bps`, validate before network access, forward it to simulation, and test forwarding with offline mocks.
2. Include costs, zero cash yield, idealized lagged execution and no terminal liquidation in report assumptions. Remove unsupported performance claims from module documentation.
3. Run focused tests and `.venv/bin/python -m pytest -q`.
4. Run `git diff --check`, compile changed modules, scan staged additions for secrets, stage only named files, commit and push `main` when safe.

Execution: continue directly under the user's implementation authorization; the referenced superpowers execution skill is not installed, so use the existing test-first workflow locally.

## Verification

- Eight new core tests first failed because the cost parameter was not implemented.
- Core backtest and validation tests passed after implementation; two additional CLI integration tests verify saved assumptions and reject invalid costs before network access.
- Full offline suite: 111 passed; one existing joblib physical-core detection warning.
- Compile checks and `git diff --check` passed. All performance checks used synthetic test fixtures; no real historical performance is claimed.
