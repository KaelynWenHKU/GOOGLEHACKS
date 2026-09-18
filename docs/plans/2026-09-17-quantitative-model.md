# HealthQuant Quantitative Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a tested, no-lookahead 8-feature Gaussian HMM pipeline with regime prediction, walk-forward evaluation, portfolio backtesting, and reproducible reports.

**Architecture:** Keep market feature calculation in `data/ingestion/market_data.py`, assemble point-in-time clinical and market signals in `hmm/features.py`, and fit only on expanding historical windows in `hmm/train.py`. Prediction consumes a saved model/scaler/semantic state map, while `hmm/backtest.py` evaluates only out-of-sample predictions and creates judge-facing Plotly artifacts.

**Tech Stack:** Python 3.11+, NumPy, pandas, scikit-learn, hmmlearn, PyMongo-compatible collections, yfinance, Plotly, pytest.

---

### Task 1: Secure and testable repository baseline

**Files:**
- Create: `.gitignore`
- Modify: `healthquant-agent/requirements.txt`
- Create: `healthquant-agent/tests/conftest.py`

**Steps:**
1. Ignore `.env`, credentials, caches, generated model checkpoints, data caches, OS files, and generated reports.
2. Add `pytest` as a development/test dependency.
3. Add a test bootstrap that imports the project package from `healthquant-agent/`.
4. Run `python -m pytest --collect-only` and confirm collection succeeds.

### Task 2: Implement point-in-time feature construction

**Files:**
- Modify: `healthquant-agent/hmm/features.py`
- Create: `healthquant-agent/tests/test_features.py`

**Steps:**
1. Write failing tests for date parsing, 30-day PDUFA/Phase 3 queries, as-of enrollment aggregation, feature order, matrix construction, NaN rejection, and scaler reuse.
2. Run `pytest tests/test_features.py -v` and verify the tests fail on the current stubs.
3. Implement the MongoDB queries with knowledge-date guards (`created_at`/`last_updated` when present) so historical features never consume records first known in the future.
4. Make market feature providers injectable for deterministic tests and batch workflows.
5. Implement strict shape/finite-value validation and `StandardScaler` fit/transform paths.
6. Run `pytest tests/test_features.py -v` and confirm all tests pass.

### Task 3: Implement robust HMM training and semantic state labels

**Files:**
- Modify: `healthquant-agent/hmm/train.py`
- Create: `healthquant-agent/tests/test_train.py`

**Steps:**
1. Write failing tests for multi-restart model selection, distinct semantic labels, checkpoint round-trip, and expanding monthly walk-forward predictions.
2. Implement deterministic multi-start fitting and reject non-finite/undersized matrices.
3. Infer labels using a composite fear/risk score while guaranteeing a one-to-one state mapping.
4. Persist the model, scaler, state labels, feature names, and train-end metadata atomically.
5. Implement monthly expanding-window refits from a supplied historical feature DataFrame, ensuring each test month uses only prior observations.
6. Run `pytest tests/test_train.py -v` and confirm all tests pass.

### Task 4: Implement regime prediction

**Files:**
- Modify: `healthquant-agent/hmm/predict.py`
- Create: `healthquant-agent/tests/test_predict.py`

**Steps:**
1. Write failing tests for posterior probabilities, semantic transition mapping, upcoming catalyst queries, and complete result schema.
2. Compute posterior probabilities from a chronological feature sequence rather than an isolated row.
3. Map transition probabilities by semantic label, not raw state number.
4. Add MongoDB catalyst helpers with clean JSON-safe output.
5. Run `pytest tests/test_predict.py -v` and confirm all tests pass.

### Task 5: Implement strategy backtest and visual reports

**Files:**
- Modify: `healthquant-agent/hmm/backtest.py`
- Create: `healthquant-agent/tests/test_backtest.py`

**Steps:**
1. Write failing tests for lagged execution weights, compounding, drawdowns, Sharpe ratio, 10-day directional hit rate, and regime-period grouping.
2. Shift signals by one trading day so a close-derived signal is not traded at the same close.
3. Implement stable performance metrics with explicit empty/zero-volatility behavior.
4. Implement timeline and HTML report generation without requiring live downloads when `xlv_price` is supplied.
5. Run `pytest tests/test_backtest.py -v` and confirm all tests pass.

### Task 6: Complete the reproducible validation CLI

**Files:**
- Modify: `healthquant-agent/scripts/validate_backtest.py`
- Modify: `healthquant-agent/README.md`
- Create: `healthquant-agent/tests/test_validate_backtest.py`

**Steps:**
1. Implement MongoDB prediction loading and price alignment.
2. Implement formatted metric output and JSON-safe artifact writing.
3. Add CLI documentation that distinguishes model validation from unverified target metrics.
4. Run the full test suite and compile check.

### Task 7: Commit and publish

**Files:** all files above plus the existing project scaffold.

**Steps:**
1. Scan tracked candidates for secrets and remove embedded credentials from the Git remote URL.
2. Run `python -m pytest -q` and `python -m compileall`.
3. Review `git diff --check` and `git status`.
4. Create the initial commit with a descriptive message.
5. Push branch `main` to `origin` and verify the remote branch exists.

