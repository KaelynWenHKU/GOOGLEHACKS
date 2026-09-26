"""
hmm/train.py
=============
HMM training with walk-forward validation.

Model: GaussianHMM with 3 states and full covariance matrices.
Training protocol: expanding-window walk-forward (never in-sample prediction).

Walk-forward schedule (from Section 7.2):
  Train: 2015–2019  → Test: 2020
  Train: 2015–2020  → Test: 2021
  Train: 2015–2021  → Test: 2022
  Train: 2015–2022  → Test: 2023
  Train: 2015–2023  → Test: 2024

Within each test year, refit monthly on expanding data.

State labelling convention (verified after each training run):
  State with highest xlv_vol_20d + high pdufa_events_30d → "catalyst-fear"
  State with highest xlv_ret_5d + low pdufa_events_30d  → "risk-on"
  Remaining state                                         → "neutral"
"""

import logging
import pickle
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from hmm.features import FEATURE_NAMES, build_feature_matrix

logger = logging.getLogger(__name__)

# Serialised model checkpoints are saved here
MODEL_DIR = Path(__file__).parent / "models"

# HMM hyperparameters — do not change after initial validation
HMM_CONFIG = {
    "n_components": 3,
    "covariance_type": "full",
    "n_iter": 200,
    "random_state": 42,
    "tol": 1e-4,
}

# Default state label map — MUST be re-verified after each training run
# Keys are state integers 0/1/2; values are canonical label strings
DEFAULT_STATE_LABEL_MAP: dict[int, str] = {
    0: "risk-on",
    1: "neutral",
    2: "catalyst-fear",
}


def train_hmm(
    feature_matrix: np.ndarray,
    n_restarts: int = 5,
) -> GaussianHMM:
    """
    Fit a GaussianHMM on a pre-standardised feature matrix.

    Uses multiple random restarts to mitigate convergence to local optima.
    The run with the highest log-likelihood is returned.

    Args:
        feature_matrix: Standardised array of shape (T, 8).
        n_restarts: Number of random initialisations to try.

    Returns:
        Fitted GaussianHMM model with the highest log-likelihood across restarts.

    Raises:
        RuntimeError: If all restarts fail to converge.
    """
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected (n, {len(FEATURE_NAMES)}) feature matrix, got {matrix.shape}")
    if matrix.shape[0] < max(20, HMM_CONFIG["n_components"] * 5):
        raise ValueError("At least 20 observations are required to train a stable 3-state HMM")
    if not np.isfinite(matrix).all():
        raise ValueError("feature_matrix contains NaN or infinite values")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")

    best_model: GaussianHMM | None = None
    best_score = -np.inf
    failures: list[str] = []
    for restart in range(n_restarts):
        config = {**HMM_CONFIG, "random_state": HMM_CONFIG["random_state"] + restart}
        candidate = GaussianHMM(**config)
        try:
            candidate.fit(matrix)
            score = float(candidate.score(matrix))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(f"restart {restart}: {exc}")
            logger.warning("HMM restart %d failed: %s", restart, exc)
            continue
        if not candidate.monitor_.converged:
            logger.warning("HMM restart %d did not converge after %d iterations", restart, candidate.monitor_.iter)
            failures.append(f"restart {restart}: convergence monitor rejected fit")
            continue
        # hmmlearn also reports convergence when it exhausts n_iter, or when
        # likelihood falls. Inspect the final improvement before accepting it.
        history = list(candidate.monitor_.history)
        if len(history) < 2 or not np.isfinite(history[-2:]).all():
            failures.append(f"restart {restart}: insufficient finite convergence history")
            continue
        improvement = history[-1] - history[-2]
        if improvement < -1e-6 or improvement >= config["tol"]:
            failures.append(f"restart {restart}: final likelihood improvement {improvement} outside tolerance")
            continue
        if np.isfinite(score) and score > best_score:
            best_model, best_score = candidate, score

    if best_model is None:
        detail = "; ".join(failures) or "all scores were non-finite"
        raise RuntimeError(f"All HMM restarts failed: {detail}")
    logger.info("Selected HMM restart with log-likelihood %.3f", best_score)
    return best_model


def infer_state_labels(model: GaussianHMM) -> dict[int, str]:
    """
    Automatically assign semantic labels to HMM states by inspecting model.means_.

    Label assignment logic (from Section 7.3):
      - "risk-on":        state with highest xlv_ret_5d (index 0) mean
      - "catalyst-fear":  state with highest xlv_vol_20d (index 1) mean
      - "neutral":        remaining state

    Args:
        model: Fitted GaussianHMM.

    Returns:
        Dict mapping state integer → label string.
        Always contains exactly {"risk-on", "neutral", "catalyst-fear"}.

    Note:
        This is a heuristic — always visually verify the assignment by
        printing means_df and checking against historical known periods.
    """
    means = np.asarray(model.means_, dtype=float)
    if means.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(means).all():
        raise ValueError(f"Unexpected HMM means shape/content: {means.shape}")

    means_df = pd.DataFrame(means, columns=FEATURE_NAMES)
    # Training features are standardised, so sums below are dimensionally
    # comparable. Relative strength helps avoid labelling a volatile rally as
    # fear; PDUFA density and put/call pressure make fear sector-specific.
    risk_score = (
        means_df["xlv_ret_5d"]
        + means_df["ibb_spy_ratio"]
        + 0.25 * means_df["xlv_rsi_14d"]
        - 0.5 * means_df["xlv_vol_20d"]
    )
    risk_state = int(risk_score.idxmax())

    fear_score = (
        means_df["xlv_vol_20d"]
        + 0.5 * means_df["pdufa_events_30d"]
        + 0.5 * means_df["xlv_put_call_ratio"]
        - 0.5 * means_df["xlv_ret_5d"]
    )
    fear_candidates = fear_score.drop(index=risk_state)
    fear_state = int(fear_candidates.idxmax())
    neutral_state = int(({0, 1, 2} - {risk_state, fear_state}).pop())
    mapping = {
        risk_state: "risk-on",
        neutral_state: "neutral",
        fear_state: "catalyst-fear",
    }
    logger.info("State labels: %s\n%s", mapping, means_df.to_string())
    return mapping


def save_model(
    model: GaussianHMM,
    scaler: StandardScaler,
    state_label_map: dict[int, str],
    train_end_date: str,
) -> Path:
    """
    Serialise the model, scaler, and state label map to disk.

    Args:
        model: Fitted GaussianHMM.
        scaler: Fitted StandardScaler used to preprocess training data.
        state_label_map: Dict mapping state int → label string.
        train_end_date: ISO date string — last date included in training data.
                        Used to name the checkpoint file.

    Returns:
        Path to the saved .pkl file.
    """
    try:
        parsed_date = pd.Timestamp(train_end_date).strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("train_end_date must be a valid date") from exc
    if set(state_label_map.values()) != {"risk-on", "neutral", "catalyst-fear"}:
        raise ValueError("state_label_map must contain each canonical regime exactly once")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "state_label_map": dict(state_label_map),
        "train_end_date": parsed_date,
        "feature_names": FEATURE_NAMES,
        "hmm_config": HMM_CONFIG,
    }
    dated_path = MODEL_DIR / f"hmm_{parsed_date}.pkl"
    latest_path = MODEL_DIR / "hmm_latest.pkl"
    _atomic_pickle_dump(payload, dated_path)
    _atomic_pickle_dump(payload, latest_path)
    return dated_path


def load_model(checkpoint: str = "latest") -> tuple[GaussianHMM, StandardScaler, dict[int, str]]:
    """
    Load a serialised HMM checkpoint from disk.

    Args:
        checkpoint: "latest" to load the most recent model, or an ISO date
                    string matching a specific checkpoint file.

    Returns:
        Tuple of (model, scaler, state_label_map).

    Raises:
        FileNotFoundError: If no matching checkpoint file is found.
    """
    filename = "hmm_latest.pkl" if checkpoint == "latest" else f"hmm_{checkpoint}.pkl"
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"HMM checkpoint not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"model", "scaler", "state_label_map"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Invalid checkpoint {path}: missing {sorted(missing)}")
    if payload.get("feature_names", FEATURE_NAMES) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature schema differs from current FEATURE_NAMES")
    return payload["model"], payload["scaler"], payload["state_label_map"]


def run_walk_forward_training(
    db,
    start_date: str = "2015-01-01",
    test_years: list[int] | None = None,
    feature_frame: pd.DataFrame | None = None,
    xlv_prices: pd.Series | None = None,
    n_restarts: int = 5,
    persist_predictions: bool = False,
) -> list[dict]:
    """
    Execute the full walk-forward training and evaluation protocol.

    For each test year, fits a model on all data up to the test year start,
    then predicts regime labels for each trading day in the test year.

    Args:
        db: pymongo Database object (healthquant).
        start_date: Start of the expanding training window.
        test_years: List of years to test; defaults to [2020, 2021, 2022, 2023, 2024].

    Returns:
        List of result dicts, one per test period, each containing:
        {year, model_checkpoint, predictions_df (date, predicted_regime, actual_xlv_return_10d)}.

    Note:
        This function may take 30–60 minutes on full 2015–2024 data.
        Progress is logged every month. Results are also saved to MongoDB.
    """
    years = [2020, 2021, 2022, 2023, 2024] if test_years is None else test_years
    if not years:
        raise ValueError("test_years must not be empty")
    if sorted(set(years)) != years:
        raise ValueError("test_years must be unique and sorted ascending")
    frame = feature_frame.copy() if feature_frame is not None else _load_feature_frame(db, start_date, years[-1])
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    _validate_daily_index(frame.index, "feature_frame")
    if not frame.columns.is_unique:
        raise ValueError("feature_frame columns must be unique")
    frame = frame.sort_index()
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"feature_frame is missing columns: {missing}")
    frame = frame.loc[pd.Timestamp(start_date):, FEATURE_NAMES]
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("feature_frame must contain finite historical observations")

    prices = None
    if xlv_prices is not None:
        prices = xlv_prices.copy()
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
        _validate_daily_index(prices.index, "xlv_prices")
        prices = prices.sort_index().astype(float)
        if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
            raise ValueError("xlv_prices must contain finite, strictly positive closes")
    results: list[dict] = []

    for year in years:
        year_predictions: list[dict] = []
        checkpoint_path: Path | None = None
        for month_start in pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS"):
            next_month = month_start + pd.offsets.MonthBegin(1)
            train = frame.loc[(frame.index >= pd.Timestamp(start_date)) & (frame.index < month_start)]
            test = frame.loc[(frame.index >= month_start) & (frame.index < next_month)]
            if test.empty:
                continue
            if len(train) < 20:
                raise ValueError(f"Insufficient training observations before {month_start.date()}: {len(train)}")

            scaler = StandardScaler().fit(train.to_numpy(dtype=float))
            train_scaled = scaler.transform(train.to_numpy(dtype=float))
            model = train_hmm(train_scaled, n_restarts=n_restarts)
            label_map = infer_state_labels(model)
            train_end = train.index[-1].strftime("%Y-%m-%d")
            checkpoint_path = save_model(model, scaler, label_map, train_end)

            test_scaled = scaler.transform(test.to_numpy(dtype=float))
            # Only a short prior tail is needed to initialise the filter. Each
            # prediction is computed on the prefix ending at that day, never on
            # the rest of the month (which would leak future observations via
            # HMM smoothing).
            history_tail = train_scaled[-60:]
            for offset, (date, _) in enumerate(test.iterrows()):
                sequence = np.vstack([history_tail, test_scaled[: offset + 1]])
                probabilities = model.predict_proba(sequence)[-1]
                state = int(np.argmax(probabilities))
                record = {
                    "date": date,
                    "predicted_state": state,
                    "predicted_regime": label_map[state],
                    "state_probs": probabilities.tolist(),
                    "train_end_date": train_end,
                    "feature_vector": test.loc[date].to_numpy(dtype=float).tolist(),
                    "feature_names": FEATURE_NAMES,
                    "state_label_map": {str(key): value for key, value in label_map.items()},
                }
                from hmm.predict import compute_forward_transition_probs
                record["transition_probs_10d"] = compute_forward_transition_probs(
                    model, probabilities, state_label_map=label_map)
                if prices is not None:
                    # Return ending on ``date`` is paired with a one-session
                    # signal lag by backtest.run_backtest. The 10-day value is
                    # forward-looking only as an evaluation target.
                    record["actual_xlv_ret_1d"] = _return_ending_at(prices, date)
                    record["actual_xlv_return_10d"] = _return_after(prices, date, 10)
                    if np.isfinite(record["actual_xlv_return_10d"]):
                        record["return_observed_at"] = prices.index[prices.index.get_loc(date) + 10].to_pydatetime()
                year_predictions.append(record)
                if persist_predictions:
                    db["regime_states"].update_one(
                        {"date": date.to_pydatetime()},
                        {"$set": record},
                        upsert=True,
                    )

        prediction_df = pd.DataFrame(year_predictions)
        if not prediction_df.empty:
            prediction_df = prediction_df.set_index("date").sort_index()
        results.append(
            {
                "year": year,
                "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "predictions_df": prediction_df,
            }
        )
    return results


def _validate_daily_index(index: pd.DatetimeIndex, name: str) -> None:
    """Fail before fitting or writing checkpoints if daily row identity is ambiguous.

    Inputs represent session dates, not intraday timestamps. This does not
    verify exchange-calendar completeness or point-in-time source provenance.
    """
    if index.hasnans or not index.is_unique:
        raise ValueError(f"{name} dates must be unique and non-missing")
    if not index.equals(index.normalize()):
        raise ValueError(f"{name} dates must be midnight session dates, not intraday timestamps")


def _atomic_pickle_dump(payload: dict, destination: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt it."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_feature_frame(db, start_date: str, end_year: int) -> pd.DataFrame:
    """Load already-computed raw feature vectors from MongoDB."""
    cursor = db["regime_states"].find(
        {
            "date": {
                "$gte": pd.Timestamp(start_date).to_pydatetime(),
                "$lte": pd.Timestamp(f"{end_year}-12-31").to_pydatetime(),
            },
            "feature_vector": {"$exists": True},
        },
        {"_id": 0, "date": 1, "feature_vector": 1},
    ).sort("date", 1)
    records = [
        {"date": item["date"], **dict(zip(FEATURE_NAMES, item["feature_vector"]))}
        for item in cursor
        if len(item.get("feature_vector", [])) == len(FEATURE_NAMES)
    ]
    if not records:
        raise ValueError(
            "No historical feature vectors found in MongoDB. Seed regime_states "
            "or pass feature_frame explicitly."
        )
    return pd.DataFrame(records).set_index("date")


def _return_after(prices: pd.Series, date: pd.Timestamp, periods: int) -> float:
    """Return close-to-close performance from date to N trading rows ahead."""
    position = prices.index.searchsorted(date)
    if position >= len(prices) or prices.index[position] != date or position + periods >= len(prices):
        return float("nan")
    return float(prices.iloc[position + periods] / prices.iloc[position] - 1.0)


def _return_ending_at(prices: pd.Series, date: pd.Timestamp) -> float:
    """Return from the previous trading close through ``date``."""
    position = prices.index.searchsorted(date)
    if position <= 0 or position >= len(prices) or prices.index[position] != date:
        return float("nan")
    return float(prices.iloc[position] / prices.iloc[position - 1] - 1.0)
