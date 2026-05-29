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
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    # TODO: loop n_restarts times with different random_state values
    # TODO: fit GaussianHMM(**HMM_CONFIG) on feature_matrix
    # TODO: track log-likelihood via model.score(feature_matrix)
    # TODO: return the model with the best score
    # TODO: log a ConvergenceWarning if model.monitor_.converged is False
    raise NotImplementedError


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
    # TODO: build means_df = pd.DataFrame(model.means_, columns=FEATURE_NAMES)
    # TODO: identify risk-on state: argmax of xlv_ret_5d column
    # TODO: identify catalyst-fear state: argmax of xlv_vol_20d column
    # TODO: remaining state = neutral
    # TODO: handle edge case where same state wins both criteria (log a warning)
    raise NotImplementedError


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
    # TODO: create MODEL_DIR if it doesn't exist
    # TODO: bundle {model, scaler, state_label_map, train_end_date} into a dict
    # TODO: pickle.dump to MODEL_DIR / f"hmm_{train_end_date}.pkl"
    # TODO: also save as MODEL_DIR / "hmm_latest.pkl" (overwrite)
    raise NotImplementedError


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
    # TODO: resolve the .pkl file path based on checkpoint argument
    # TODO: pickle.load the checkpoint dict
    # TODO: return (checkpoint["model"], checkpoint["scaler"], checkpoint["state_label_map"])
    raise NotImplementedError


def run_walk_forward_training(
    db,
    start_date: str = "2015-01-01",
    test_years: list[int] | None = None,
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
    # TODO: default test_years = [2020, 2021, 2022, 2023, 2024]
    # TODO: for each test_year:
    #   1. train_end = f"{test_year - 1}-12-31"
    #   2. build feature matrix for 2015-01-01 → train_end
    #   3. call train_hmm() and save_model()
    #   4. for each month in test_year, refit (expanding window)
    #   5. predict daily regime for the month
    #   6. look up actual XLV 10-day forward return for each date
    # TODO: aggregate results into DataFrame and return
    raise NotImplementedError
