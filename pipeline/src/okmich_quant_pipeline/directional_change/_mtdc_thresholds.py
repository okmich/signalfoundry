"""
MTDC — Threshold pool generation and top-K selection by GP RMSE.

generate_theta_pool    — build a candidate theta grid.
select_top_thresholds  — rank candidates by GP RMSE, return top-K (theta, stdc_model) pairs.
FitnessMode            — enum of GA fitness objectives: 'sharpe', 'return'.
ConsensusMode          — enum of MTDC consensus rules: 'weight', 'majority'.

Reference: Adegboye, Kampouridis & Otero (Artificial Intelligence Review, 2023) Section 3.1.
Paper default pool: 100 candidates, 0.005 → 0.2525, step 0.0025.
"""
import enum

import numpy as np
import pandas as pd


class FitnessMode(str, enum.Enum):
    """GA fitness objective for MTDC weight training."""
    SHARPE = 'sharpe'
    RETURN = 'return'


class ConsensusMode(str, enum.Enum):
    """Multi-threshold consensus rule for MTDC signal aggregation."""
    WEIGHT = 'weight'
    MAJORITY = 'majority'

from ._cgpts_model import train_cgpts_model

# Paper default pool bounds (Adegboye et al. 2023 Section 3.1)
MTDC_THETA_MIN_DEFAULT = 0.005
MTDC_THETA_MAX_DEFAULT = 0.2525
MTDC_THETA_STEP_DEFAULT = 0.0025


def generate_theta_pool(theta_min: float = MTDC_THETA_MIN_DEFAULT, theta_max: float = MTDC_THETA_MAX_DEFAULT, theta_step: float = MTDC_THETA_STEP_DEFAULT) -> list:
    """
    Generate a grid of candidate DC threshold values.

    Parameters
    ----------
    theta_min : float
        Smallest threshold to include (inclusive).
    theta_max : float
        Largest threshold to include (inclusive up to float precision).
    theta_step : float
        Step size between candidates.

    Returns
    -------
    list of float
        Sorted candidate theta values.
    """
    n = int(round((theta_max - theta_min) / theta_step)) + 1
    return [round(theta_min + i * theta_step, 8) for i in range(n)]


def select_top_thresholds(prices_train: pd.Series, k: int = 5, alpha: float = 1.0, theta_pool: list | None = None, gp_n_generations: int = 37, gp_population_size: int = 500, cx_prob: float = 0.98, mut_prob: float = 0.02, elitism_fraction: float = 0.10, random_seed: int = 42, n_jobs: int = 1) -> list:
    """
    Select top-K DC thresholds from a candidate pool by lowest GP RMSE.

    For each candidate theta, trains a full C+GP+TS model on prices_train and
    records the GP RMSE.  The K thetas with the lowest RMSE are returned along
    with their pre-trained C+GP models (ready for use as STDC components in MTDC).

    Thetas that fail training (too few αDC trends, single-class, etc.) are
    silently skipped and assigned RMSE = inf.

    Parameters
    ----------
    prices_train : pd.Series
        Training price series.
    k : int
        Number of top thresholds to select.
    alpha : float
        Asymmetric attenuation coefficient passed to parse_dc_events.
    theta_pool : list of float, optional
        Explicit candidate theta values.  If None, uses generate_theta_pool()
        with paper defaults (100 candidates, 0.005→0.2525, step 0.0025).
    gp_n_generations : int
        GP generations per candidate.  Reduce for faster screening (e.g. 5).
    gp_population_size : int
        GP population per candidate.  Reduce for faster screening (e.g. 50).
    cx_prob, mut_prob, elitism_fraction : float
        GP operator probabilities passed to train_cgpts_model().
    random_seed : int
        Random seed for reproducibility.
    n_jobs : int
        Parallel workers.  1 = sequential.  -1 = all CPUs.  Requires joblib.

    Returns
    -------
    list of tuple[float, dict]
        Top-K (theta, stdc_model) pairs sorted by GP RMSE ascending.
        len(result) <= k (may be less if fewer than k valid thetas exist).

    Raises
    ------
    ValueError
        If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if theta_pool is None:
        theta_pool = generate_theta_pool()

    def _train_one(theta):
        try:
            model = train_cgpts_model(
                prices_train, theta, alpha,
                n_generations=gp_n_generations, population_size=gp_population_size,
                cx_prob=cx_prob, mut_prob=mut_prob, elitism_fraction=elitism_fraction,
                random_seed=random_seed,
            )
            return theta, model, float(model['gp_rmse'])
        except Exception:
            return theta, None, float('inf')

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(_train_one)(t) for t in theta_pool)
        except ImportError:
            results = [_train_one(t) for t in theta_pool]
    else:
        results = [_train_one(t) for t in theta_pool]

    valid = [(t, m, r) for t, m, r in results if m is not None and np.isfinite(r)]
    valid.sort(key=lambda x: x[2])
    return [(t, m) for t, m, _ in valid[:k]]
