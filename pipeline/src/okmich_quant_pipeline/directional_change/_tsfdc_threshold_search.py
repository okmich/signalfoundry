"""
TSFDC — Optimal STheta/BTheta threshold search utility.

evaluate_threshold_pair  — evaluate one (stheta, btheta) pair and return metrics.
search_optimal_thresholds — grid search over stheta/btheta pairs with optional parallelism.

Reference: Bakhach, Tsang & Chinthalapati (ISAFM, 2018) Section 16 (experimental config).
"""
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from ._tsfdc_pipeline import TSFDCPipeline


def evaluate_threshold_pair(prices_train: pd.Series, prices_test: pd.Series, stheta: float, btheta: float, initial_capital: float = 10_000.0, min_trades: int = 5, random_seed: int = 42) -> dict:
    """
    Evaluate one (stheta, btheta) pair on a train/test split.

    Returns a flat metrics dict.  If the pair fails (too few DC events, only one
    class, or fewer than min_trades trades per direction), metrics are set to
    NaN/0 and 'valid' is False.

    Parameters
    ----------
    prices_train : pd.Series
        Training price series.
    prices_test : pd.Series
        Test price series.
    stheta : float
        Small DC threshold.
    btheta : float
        Big DC threshold (must be > stheta).
    initial_capital : float
        Starting capital for simulation.
    min_trades : int
        Minimum number of trades (per direction) required for the pair to be
        considered valid.  Pairs below this threshold have 'valid' = False.
    random_seed : int
        Random seed for the classifier.

    Returns
    -------
    dict
        stheta, btheta, valid, plus per-direction columns:
        rr_down, rr_up, mdd_down, mdd_up, n_trades_down, n_trades_up,
        win_ratio_down, win_ratio_up, profit_factor_down, profit_factor_up.
    """
    base = {'stheta': stheta, 'btheta': btheta}

    if btheta <= stheta:
        return {**base, 'valid': False,
                **_nan_metrics('down'), **_nan_metrics('up')}

    try:
        pipeline = TSFDCPipeline(stheta=stheta, btheta=btheta, random_seed=random_seed)
        pipeline.fit(prices_train)
        result = pipeline.run(prices_test, initial_capital_down=initial_capital, initial_capital_up=initial_capital)
    except (ValueError, RuntimeError, ImportError):
        return {**base, 'valid': False,
                **_nan_metrics('down'), **_nan_metrics('up')}

    r_d = result['down']
    r_u = result['up']

    valid = (r_d['n_trades'] >= min_trades and r_u['n_trades'] >= min_trades)

    return {
        **base,
        'valid': valid,
        'rr_down': r_d['cumulative_return'],
        'rr_up': r_u['cumulative_return'],
        'mdd_down': r_d['max_drawdown'],
        'mdd_up': r_u['max_drawdown'],
        'n_trades_down': r_d['n_trades'],
        'n_trades_up': r_u['n_trades'],
        'win_ratio_down': r_d['win_ratio'],
        'win_ratio_up': r_u['win_ratio'],
        'profit_factor_down': r_d['profit_factor'],
        'profit_factor_up': r_u['profit_factor'],
    }


def search_optimal_thresholds(prices_train: pd.Series, prices_test: pd.Series, stheta_values: Sequence[float], btheta_values: Sequence[float], initial_capital: float = 10_000.0, min_trades: int = 5, random_seed: int = 42, n_jobs: int = 1) -> pd.DataFrame:
    """
    Grid search over (stheta, btheta) pairs and return a full metrics table.

    Only pairs where btheta > stheta are evaluated.  Pairs that produce fewer
    than min_trades trades in either direction are marked valid=False but are
    still included in the returned DataFrame.

    Parameters
    ----------
    prices_train : pd.Series
        Training price series (used to fit the BBTheta classifier).
    prices_test : pd.Series
        Test price series (used to evaluate trading performance).
    stheta_values : sequence of float
        Grid of small threshold values to search.
    btheta_values : sequence of float
        Grid of big threshold values to search.
    initial_capital : float
        Starting capital for simulation.
    min_trades : int
        Minimum trades per direction for a pair to be considered valid.
    random_seed : int
        Random seed for all classifiers.
    n_jobs : int
        Number of parallel workers.  1 = sequential.  -1 = all available CPUs.
        Requires joblib to be installed for n_jobs != 1.

    Returns
    -------
    pd.DataFrame
        One row per (stheta, btheta) pair, sorted by combined return
        (rr_down + rr_up) descending.  Includes all pairs, both valid and
        invalid.  Columns: stheta, btheta, valid, rr_down, rr_up, mdd_down,
        mdd_up, n_trades_down, n_trades_up, win_ratio_down, win_ratio_up,
        profit_factor_down, profit_factor_up.
    """
    pairs = [(s, b) for s, b in product(stheta_values, btheta_values) if b > s]

    def _eval(s, b):
        return evaluate_threshold_pair(
            prices_train, prices_test, s, b,
            initial_capital=initial_capital, min_trades=min_trades, random_seed=random_seed,
        )

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=n_jobs)(delayed(_eval)(s, b) for s, b in pairs)
        except ImportError:
            rows = [_eval(s, b) for s, b in pairs]
    else:
        rows = [_eval(s, b) for s, b in pairs]

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Add combined return column for sorting
    df['rr_combined'] = df['rr_down'].fillna(0) + df['rr_up'].fillna(0)
    df = df.sort_values('rr_combined', ascending=False).drop(columns='rr_combined')
    df = df.reset_index(drop=True)
    return df


def _nan_metrics(direction: str) -> dict:
    return {
        f'rr_{direction}': np.nan,
        f'mdd_{direction}': np.nan,
        f'n_trades_{direction}': 0,
        f'win_ratio_{direction}': np.nan,
        f'profit_factor_{direction}': np.nan,
    }
