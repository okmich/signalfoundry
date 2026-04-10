"""
MTDC pipeline — high-level orchestrator, sliding window, and K search utility.

MTDCPipeline            — fit (threshold selection + GA) and run on test data.
run_mtdc_sliding_window — rolling walk-forward backtest (2-month train / 1-month test).
search_optimal_k        — evaluate k=1..k_max on a train/test split, return metrics table.

Reference: Adegboye, Kampouridis & Otero (AI Review, 2023) Sections 3–4, 20.
"""
import warnings

import numpy as np
import pandas as pd

from okmich_quant_ml.hmm import PomegranateHMM

from ._mtdc_engine import run_mtdc_algorithm
from ._mtdc_ga import train_ga_weights
from ._mtdc_thresholds import ConsensusMode, FitnessMode, generate_theta_pool, select_top_thresholds
from ._rcd_hmm import s1_state_index


class MTDCPipeline:
    """
    Full MTDC pipeline: threshold selection → GA weight training → trading engine.

    Implements Adegboye, Kampouridis & Otero (2023):
      1. fit():  select top-K thetas by GP RMSE, evolve GA weights.
      2. run():  apply K STDC models + GA weights + ITA regime gate to test data.

    Supports both explicit theta lists and auto-generated pools.

    Parameters
    ----------
    k : int
        Number of thresholds.  Paper default: 5.
    alpha : float
        Asymmetric DC attenuation coefficient.
    theta_pool : list of float, optional
        Explicit candidate pool.  None → auto-generate with paper defaults.
    thetas : list of float, optional
        Explicit theta values to use directly (skips threshold selection).
        Cannot be combined with theta_pool.
    gp_n_generations : int
        GP generations for threshold screening.
    gp_population_size : int
        GP population for threshold screening.
    ga_population_size : int
        GA population for weight optimisation.  Paper default: 500.
    ga_n_generations : int
        GA generations.  Paper default: 50.
    fitness_mode : FitnessMode or str
        FitnessMode.SHARPE or FitnessMode.RETURN.
    consensus_mode : ConsensusMode or str
        ConsensusMode.WEIGHT or ConsensusMode.MAJORITY.
    majority_weight : float
        Supermajority threshold when consensus_mode=ConsensusMode.MAJORITY.
    n_jobs : int
        Parallel workers for threshold screening.
    random_seed : int
        Random seed.
    """

    def __init__(self, k: int = 5, alpha: float = 1.0, theta_pool: list | None = None, thetas: list | None = None, gp_n_generations: int = 37, gp_population_size: int = 500, ga_population_size: int = 500, ga_n_generations: int = 50, ga_tournament_size: int = 7, ga_cx_prob: float = 0.90, ga_mut_prob: float = 0.10, ga_elitism_fraction: float = 0.10, fitness_mode: FitnessMode | str = FitnessMode.SHARPE, consensus_mode: ConsensusMode | str = ConsensusMode.WEIGHT, majority_weight: float = 0.5, n_jobs: int = 1, random_seed: int = 42):
        if theta_pool is not None and thetas is not None:
            raise ValueError("Cannot specify both theta_pool and thetas.")
        self.k = k
        self.alpha = alpha
        self.theta_pool = theta_pool
        self._explicit_thetas = thetas
        self.gp_n_generations = gp_n_generations
        self.gp_population_size = gp_population_size
        self.ga_population_size = ga_population_size
        self.ga_n_generations = ga_n_generations
        self.ga_tournament_size = ga_tournament_size
        self.ga_cx_prob = ga_cx_prob
        self.ga_mut_prob = ga_mut_prob
        self.ga_elitism_fraction = ga_elitism_fraction
        self.fitness_mode = fitness_mode
        self.consensus_mode = consensus_mode
        self.majority_weight = majority_weight
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        self.thetas_: list | None = None
        self.stdc_models_: list | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, prices_train: pd.Series) -> 'MTDCPipeline':
        """
        Select top-K thresholds and evolve GA weights on training data.

        Parameters
        ----------
        prices_train : pd.Series
            Training price series.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If threshold selection yields fewer than 1 valid model.
        """
        if self._explicit_thetas is not None:
            # Use explicit thetas: train each STDC model directly
            from ._cgpts_model import train_cgpts_model
            pairs = []
            for theta in self._explicit_thetas:
                try:
                    model = train_cgpts_model(
                        prices_train, theta, self.alpha,
                        n_generations=self.gp_n_generations,
                        population_size=self.gp_population_size,
                        random_seed=self.random_seed,
                    )
                    pairs.append((theta, model))
                except Exception:
                    pass
            if not pairs:
                raise ValueError("All explicit thetas failed to train. Check data length and theta values.")
        else:
            pairs = select_top_thresholds(
                prices_train, k=self.k, alpha=self.alpha,
                theta_pool=self.theta_pool,
                gp_n_generations=self.gp_n_generations,
                gp_population_size=self.gp_population_size,
                random_seed=self.random_seed, n_jobs=self.n_jobs,
            )
            if not pairs:
                raise ValueError("Threshold selection yielded no valid models. Use more data or a different theta pool.")

        self.thetas_ = [t for t, _ in pairs]
        self.stdc_models_ = [m for _, m in pairs]

        self.weights_ = train_ga_weights(
            prices_train, self.stdc_models_, self.thetas_,
            alpha=self.alpha,
            population_size=self.ga_population_size,
            n_generations=self.ga_n_generations,
            tournament_size=self.ga_tournament_size,
            cx_prob=self.ga_cx_prob, mut_prob=self.ga_mut_prob,
            elitism_fraction=self.ga_elitism_fraction,
            fitness_mode=self.fitness_mode, random_seed=self.random_seed,
        )
        return self

    def run(self, prices_test: pd.Series, hmm: PomegranateHMM, s1_idx: int, initial_capital: float = 10_000.0) -> dict:
        """
        Run MTDC on a test price series using fitted models and GA weights.

        Parameters
        ----------
        prices_test : pd.Series
            Test price series.
        hmm : PomegranateHMM
            Fitted 2-state HMM from the training window.
        s1_idx : int
            HMM state index for S1 from s1_state_index().
        initial_capital : float
            Starting capital.

        Returns
        -------
        dict
            Same structure as run_mtdc_algorithm().

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self.stdc_models_ is None:
            raise RuntimeError("MTDCPipeline.fit() must be called before run().")
        return run_mtdc_algorithm(
            prices_test, hmm, s1_idx,
            self.stdc_models_, self.thetas_, self.weights_,
            alpha=self.alpha, initial_capital=initial_capital,
            consensus_mode=self.consensus_mode, majority_weight=self.majority_weight,
        )


def search_optimal_k(prices_train: pd.Series, prices_test: pd.Series, hmm: PomegranateHMM, s1_idx: int, k_max: int = 10, alpha: float = 1.0, theta_pool: list | None = None, gp_n_generations: int = 37, gp_population_size: int = 500, ga_population_size: int = 500, ga_n_generations: int = 50, fitness_mode: FitnessMode | str = FitnessMode.SHARPE, consensus_mode: ConsensusMode | str = ConsensusMode.WEIGHT, majority_weight: float = 0.5, initial_capital: float = 10_000.0, n_jobs: int = 1, random_seed: int = 42) -> pd.DataFrame:
    """
    Evaluate MTDC performance for k=1..k_max on a train/test split.

    Trains the top-k_max threshold models once, then for each k:
      1. Takes the top-k (best by GP RMSE) subset.
      2. Trains GA weights for that subset.
      3. Runs the strategy on prices_test.
      4. Records performance metrics.

    Parameters
    ----------
    prices_train : pd.Series
        Training price series.
    prices_test : pd.Series
        Held-out test price series for performance evaluation.
    hmm : PomegranateHMM
        Fitted HMM (from the training window).
    s1_idx : int
        HMM S1 state index.
    k_max : int
        Maximum K to evaluate.
    alpha : float
        Asymmetric DC coefficient.
    theta_pool : list of float, optional
        Candidate theta pool.  None → auto-generate.
    gp_n_generations, gp_population_size : int
        GP parameters for threshold screening.
    ga_population_size, ga_n_generations : int
        GA parameters for weight training.
    fitness_mode : FitnessMode or str
        FitnessMode.SHARPE or FitnessMode.RETURN.
    consensus_mode : ConsensusMode or str
        ConsensusMode.WEIGHT or ConsensusMode.MAJORITY.
    majority_weight : float
        Supermajority threshold (for consensus_mode='majority').
    initial_capital : float
        Starting capital per simulation.
    n_jobs : int
        Parallel workers for threshold screening.
    random_seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per k, sorted by k ascending.  Columns: k, thetas,
        cumulative_return, max_drawdown, n_trades, win_ratio, sharpe,
        profit_factor.
    """
    # Train top-k_max models once
    top_pairs = select_top_thresholds(
        prices_train, k=k_max, alpha=alpha, theta_pool=theta_pool,
        gp_n_generations=gp_n_generations, gp_population_size=gp_population_size,
        random_seed=random_seed, n_jobs=n_jobs,
    )

    if not top_pairs:
        return pd.DataFrame()

    rows = []
    for k in range(1, len(top_pairs) + 1):
        subset = top_pairs[:k]
        thetas_k = [t for t, _ in subset]
        models_k = [m for _, m in subset]

        try:
            weights_k = train_ga_weights(
                prices_train, models_k, thetas_k, alpha=alpha,
                population_size=ga_population_size, n_generations=ga_n_generations,
                fitness_mode=fitness_mode, random_seed=random_seed,
            )
            result = run_mtdc_algorithm(
                prices_test, hmm, s1_idx, models_k, thetas_k, weights_k,
                alpha=alpha, initial_capital=initial_capital,
                consensus_mode=consensus_mode, majority_weight=majority_weight,
            )
        except Exception as exc:
            warnings.warn(f"k={k} evaluation failed: {exc!r}", stacklevel=2)
            result = None

        rows.append({
            'k': k,
            'thetas': thetas_k,
            'cumulative_return': result['cumulative_return'] if result else np.nan,
            'max_drawdown': result['max_drawdown'] if result else np.nan,
            'n_trades': result['n_trades'] if result else 0,
            'win_ratio': result['win_ratio'] if result else np.nan,
            'sharpe': result['sharpe'] if result else np.nan,
            'profit_factor': result['profit_factor'] if result else np.nan,
        })

    return pd.DataFrame(rows)


def run_mtdc_sliding_window(prices: pd.Series, hmm_train_func, stheta_for_hmm: float = 0.01, k: int = 5, alpha: float = 1.0, theta_pool: list | None = None, gp_n_generations: int = 37, gp_population_size: int = 500, ga_population_size: int = 500, ga_n_generations: int = 50, fitness_mode: FitnessMode | str = FitnessMode.SHARPE, consensus_mode: ConsensusMode | str = ConsensusMode.WEIGHT, majority_weight: float = 0.5, initial_capital: float = 10_000.0, train_months: int = 2, n_jobs: int = 1, random_seed: int = 42) -> pd.DataFrame:
    """
    Execute MTDC across rolling walk-forward windows.

    At each window: trains HMM on training data, fits MTDCPipeline, runs on test month.
    Capital is compounded across windows.

    Parameters
    ----------
    prices : pd.Series
        Full price series with DatetimeIndex.
    hmm_train_func : callable
        Function ``hmm_train_func(prices_train) -> (hmm, s1_idx)``
        that returns a fitted HMM and its S1 state index.
        Typically a thin wrapper around fit_rcd_hmm + s1_state_index.
    stheta_for_hmm : float
        Theta used internally by hmm_train_func (informational only; logic lives
        inside the callable).
    k : int
        Number of MTDC thresholds.
    alpha : float
        Asymmetric DC coefficient.
    theta_pool : list, optional
        Candidate pool.  None → auto-generate.
    gp_n_generations, gp_population_size : int
        GP screening parameters.
    ga_population_size, ga_n_generations : int
        GA optimisation parameters.
    fitness_mode : FitnessMode or str
        FitnessMode.SHARPE or FitnessMode.RETURN.
    consensus_mode : ConsensusMode or str
        ConsensusMode.WEIGHT or ConsensusMode.MAJORITY.
    majority_weight : float
        Supermajority threshold for 'majority' mode.
    initial_capital : float
        Starting capital.
    train_months : int
        Training window length in months.  Default 2 (ITA convention).
    n_jobs : int
        Parallel workers for threshold screening.
    random_seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per test month with columns: test_month, cumulative_return,
        max_drawdown, n_trades, win_ratio, sharpe, profit_factor, capital.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex.")

    months = prices.index.to_period('M').unique()
    if len(months) < train_months + 1:
        raise ValueError(f"Need at least {train_months + 1} months of data.")

    results = []
    capital = float(initial_capital)

    for i in range(train_months, len(months)):
        train_start = months[i - train_months].start_time
        train_end = months[i].start_time
        test_start = months[i].start_time
        test_end = months[i].end_time + pd.Timedelta(days=1)

        train_mask = (prices.index >= train_start) & (prices.index < train_end)
        test_mask = (prices.index >= test_start) & (prices.index < test_end)

        prices_train = prices[train_mask]
        prices_test = prices[test_mask]

        if len(prices_train) < 200 or len(prices_test) < 10:
            continue

        try:
            hmm, s1_idx = hmm_train_func(prices_train)
            pipeline = MTDCPipeline(
                k=k, alpha=alpha, theta_pool=theta_pool,
                gp_n_generations=gp_n_generations, gp_population_size=gp_population_size,
                ga_population_size=ga_population_size, ga_n_generations=ga_n_generations,
                fitness_mode=fitness_mode, consensus_mode=consensus_mode,
                majority_weight=majority_weight, n_jobs=n_jobs, random_seed=random_seed,
            )
            pipeline.fit(prices_train)
            result = pipeline.run(prices_test, hmm, s1_idx, initial_capital=capital)
        except Exception as exc:
            warnings.warn(f"Window {months[i]} skipped due to error: {exc!r}", stacklevel=2)
            continue

        results.append({
            'test_month': str(months[i]),
            'cumulative_return': result['cumulative_return'],
            'max_drawdown': result['max_drawdown'],
            'n_trades': result['n_trades'],
            'win_ratio': result['win_ratio'],
            'sharpe': result['sharpe'],
            'profit_factor': result['profit_factor'],
            'capital': result['final_capital'],
        })
        capital = result['final_capital']

    return pd.DataFrame(results)
