"""
ITAPipeline and run_sliding_window — end-to-end ITA execution.

ITAPipeline orchestrates the four pipeline stages defined in Wu & Han (2023)
Algorithm 1 (top-level steps):
  1. Bayesian optimisation of (θ, α) on training window → optimise_idc_params()
  2. IDC parsing with optimal (θ, α) → idc_parse()
  3. HMM fitting on training RDC sequence → fit_rcd_hmm() / s1_state_index()
  4. ITA Algorithm 1 on test window → run_ita_algorithm1()

run_sliding_window() applies this pipeline across rolling 2-month training /
1-month test windows, as described in Wu & Han (2023) Section 4.1.

References
----------
Wu, Y. & Han, J. (2023). Intelligent Trading Strategy Based on Improved
    Directional Change and Regime Change Detection. arXiv:2309.15383v1.
    Section 4.1 (sliding window protocol), Algorithm 1 (pipeline stages).

Hu, Z., Li, Y. & Wu, Y. (2022). Incorporating Improved Directional Change and
    Regime Change Detection to Formulate Trading Strategies in Foreign Exchange
    Markets. SSRN:4048864. Section 3.1.1 (window specification).
"""
import warnings

import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse

from ._bayesian_opt import ALPHA_BOUNDS, THETA_BOUNDS_TICK, optimise_idc_params
from ._ita_algo1 import run_ita_algorithm1
from ._rcd_hmm import fit_rcd_hmm, s1_state_index

# Minimum bars required in a training or test window
_MIN_TRAIN_BARS = 100
_MIN_TEST_BARS = 10


class ITAPipeline:
    """
    End-to-end ITA pipeline: Bayesian optimisation → IDC → RCD-HMM → Algorithm 1.

    Implements the four pipeline stages of Wu & Han (2023) Algorithm 1.
    Fit on a training price series, then run on a test price series.

    Parameters
    ----------
    theta_bounds : tuple[float, float]
        Search bounds for θ. Use THETA_BOUNDS_TICK (default) or THETA_BOUNDS_5MIN.
    alpha_bounds : tuple[float, float]
        Search bounds for α. Default: ALPHA_BOUNDS = (0.10, 1.00).
    n_calls : int
        Total Bayesian optimisation evaluations. Wu & Han (2023) default: 100.
    n_initial : int
        Initial random evaluations before GP surrogate. Default: 10.
    random_state : int
        Seed for reproducibility of Bayesian optimisation.
    hmm_random_state : int
        Seed for HMM fitting reproducibility.

    Attributes (set after fit())
    ----------------------------
    theta_ : float
        Optimised DC upward threshold.
    alpha_ : float
        Optimised asymmetric attenuation coefficient.
    hmm_ : PomegranateHMM
        Fitted HMM trained on training window RDC sequence.
    s1_idx_ : int
        HMM state index for S1 (normal regime).
    """

    def __init__(self, theta_bounds: tuple[float, float] = THETA_BOUNDS_TICK, alpha_bounds: tuple[float, float] = ALPHA_BOUNDS, n_calls: int = 100, n_initial: int = 10, random_state: int = 42, hmm_random_state: int = 42):
        self.theta_bounds = theta_bounds
        self.alpha_bounds = alpha_bounds
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.random_state = random_state
        self.hmm_random_state = hmm_random_state

        self.theta_: float | None = None
        self.alpha_: float | None = None
        self.hmm_ = None
        self.s1_idx_: int | None = None
        self._is_fitted = False

    def fit(self, prices_train: pd.Series) -> "ITAPipeline":
        """
        Fit the ITA pipeline on a training price series.

        Runs Bayesian optimisation → IDC parsing → HMM fitting.

        Parameters
        ----------
        prices_train : pd.Series
            Close price series for the training window, in chronological order.

        Returns
        -------
        self
        """
        # Stage 1: Bayesian optimisation
        self.theta_, self.alpha_ = optimise_idc_params(
            prices_train,
            theta_bounds=self.theta_bounds,
            alpha_bounds=self.alpha_bounds,
            n_calls=self.n_calls,
            n_initial=self.n_initial,
            random_state=self.random_state,
        )

        # Stage 2: IDC parsing with optimal parameters
        idc = idc_parse(prices_train, self.theta_, self.alpha_)

        # Stage 3: HMM fitting on training RDC sequence
        rdc_train = idc["rdc"].dropna()
        self.hmm_ = fit_rcd_hmm(rdc_train, random_state=self.hmm_random_state)
        self.s1_idx_ = s1_state_index(self.hmm_)

        self._is_fitted = True
        return self

    def run(self, prices_test: pd.Series, initial_capital: float = 10_000.0) -> dict:
        """
        Execute ITA Algorithm 1 on a test price series.

        Parameters
        ----------
        prices_test : pd.Series
            Close price series for the test window, in chronological order.
        initial_capital : float
            Starting capital. Wu & Han (2023) paper default: 10,000 EUR.

        Returns
        -------
        dict
            All keys from run_ita_algorithm1() plus:
            optimal_theta : float — the Bayesian-optimised θ.
            optimal_alpha : float — the Bayesian-optimised α.
        """
        self._check_fitted()
        result = run_ita_algorithm1(
            prices_test,
            theta=self.theta_,
            alpha=self.alpha_,
            hmm=self.hmm_,
            s1_idx=self.s1_idx_,
            initial_capital=initial_capital,
        )
        result["optimal_theta"] = self.theta_
        result["optimal_alpha"] = self.alpha_
        return result

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ITAPipeline is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        state = "fitted" if self._is_fitted else "unfitted"
        return (f"ITAPipeline("
                f"theta_bounds={self.theta_bounds}, "
                f"alpha_bounds={self.alpha_bounds}, "
                f"n_calls={self.n_calls}, "
                f"state={state})")


def run_sliding_window(prices: pd.Series, train_months: int = 2, test_months: int = 1, initial_capital: float = 10_000.0, compound: bool = False, **pipeline_kwargs) -> pd.DataFrame:
    """
    Execute the ITA pipeline across rolling monthly windows.

    Implements the sliding window protocol of Wu & Han (2023) Section 4.1:
    2-month training window, 1-month test window, 1-month stride.

    Parameters
    ----------
    prices : pd.Series
        Close price series with a DatetimeIndex, in chronological order.
    train_months : int
        Number of months in each training window. Wu & Han (2023) default: 2.
    test_months : int
        Number of months in each test window. Wu & Han (2023) default: 1.
    initial_capital : float
        Starting capital for the first window. Wu & Han (2023) default: 10,000 EUR.
    compound : bool
        If True, carry the final capital of each test window forward as the
        initial capital for the next window (compounding). If False (default),
        each window is evaluated independently with the same initial capital.
    **pipeline_kwargs
        Additional keyword arguments forwarded to ITAPipeline constructor
        (e.g. theta_bounds, alpha_bounds, n_calls, random_state).

    Returns
    -------
    pd.DataFrame
        One row per completed test window with columns:
        - test_start       : str   — first date of test window (YYYY-MM-DD)
        - test_end         : str   — last date of test window (YYYY-MM-DD)
        - optimal_theta    : float
        - optimal_alpha    : float
        - n_trades         : int
        - n_winners        : int
        - win_ratio        : float — n_winners / n_trades, 0 if no trades
        - cumulative_return: float — CRR % for this window
        - max_drawdown     : float — MDD % for this window
        - profit_factor    : float — gross profit / gross loss, inf if no losses
        - sharpe           : float — mean(pnl) / std(pnl) across trades, 0 if < 2
        - final_capital    : float — capital at end of test window

    Raises
    ------
    TypeError
        If prices does not have a DatetimeIndex.

    Notes
    -----
    - Windows with fewer than 100 training bars or 10 test bars are skipped.
    - Stride is always 1 month (test_months bars advanced per iteration).
    - Wu & Han (2023) use compound=False for per-window CRR reporting but
      compound=True reflects realistic capital deployment.

    References
    ----------
    Wu & Han (2023) Section 4.1.
    Hu et al. (2022) Section 3.1.1.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex for monthly window slicing.")

    months = prices.index.to_period("M").unique()
    total_months = len(months)
    window_size = train_months + test_months  # total months per window

    if total_months < window_size:
        return pd.DataFrame()

    capital = initial_capital
    results = []

    for i in range(total_months - window_size + 1):
        train_start_period = months[i]
        train_end_period = months[i + train_months - 1]
        test_start_period = months[i + train_months]
        test_end_period = months[i + train_months + test_months - 1]

        train_mask = (prices.index >= train_start_period.start_time) & (prices.index <= train_end_period.end_time)
        test_mask = (prices.index >= test_start_period.start_time) & (prices.index <= test_end_period.end_time)

        prices_train = prices.loc[train_mask]
        prices_test = prices.loc[test_mask]

        if len(prices_train) < _MIN_TRAIN_BARS or len(prices_test) < _MIN_TEST_BARS:
            continue

        try:
            pipeline = ITAPipeline(**pipeline_kwargs)
            pipeline.fit(prices_train)
            window_result = pipeline.run(prices_test, initial_capital=capital)
        except Exception as exc:
            warnings.warn(f"Window {months[i]} skipped due to error: {exc!r}", stacklevel=2)
            continue

        results.append({
            "test_start": str(prices_test.index[0].date()),
            "test_end": str(prices_test.index[-1].date()),
            "optimal_theta": window_result["optimal_theta"],
            "optimal_alpha": window_result["optimal_alpha"],
            "n_trades": window_result["n_trades"],
            "n_winners": window_result["n_winners"],
            "win_ratio": window_result["win_ratio"],
            "cumulative_return": window_result["cumulative_return"],
            "max_drawdown": window_result["max_drawdown"],
            "profit_factor": window_result["profit_factor"],
            "sharpe": window_result["sharpe"],
            "final_capital": window_result["final_capital"],
        })

        if compound:
            capital = window_result["final_capital"]

    return pd.DataFrame(results)
