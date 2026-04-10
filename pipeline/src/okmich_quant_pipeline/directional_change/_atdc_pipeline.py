"""
ATDC pipeline — high-level orchestrator and sliding window.

ATDCPipeline           — fit (theta_init optimisation + optional HMM) and run on test data.
run_atdc_sliding_window — rolling walk-forward backtest (default 2-month train / 1-month test).

Reference: Adaptive extension of Wu & Han (2023) ITA, generalised to both
directions and pluggable adaptation modes.
"""
import warnings

import numpy as np
import pandas as pd

from ._atdc_adapter import AdaptationMode
from ._atdc_engine import run_atdc_algorithm
from ._bayesian_opt import ALPHA_BOUNDS, THETA_BOUNDS_TICK, optimise_idc_params
from ._rcd_hmm import fit_rcd_hmm, s1_state_index

from okmich_quant_features.directional_change import idc_parse


class ATDCPipeline:
    """
    Adaptive Threshold DC pipeline: theta optimisation → optional HMM → ATDC engine.

    Fit phase:
      1. Bayesian optimisation of (theta_init, alpha) on training data.
      2. Optional HMM fitting on training RDC sequence.
      3. Optional C+GP model training (when use_gp=True).

    Run phase:
      Execute run_atdc_algorithm() on test data using fitted parameters, with
      theta adapting every `adaptation_step` bars according to `adaptation_mode`.

    Parameters
    ----------
    adaptation_mode : str or AdaptationMode
        'volatility' (default), 'rdc', 'tmv', or 'custom'.
    adaptation_rate : float
        Theta sensitivity to metric deviations. Default: 0.5.
    lookback_window : int
        Bars in adaptation lookback. Default: 100.
    adaptation_step : int
        Bars between theta updates. Default: 50.
    theta_min : float
        Lower theta bound. Default: 0.001.
    theta_max : float
        Upper theta bound. Default: 0.50.
    alpha : float or None
        Fixed asymmetric coefficient. None → Bayesian-optimised alongside theta.
    use_hmm : bool
        Fit and apply HMM regime gate. Default: True.
    use_gp : bool
        Train and apply C+GP model for long exit timing. Default: False.
    theta_bounds : tuple[float, float]
        Search range for Bayesian theta optimisation.
    alpha_bounds : tuple[float, float]
        Search range for Bayesian alpha optimisation (ignored when alpha is set).
    n_calls : int
        Bayesian optimisation evaluations. Default: 100.
    n_initial : int
        Initial random evaluations. Default: 10.
    random_state : int
        Reproducibility seed.
    custom_fn : callable or None
        Custom metric function (mode='custom' only).
    """

    def __init__(self, adaptation_mode: str = 'volatility', adaptation_rate: float = 0.5, lookback_window: int = 100, adaptation_step: int = 50, theta_min: float = 0.001, theta_max: float = 0.50, alpha: float | None = None, use_hmm: bool = True, use_gp: bool = False, theta_bounds: tuple = THETA_BOUNDS_TICK, alpha_bounds: tuple = ALPHA_BOUNDS, n_calls: int = 100, n_initial: int = 10, random_state: int = 42, custom_fn=None):
        self.adaptation_mode = AdaptationMode(adaptation_mode)
        self.adaptation_rate = adaptation_rate
        self.lookback_window = lookback_window
        self.adaptation_step = adaptation_step
        self.theta_min = theta_min
        self.theta_max = theta_max
        self._fixed_alpha = alpha
        self.use_hmm = use_hmm
        self.use_gp = use_gp
        self.theta_bounds = theta_bounds
        self.alpha_bounds = alpha_bounds
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.random_state = random_state
        self.custom_fn = custom_fn

        self.theta_init_: float | None = None
        self.alpha_: float | None = None
        self.hmm_ = None
        self.s1_idx_: int | None = None
        self.cgpts_model_: dict | None = None
        self._is_fitted = False

    def fit(self, prices_train: pd.Series) -> 'ATDCPipeline':
        """
        Fit ATDC pipeline on training data.

        Runs Bayesian optimisation to find theta_init (and alpha if not fixed),
        then optionally fits an HMM and a C+GP model.

        Parameters
        ----------
        prices_train : pd.Series
            Training price series.

        Returns
        -------
        self
        """
        # Stage 1: optimise theta_init (and alpha if not fixed)
        if self._fixed_alpha is not None:
            self.alpha_ = float(self._fixed_alpha)
            # Optimize theta only; alpha is held fixed in the simulation
            from ._ita_sim import run_ita_simulation
            from okmich_quant_features.directional_change import idc_parse as _idc_parse
            from skopt import gp_minimize
            from skopt.space import Real

            def _objective(params):
                theta = params[0]
                try:
                    idc = _idc_parse(prices_train, theta, self.alpha_)
                    res = run_ita_simulation(idc, prices_train, theta)
                    return -res['cumulative_return']
                except Exception:
                    return 1e6  # penalise degenerate threshold strongly

            space = [Real(self.theta_bounds[0], self.theta_bounds[1], name='theta')]
            opt = gp_minimize(_objective, space, n_calls=self.n_calls, n_initial_points=self.n_initial, random_state=self.random_state)
            self.theta_init_ = float(opt.x[0])
        else:
            self.theta_init_, self.alpha_ = optimise_idc_params(
                prices_train,
                theta_bounds=self.theta_bounds,
                alpha_bounds=self.alpha_bounds,
                n_calls=self.n_calls,
                n_initial=self.n_initial,
                random_state=self.random_state,
            )

        # Clamp theta_init to [theta_min, theta_max]
        self.theta_init_ = float(np.clip(self.theta_init_, self.theta_min, self.theta_max))

        # Stage 2: optional HMM
        if self.use_hmm:
            idc = idc_parse(prices_train, self.theta_init_, self.alpha_)
            rdc_train = idc['rdc'].dropna()
            self.hmm_ = fit_rcd_hmm(rdc_train, random_state=self.random_state)
            self.s1_idx_ = s1_state_index(self.hmm_)

        # Stage 3: optional C+GP
        if self.use_gp:
            from ._cgpts_model import train_cgpts_model
            try:
                self.cgpts_model_ = train_cgpts_model(
                    prices_train, self.theta_init_, self.alpha_,
                    random_seed=self.random_state,
                )
            except Exception:
                self.cgpts_model_ = None

        self._is_fitted = True
        return self

    def run(self, prices_test: pd.Series, initial_capital: float = 10_000.0) -> dict:
        """
        Execute ATDC on a test price series using fitted parameters.

        Parameters
        ----------
        prices_test : pd.Series
            Test price series.
        initial_capital : float
            Starting capital.

        Returns
        -------
        dict
            All keys from run_atdc_algorithm() plus optimal_theta, optimal_alpha.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("ATDCPipeline.fit() must be called before run().")

        result = run_atdc_algorithm(
            prices_test,
            theta_init=self.theta_init_,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            adaptation_rate=self.adaptation_rate,
            lookback_window=self.lookback_window,
            adaptation_step=self.adaptation_step,
            adaptation_mode=self.adaptation_mode,
            alpha=self.alpha_,
            hmm=self.hmm_,
            s1_idx=self.s1_idx_ if self.s1_idx_ is not None else 0,
            use_gp=self.use_gp,
            cgpts_model=self.cgpts_model_,
            initial_capital=initial_capital,
            custom_fn=self.custom_fn,
        )
        result['optimal_theta_init'] = self.theta_init_
        result['optimal_alpha'] = self.alpha_
        return result


def run_atdc_sliding_window(prices: pd.Series, adaptation_mode: str = 'volatility', adaptation_rate: float = 0.5, lookback_window: int = 100, adaptation_step: int = 50, theta_min: float = 0.001, theta_max: float = 0.50, alpha: float | None = None, use_hmm: bool = True, use_gp: bool = False, theta_bounds: tuple = THETA_BOUNDS_TICK, alpha_bounds: tuple = ALPHA_BOUNDS, n_calls: int = 100, n_initial: int = 10, initial_capital: float = 10_000.0, train_months: int = 2, compound: bool = True, random_state: int = 42, custom_fn=None) -> pd.DataFrame:
    """
    Execute ATDC across rolling walk-forward windows.

    At each window: fits ATDCPipeline on training data, runs on test month.
    Capital is optionally compounded across windows.

    Parameters
    ----------
    prices : pd.Series
        Full price series with DatetimeIndex.
    adaptation_mode : str
        Theta adaptation mode.
    adaptation_rate : float
        Sensitivity of theta to metric.
    lookback_window : int
        Bars in adaptation lookback.
    adaptation_step : int
        Bars between theta updates.
    theta_min : float
        Lower theta bound.
    theta_max : float
        Upper theta bound.
    alpha : float or None
        Fixed alpha. None → Bayesian-optimised.
    use_hmm : bool
        Apply HMM regime gate.
    use_gp : bool
        Use C+GP for long exit timing.
    theta_bounds : tuple
        Bayesian search bounds for theta.
    alpha_bounds : tuple
        Bayesian search bounds for alpha.
    n_calls : int
        Bayesian optimisation evaluations.
    n_initial : int
        Initial random Bayesian evaluations.
    initial_capital : float
        Starting capital.
    train_months : int
        Training window size. Default 2.
    compound : bool
        Carry final capital forward across windows. Default True.
    random_state : int
        Random seed.
    custom_fn : callable or None
        Custom metric function (mode='custom' only).

    Returns
    -------
    pd.DataFrame
        One row per test month: test_month, cumulative_return, max_drawdown,
        n_trades, win_ratio, sharpe, profit_factor, capital,
        optimal_theta_init, optimal_alpha.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")

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

        prices_train = prices[(prices.index >= train_start) & (prices.index < train_end)]
        prices_test = prices[(prices.index >= test_start) & (prices.index < test_end)]

        if len(prices_train) < 100 or len(prices_test) < 10:
            continue

        try:
            pipeline = ATDCPipeline(
                adaptation_mode=adaptation_mode,
                adaptation_rate=adaptation_rate,
                lookback_window=lookback_window,
                adaptation_step=adaptation_step,
                theta_min=theta_min,
                theta_max=theta_max,
                alpha=alpha,
                use_hmm=use_hmm,
                use_gp=use_gp,
                theta_bounds=theta_bounds,
                alpha_bounds=alpha_bounds,
                n_calls=n_calls,
                n_initial=n_initial,
                random_state=random_state,
                custom_fn=custom_fn,
            )
            pipeline.fit(prices_train)
            result = pipeline.run(prices_test, initial_capital=capital)
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
            'optimal_theta_init': result['optimal_theta_init'],
            'optimal_alpha': result['optimal_alpha'],
        })

        if compound:
            capital = result['final_capital']

    return pd.DataFrame(results)
