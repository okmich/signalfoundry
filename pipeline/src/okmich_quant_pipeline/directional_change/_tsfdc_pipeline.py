"""
TSFDC pipeline — high-level orchestrator and rolling walk-forward backtest.

TSFDCPipeline         — fit BBTheta classifier on training data, run on test data.
run_tsfdc_sliding_window — 24-month train / 1-month test rolling walk-forward.

Reference: Bakhach, Tsang & Chinthalapati (ISAFM, 2018) Sections 5.4, 13.
"""
import warnings

import pandas as pd

from okmich_quant_features.directional_change import (
    extract_tsfdc_features,
    label_bbtheta,
    parse_dual_dc,
)

from ._tsfdc_classifier import train_bbtheta_classifier
from ._tsfdc_engine import run_tsfdc_algorithm


class TSFDCPipeline:
    """
    Full TSFDC pipeline: dual DC parsing → BBTheta classifier → trading engine.

    Implements Bakhach, Tsang & Chinthalapati (2018):
      1. Train:  run dual DC on training window, label BBTheta, extract features,
                 fit LightGBM classifier.
      2. Run:    apply classifier to test window, execute Rules 1-3 for both
                 TSFDC-down and TSFDC-up simultaneously.

    Parameters
    ----------
    stheta : float
        Small DC threshold (e.g., 0.001 for 0.1%).
    btheta : float
        Big DC threshold.  Must be > stheta.
    random_seed : int
        Random seed passed to the classifier.
    """

    def __init__(self, stheta: float = 0.001, btheta: float = 0.0013, random_seed: int = 42):
        if btheta <= stheta:
            raise ValueError(f"btheta ({btheta}) must be greater than stheta ({stheta})")
        self.stheta = stheta
        self.btheta = btheta
        self.random_seed = random_seed
        self.classifier_ = None

    def fit(self, prices_train: pd.Series) -> 'TSFDCPipeline':
        """
        Train BBTheta classifier on a training price series.

        Parameters
        ----------
        prices_train : pd.Series
            Training price series (close prices).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the training window produces too few DC events or only one class.
        """
        trends_s, trends_b = parse_dual_dc(prices_train, self.stheta, self.btheta)

        if len(trends_s) < 30:
            raise ValueError(
                f"Only {len(trends_s)} STheta DC events in training window. "
                "Use a longer training window or smaller stheta."
            )

        labelled = label_bbtheta(trends_s, trends_b)
        features_df = extract_tsfdc_features(labelled, trends_b, self.stheta, self.btheta)
        features_df['bbtheta'] = labelled['bbtheta']

        self.classifier_ = train_bbtheta_classifier(features_df, random_seed=self.random_seed)
        return self

    def run(self, prices_test: pd.Series, initial_capital_down: float = 10_000.0, initial_capital_up: float = 10_000.0) -> dict:
        """
        Run TSFDC on a test price series using the fitted classifier.

        Parameters
        ----------
        prices_test : pd.Series
            Test price series (must follow the training window in time).
        initial_capital_down : float
            Starting capital for the TSFDC-down (long) strategy.
        initial_capital_up : float
            Starting capital for the TSFDC-up (short) strategy.

        Returns
        -------
        dict
            Same structure as run_tsfdc_algorithm():
            {'down': {...}, 'up': {...}}

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self.classifier_ is None:
            raise RuntimeError("TSFDCPipeline.fit() must be called before run().")
        return run_tsfdc_algorithm(
            prices_test, self.stheta, self.btheta, self.classifier_,
            initial_capital_down, initial_capital_up,
        )


def run_tsfdc_sliding_window(prices: pd.Series, stheta: float = 0.001, btheta: float = 0.0013, initial_capital: float = 10_000.0, train_months: int = 24, random_seed: int = 42) -> pd.DataFrame:
    """
    Execute TSFDC across all rolling windows (24-month train / 1-month test).

    Capital is compounded across windows: end capital of window N becomes
    the start capital of window N+1 (separate tracking for down and up).

    Parameters
    ----------
    prices : pd.Series
        Full price series with a DatetimeIndex.
    stheta : float
        Small DC threshold.
    btheta : float
        Big DC threshold (> stheta).
    initial_capital : float
        Starting capital for the first window.
    train_months : int
        Length of training window in months.  Paper default: 24.
    random_seed : int
        Random seed for the classifier.

    Returns
    -------
    pd.DataFrame
        One row per test month with columns:
        test_month, rr_down, rr_up, mdd_down, mdd_up,
        n_trades_down, n_trades_up, win_ratio_down, win_ratio_up,
        profit_factor_down, profit_factor_up, capital_down, capital_up.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex for rolling window execution.")

    months = prices.index.to_period('M').unique()
    if len(months) < train_months + 1:
        raise ValueError(
            f"Need at least {train_months + 1} months of data, got {len(months)}."
        )

    results = []
    capital_d = float(initial_capital)
    capital_u = float(initial_capital)

    for i in range(train_months, len(months)):
        train_start = months[i - train_months].start_time
        train_end = months[i].start_time
        test_start = months[i].start_time
        test_end = months[i].end_time + pd.Timedelta(days=1)

        train_mask = (prices.index >= train_start) & (prices.index < train_end)
        test_mask = (prices.index >= test_start) & (prices.index < test_end)

        prices_train = prices[train_mask]
        prices_test = prices[test_mask]

        if len(prices_train) < 500 or len(prices_test) < 20:
            continue

        try:
            pipeline = TSFDCPipeline(stheta=stheta, btheta=btheta, random_seed=random_seed)
            pipeline.fit(prices_train)
            result = pipeline.run(prices_test, initial_capital_down=capital_d, initial_capital_up=capital_u)
        except Exception as exc:
            warnings.warn(f"Window {months[i]} skipped due to error: {exc!r}", stacklevel=2)
            continue

        r_d = result['down']
        r_u = result['up']

        results.append({
            'test_month': str(months[i]),
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
            'capital_down': r_d['final_capital'],
            'capital_up': r_u['final_capital'],
        })

        capital_d = r_d['final_capital']
        capital_u = r_u['final_capital']

    return pd.DataFrame(results)
