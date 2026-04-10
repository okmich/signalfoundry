"""
Batch ROC evaluation across multiple indicators.

Source / Attribution
--------------------
The evaluation methodology — running Masters' ROC analysis on each indicator and ranking by p-value — follows the workflow described in:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

The ``BatchROCEvaluator`` class is an original contribution of this project that wraps :class:`ROCAnalyzer` to support
multi-indicator research workflows.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .roc_analysis import ROCAnalyzer, ROCResults


class BatchROCEvaluator:
    """
    Evaluate multiple indicators against the same market/returns dataset.

    Wraps :class:`ROCAnalyzer` so that a single market dataset can be reused across many indicator functions without
    re-computing forward returns each time.

    Parameters
    ----------
    market_data : pd.DataFrame
        OHLCV data with a datetime index and at minimum a ``close`` column.
    min_kept_pct : float, default 0.01
        Passed through to :class:`ROCAnalyzer`.
    nreps : int, default 1000
        Permutation replications passed through to :class:`ROCAnalyzer`.
    random_seed : int, default 42
        RNG seed for reproducibility.

    Examples
    --------
    >>> evaluator = BatchROCEvaluator(market_df, nreps=1000)
    >>> summary = evaluator.evaluate_batch(indicator_configs)
    >>> print(summary[summary["significant"]])
    """

    def __init__(self, market_data: pd.DataFrame, min_kept_pct: float = 0.01, nreps: int = 1000, random_seed: int = 42) -> None:
        self.market = market_data
        self.analyzer = ROCAnalyzer(min_kept_pct=min_kept_pct, nreps=nreps, random_seed=random_seed)
        self._logger = logging.getLogger(__name__)

        # Pre-compute one-step-ahead log returns
        close = market_data["close"]
        self._returns: np.ndarray = np.log(close.shift(-1) / close).values

    # ------------------------------------------------------------------ #
    # Single-indicator evaluation                                          #
    # ------------------------------------------------------------------ #

    def evaluate_indicator(self, name: str, indicator_func: Callable, params: Optional[Dict] = None,
                           flip_sign: bool = False) -> ROCResults:
        """
        Compute an indicator and run ROC analysis against the stored returns.

        Parameters
        ----------
        name : str
            Human-readable label used in logging.
        indicator_func : Callable
            Function with signature ``func(market_df, **params) -> pd.Series`` that returns indicator values aligned with ``market_data``.
        params : dict, optional
            Keyword arguments forwarded to *indicator_func*.
        flip_sign : bool, default False
            Negate signals before analysis (use when a *low* value is bullish).

        Returns
        -------
        ROCResults
        """
        self._logger.info("Evaluating indicator: %s", name)
        params = params or {}
        signals: pd.Series = indicator_func(self.market, **params)

        valid = signals.notna() & np.isfinite(self._returns)
        signals_arr = signals[valid].values
        returns_arr = self._returns[valid]

        return self.analyzer.analyze(signals_arr, returns_arr, flip_sign=flip_sign)

    # ------------------------------------------------------------------ #
    # Batch evaluation                                                     #
    # ------------------------------------------------------------------ #

    def evaluate_batch(self, indicator_configs: List[Dict], output_dir: Optional[Path] = None,
                       significance_level: float = 0.05) -> pd.DataFrame:
        """
        Evaluate a list of indicators and return a ranked summary DataFrame.

        Each element of *indicator_configs* is a dict with keys:

        - ``name``       (str)      – indicator label
        - ``func``       (Callable) – indicator function
        - ``params``     (dict)     – optional keyword arguments
        - ``flip_sign``  (bool)     – optional, default ``False``

        Parameters
        ----------
        indicator_configs : list of dict
            Indicator specifications as described above.
        output_dir : Path, optional
            When provided, per-indicator text reports and ROC tables are
            saved here as ``{name}_roc.txt`` and ``{name}_roc_table.csv``.
        significance_level : float, default 0.05
            Threshold applied to ``best_pval`` for the ``significant`` flag
            in the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per indicator, sorted by ``best_pval`` ascending.
            Columns: indicator, n_cases, pf_all, long_pf, long_pval,
            short_pf, short_pval, best_pval, significant.
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for cfg in indicator_configs:
            name = cfg["name"]
            try:
                result = self.evaluate_indicator(
                    name=name,
                    indicator_func=cfg["func"],
                    params=cfg.get("params"),
                    flip_sign=cfg.get("flip_sign", False),
                )
                rows.append({
                    "indicator": name,
                    "n_cases": result.n_cases,
                    "pf_all": result.pf_all,
                    "long_pf": result.long_pf,
                    "long_pval": result.long_pval,
                    "short_pf": result.short_pf,
                    "short_pval": result.short_pval,
                    "best_pval": result.best_pval,
                    "significant": result.best_pval < significance_level,
                })

                if output_dir is not None:
                    (output_dir / f"{name}_roc.txt").write_text(str(result))
                    result.roc_table.to_csv(output_dir / f"{name}_roc_table.csv", index=False)
            except Exception:
                self._logger.exception("Error evaluating indicator '%s'.", name)
        if not rows:
            return pd.DataFrame()

        summary = pd.DataFrame(rows).sort_values("best_pval").reset_index(drop=True)
        return summary
