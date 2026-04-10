"""
ROC (Receiver Operating Characteristics) analysis for trading indicators.

Source / Attribution
--------------------
The three core algorithms — ROC table generation (``print_ROC``), optimal threshold search (``opt_thresh``), and
Monte Carlo Permutation Testing (``opt_MCPT``) — are Python ports of C++ functions by Timothy Masters, published in:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.
    C++ source files: ROC.CPP (companion code distributed with the book).

Key implementation notes taken directly from Masters:
- Unbiased percentile location: ``k = int(pct * (n + 1)) - 1``
- Tie handling: boundaries are moved to the start of the tied block
- Incremental win/loss accumulation for O(n) threshold search
- Conservative p-values: permuted stat must be ``>=`` (not ``>``) the observed stat before incrementing the counter
- Fisher-Yates shuffle replicated via ``numpy.random.shuffle``

The dataclass ``ROCResults`` and the ``ROCAnalyzer`` class wrapper are original contributions of this project.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ._numba_kernels import _opt_thresh_core, mcpt_kernel as _mcpt_kernel_nb


# --------------------------------------------------------------------------- #
# Result dataclass                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class ROCResults:
    """
    Complete results from a single ROC analysis run.

    Attributes
    ----------
    pf_all : float
        Grand profit factor computed over the entire dataset (all signals, all returns).
        Values > 1 indicate that wins exceed losses in aggregate.
    n_cases : int
        Number of valid signal/return pairs used in the analysis.
    long_threshold : float
        Optimal signal boundary for the long strategy.  Trade long when ``signal <= long_threshold``
        (the long set consists of the lowest-signal observations — contrarian convention).
        Use ``flip_sign=True`` in :meth:`analyze` to invert this for trend-following indicators.
    long_pf : float
        Profit factor achieved by the long strategy at its optimal threshold.
    long_pval : float
        MCPT p-value for the long strategy. Probability that random chance could produce a profit factor >= ``long_pf``.
    long_n_trades : int
        Number of long trades triggered at ``long_threshold``.
    short_threshold : float
        Optimal signal boundary for the short strategy.  Trade short when ``signal > short_threshold``
        (the short set consists of the highest-signal observations — contrarian convention).
    short_pf : float
        Profit factor achieved by the short strategy at its optimal threshold.
    short_pval : float
        MCPT p-value for the short strategy.
    short_n_trades : int
        Number of short trades triggered at ``short_threshold``.
    best_pval : float
        Most conservative p-value: probability that choosing the *better* of long/short by chance alone could match or
        beat the observed result. Use this value for go/no-go decisions.
    roc_table : pd.DataFrame
        Profit-factor table at a range of percentile thresholds.
        Columns: threshold, frac_above, pf_long_above, pf_short_above, frac_below, pf_short_below, pf_long_below.
    """

    pf_all: float
    n_cases: int
    long_threshold: float
    long_pf: float
    long_pval: float
    long_n_trades: int
    short_threshold: float
    short_pf: float
    short_pval: float
    short_n_trades: int
    best_pval: float
    roc_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:
        sig_long = "***" if self.long_pval < 0.01 else ("**" if self.long_pval < 0.05 else ("*" if self.long_pval < 0.10 else ""))
        sig_short = "***" if self.short_pval < 0.01 else ("**" if self.short_pval < 0.05 else ("*" if self.short_pval < 0.10 else ""))
        sig_best = "***" if self.best_pval < 0.01 else ("**" if self.best_pval < 0.05 else ("*" if self.best_pval < 0.10 else ""))
        return (
            f"\nROC Analysis Results\n"
            f"====================\n"
            f"Total cases : {self.n_cases}\n"
            f"Grand PF    : {self.pf_all:.4f}\n\n"
            f"Long  strategy  (signal <= {self.long_threshold:+.4f}) :\n"
            f"  Profit factor : {self.long_pf:.4f}\n"
            f"  P-value       : {self.long_pval:.4f} {sig_long}\n"
            f"  N trades      : {self.long_n_trades}\n\n"
            f"Short strategy  (signal >  {self.short_threshold:+.4f}) :\n"
            f"  Profit factor : {self.short_pf:.4f}\n"
            f"  P-value       : {self.short_pval:.4f} {sig_short}\n"
            f"  N trades      : {self.short_n_trades}\n\n"
            f"Best-side p-value : {self.best_pval:.4f} {sig_best}\n"
            f"Significance codes: *** p<0.01  ** p<0.05  * p<0.10\n"
        )


# --------------------------------------------------------------------------- #
# ROCAnalyzer                                                                  #
# --------------------------------------------------------------------------- #

class ROCAnalyzer:
    """
    Receiver Operating Characteristics analyser for trading indicators.

    Evaluates the predictive power of a single indicator by:

    1. Building a *ROC table* — profit factors at a range of percentile
       thresholds (replicates Masters' ``print_ROC``).
    2. Finding *optimal long and short thresholds* via an O(n) incremental
       search (replicates Masters' ``opt_thresh``).
    3. Assessing *statistical significance* with Monte Carlo Permutation
       Testing: the returns vector is repeatedly shuffled to destroy any
       real signal–return relationship, and the fraction of permutations
       that achieve a profit factor >= the observed value is reported as a
       p-value (replicates Masters' ``opt_MCPT``).

    Source
    ------
    Timothy Masters, "Statistically Sound Indicators For Financial Market
    Prediction", Apress, 2013.  C++ source: ROC.CPP.

    Parameters
    ----------
    min_kept_pct : float, default 0.01
        Minimum fraction of data that must remain above (or below) a
        candidate threshold.  Prevents the optimiser from selecting
        thresholds that fire on only one or two trades.
    nreps : int, default 1000
        Number of permutation replications for MCPT p-value estimation.
        Masters recommends >= 1 000 for reliable 0.05-level inference.
    bins : np.ndarray, optional
        Percentile cutpoints used for the ROC table.  Defaults to Masters'
        standard set: [0.01, 0.05, 0.10, ..., 0.90, 0.95, 0.99].
    random_seed : int, default 42
        Seed for the permutation RNG (reproducibility).
    """

    DEFAULT_BINS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

    def __init__(self, min_kept_pct: float = 0.01, nreps: int = 1000, bins: Optional[np.ndarray] = None,
                 random_seed: int = 42) -> None:
        self.min_kept_pct = min_kept_pct
        self.nreps = nreps
        self.bins = bins if bins is not None else self.DEFAULT_BINS
        self.random_seed = random_seed
        self._logger = logging.getLogger(__name__)

    def analyze(self, signals: np.ndarray, returns: np.ndarray, flip_sign: bool = False) -> ROCResults:
        """
        Run a complete ROC analysis.

        Parameters
        ----------
        signals : array-like, shape (n,)
            Indicator values aligned with *returns*.
        returns : array-like, shape (n,)
            Forward log-returns aligned with *signals*.
            Use ``log(close[t+1] / close[t])`` (Masters' convention).
        flip_sign : bool, default False
            Negate *signals* before analysis.  Use when a *low* indicator
            value is the bullish signal (e.g. oversold RSI).

        Returns
        -------
        ROCResults
        """
        signals, returns = self._validate(signals, returns)
        n = len(signals)
        min_kept = max(1, int(n * self.min_kept_pct))
        if flip_sign:
            signals = -signals

        roc_table = self._roc_table(signals, returns)
        result = self._mcpt(signals, returns, min_kept)

        return ROCResults(
            pf_all=result["pf_all"],
            n_cases=n,
            long_threshold=result["long_threshold"],
            long_pf=result["long_pf"],
            long_pval=result["pval_long"],
            long_n_trades=result["n_long"],
            short_threshold=result["short_threshold"],
            short_pf=result["short_pf"],
            short_pval=result["pval_short"],
            short_n_trades=result["n_short"],
            best_pval=result["pval_best"],
            roc_table=roc_table,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _validate(self, signals: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        signals = np.asarray(signals, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)

        if signals.ndim != 1 or returns.ndim != 1:
            raise ValueError("signals and returns must be 1-D arrays.")
        if len(signals) != len(returns):
            raise ValueError("signals and returns must have the same length.")

        valid = np.isfinite(signals) & np.isfinite(returns)
        n_invalid = int((~valid).sum())
        if n_invalid:
            self._logger.warning("Dropping %d non-finite values.", n_invalid)
            signals = signals[valid]
            returns = returns[valid]

        if len(signals) < 30:
            raise ValueError(
                f"Need at least 30 valid cases; got {len(signals)}."
            )
        return signals, returns

    # ------------------------------------------------------------------ #
    # ROC table  (Masters' print_ROC)                                      #
    # ------------------------------------------------------------------ #

    def _roc_table(self, signals: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """
        Compute profit factors at each percentile threshold.

        Replicates Masters' ``print_ROC`` function.

        The unbiased percentile index formula ``k = int(pct * (n + 1)) - 1`` and the tie-resolution logic
        (walk left until the start of the tied block) are taken verbatim from Masters' C++ source.
        """
        n = len(signals)
        sort_idx = np.argsort(-signals)            # descending
        sorted_signals = signals[sort_idx]
        sorted_returns = returns[sort_idx]

        rows = []
        for pct in self.bins:
            # Unbiased percentile location (Masters' formula)
            k = int(pct * (n + 1)) - 1
            k = int(np.clip(k, 0, n - 1))

            # Walk left to the start of the tied block
            while k > 0 and sorted_signals[k - 1] == sorted_signals[k]:
                k -= 1

            if k == 0 or k == n - 1:
                continue

            threshold = sorted_signals[k]
            above = sorted_returns[:k]
            below = sorted_returns[k:]

            pf_long_above, pf_short_above = self._pf_pair(above)
            pf_long_below, pf_short_below = self._pf_pair(below)

            rows.append({
                "threshold": threshold,
                "frac_above": k / n,
                "pf_long_above": pf_long_above,
                "pf_short_above": pf_short_above,
                "frac_below": (n - k) / n,
                "pf_short_below": pf_short_below,
                "pf_long_below": pf_long_below,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _pf_pair(returns: np.ndarray, eps: float = 1e-30) -> Tuple[float, float]:
        """Return (pf_long, pf_short) for a slice of returns."""
        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        return wins / (losses + eps), losses / (wins + eps)

    # ------------------------------------------------------------------ #
    # Optimal threshold search  (Masters' opt_thresh)                      #
    # ------------------------------------------------------------------ #

    def _opt_thresh(self, signals: np.ndarray, returns: np.ndarray, min_kept: int) -> Dict:
        """
        Find optimal long and short thresholds via incremental O(n) search.

        Replicates Masters' ``opt_thresh`` function.  The inner scan loop is
        executed by the Numba-JIT kernel ``_opt_thresh_core`` for speed.
        """
        sort_idx = np.argsort(-signals)
        sorted_signals = signals[sort_idx]
        sorted_returns = returns[sort_idx]

        pf_all, long_pf, long_idx, short_pf, short_idx = _opt_thresh_core(
            sorted_signals, sorted_returns, min_kept, 1e-30
        )
        return {
            "pf_all": pf_all,
            "long_threshold": sorted_signals[long_idx],
            "long_pf": long_pf,
            "n_long": len(signals) - long_idx,
            "short_threshold": sorted_signals[short_idx],
            "short_pf": short_pf,
            "n_short": short_idx,
        }

    # ------------------------------------------------------------------ #
    # Monte Carlo Permutation Testing  (Masters' opt_MCPT)                 #
    # ------------------------------------------------------------------ #

    def _mcpt(self, signals: np.ndarray, returns: np.ndarray, min_kept: int) -> Dict:
        """
        Estimate p-values via Monte Carlo Permutation Testing.

        Replicates Masters' ``opt_MCPT`` function.

        The entire permutation loop — including Fisher-Yates shuffle and the
        O(n) threshold scan — runs inside the Numba-JIT ``mcpt_kernel``:
        signals are sorted once here; every permutation reuses that sort order
        (shuffling the already-sorted returns is equivalent to shuffling the
        original returns and re-applying the sort key).

        Conservative ``>=`` comparison and counters-start-at-1 follow Masters.
        """
        sort_idx = np.argsort(-signals)
        sorted_signals = signals[sort_idx]
        sorted_returns = returns[sort_idx]

        (
            pf_all, long_pf, long_idx, n_long,
            short_pf, short_idx, n_short,
            pval_long, pval_short, pval_best,
        ) = _mcpt_kernel_nb(
            sorted_signals, sorted_returns, min_kept, self.nreps, self.random_seed
        )

        return {
            "pf_all": pf_all,
            "long_threshold": sorted_signals[long_idx],
            "long_pf": long_pf,
            "n_long": int(n_long),
            "short_threshold": sorted_signals[short_idx],
            "short_pf": short_pf,
            "n_short": int(n_short),
            "pval_long": pval_long,
            "pval_short": pval_short,
            "pval_best": pval_best,
        }
