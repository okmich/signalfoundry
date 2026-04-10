"""
Structural-break / stationarity testing for time-series data.

Source / Attribution
--------------------
The break-in-mean detection algorithm is a Python port of ``BREAK_MEAN.CPP`` by Timothy Masters, published in:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

    C++ source file: BREAK_MEAN.CPP (companion code distributed with the book).

The statistical test used is the **Mann-Whitney U-test** (also known as the Wilcoxon rank-sum test),
a non-parametric test that does not assume normality and is robust to heavy tails — appropriate for financial return series.
Masters' implementation slides a window of varying size across the series and reports the maximum U-statistic,
then uses permutation testing (shuffling the full series) to assess significance.

The ``StationarityTester`` class and ``BreakTestResult`` dataclass are original contributions of this project.
The Mann-Whitney U statistic and full MCPT permutation loop are implemented as Numba-JIT kernels in ``_numba_kernels.py``; the outer search and permutation logic replicate Masters' C++ code.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ._numba_kernels import _max_u_stat_kernel, stationarity_mcpt_kernel


@dataclass
class BreakTestResult:
    """
    Result of a single structural-break test.

    Attributes
    ----------
    test_statistic : float
        Maximum standardised U-statistic (z-score) found across all window
        sizes and positions searched.
    p_value : float
        Permutation p-value: fraction of random shuffles that produced an
        absolute z-score >= the observed value.
    break_index : int
        Index in the original array where the detected break begins (start
        of the "recent" window).
    recent_window_size : int
        Size of the recent window at the detected break point.
    is_significant : bool
        Convenience flag: ``True`` when ``p_value < 0.05``.
    """

    test_statistic: float
    p_value: float
    break_index: int
    recent_window_size: int
    is_significant: bool

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.is_significant else "not significant"
        return (
            f"\nBreak-in-Mean Test Results (Masters' BREAK_MEAN algorithm)\n"
            f"===========================================================\n"
            f"Test statistic (max |z|) : {self.test_statistic:.4f}\n"
            f"Permutation p-value      : {self.p_value:.4f}\n"
            f"Result                   : {verdict} at alpha=0.05\n"
            f"Break location (index)   : {self.break_index}\n"
            f"Recent window size       : {self.recent_window_size} bars\n"
        )


class StationarityTester:
    """
    Detect structural breaks (mean shifts) in financial time series.

    Python port of Timothy Masters' ``BREAK_MEAN.CPP``.

    The algorithm tests whether the most recent *k* observations come from a
    different distribution than the preceding observations.  It searches over
    all window sizes in ``[min_recent, max_recent]`` and reports the window
    that produces the strongest Mann-Whitney U-statistic.  Statistical
    significance is assessed via permutation testing.

    Parameters
    ----------
    min_recent : int, default 20
        Minimum number of recent bars in the comparison window.
    max_recent : int, default 100
        Maximum number of recent bars in the comparison window.
    nperms : int, default 1000
        Number of permutations for the significance test.
    random_seed : int, default 42
        RNG seed for reproducibility.

    Examples
    --------
    >>> tester = StationarityTester(min_recent=20, max_recent=100, nperms=1000)
    >>> result = tester.test(returns)
    >>> if result.is_significant:
    ...     print(f"Regime change detected at index {result.break_index}")
    """

    def __init__(
        self,
        min_recent: int = 20,
        max_recent: int = 100,
        nperms: int = 1000,
        random_seed: int = 42,
    ) -> None:
        self.min_recent = min_recent
        self.max_recent = max_recent
        self.nperms = nperms
        self.random_seed = random_seed
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def test(self, data: np.ndarray, comparisons: int = 1) -> BreakTestResult:
        """
        Test for a break in the mean of *data*.

        Parameters
        ----------
        data : array-like
            1-D time series (e.g. log returns).
        comparisons : int, default 1
            Number of sliding comparison positions tested per window size.
            Increasing this beyond 1 tests whether a break exists *anywhere*
            in the most-recent portion of the series, at the cost of
            increased multiple-comparison risk (Masters' ``comparisons``
            parameter).

        Returns
        -------
        BreakTestResult
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        max_recent = min(self.max_recent, n // 2)
        min_recent = min(self.min_recent, max_recent)

        if min_recent < 2:
            raise ValueError("Series too short for break-in-mean test.")

        # Entire observed + permutation loop runs in compiled Numba code
        obs_stat, break_idx, recent_size, p_value = stationarity_mcpt_kernel(
            data, min_recent, max_recent, comparisons, self.nperms, self.random_seed
        )

        return BreakTestResult(
            test_statistic=float(obs_stat),
            p_value=float(p_value),
            break_index=int(break_idx),
            recent_window_size=int(recent_size),
            is_significant=(p_value < 0.05),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _max_u_stat(self, data: np.ndarray, min_recent: int, max_recent: int, comparisons: int, record_location: bool) -> Tuple[float, int, int]:
        """
        Compute the maximum absolute standardised U-statistic.

        Delegates to the Numba-JIT ``_max_u_stat_kernel`` for speed.
        """
        return _max_u_stat_kernel(data, min_recent, max_recent, comparisons, record_location)

    # ------------------------------------------------------------------ #
    # Convenience wrappers                                                 #
    # ------------------------------------------------------------------ #

    def is_stationary(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """
        Return ``True`` if no significant break is detected at *alpha*.

        Parameters
        ----------
        data : array-like
            1-D time series.
        alpha : float, default 0.05
            Significance level.
        """
        result = self.test(data)
        return result.p_value >= alpha
