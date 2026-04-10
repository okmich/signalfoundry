"""
Cross-market (paired) indicators.

Python/Numba ports of the paired-market indicators from:
    Timothy Masters, "Statistically Sound Indicators For Financial
    Market Prediction", Apress, 2013.  C++ source: Paired/COMP_VAR.CPP.

All inputs must be **date-aligned** — only pass bars where both markets have data on the same date.
Alignment is the caller's responsibility.

Module layout
-------------
correlation   #1–2  Spearman correlation and its delta
deviation     #3    Log-space OLS spread deviation
purify        #4–5  SVD-based spread purification (PURIFY, LOG PURIFY)
trend_diff    #6–7  Linear-trend difference, CMMA difference
"""

from .correlation import correlation, delta_correlation
from .deviation import deviation
from .purify import purify, log_purify
from .trend_diff import trend_diff, cmma_diff

__all__ = [
    "correlation",
    "delta_correlation",
    "deviation",
    "purify",
    "log_purify",
    "trend_diff",
    "cmma_diff",
]
