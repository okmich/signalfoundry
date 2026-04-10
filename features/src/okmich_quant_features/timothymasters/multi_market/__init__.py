"""
Multi-market indicators.

Python ports of the multi-market indicators from:
    Timothy Masters, "Statistically Sound Indicators For Financial
    Market Prediction", Apress, 2013.  C++ source: Multi/COMP_VAR.CPP, JANUS.CPP.

All inputs must be **date-aligned** — only pass bars where all markets have
data on the same date.  Alignment is the caller's responsibility.

Module layout
-------------
portfolio_stats   #1–10  Portfolio statistical indicators (rank/median/range/iqr/clump)
risk              #11–15 Multivariate risk indicators (Mahal, ABS ratio/shift, coherence)
janus             #16–40 JANUS relative-strength system (RS, RM, DOM/DOE, CMA)
"""

from .portfolio_stats import trend_rank, trend_median, trend_range, trend_iqr, trend_clump, cmma_rank, cmma_median, \
    cmma_range, cmma_iqr, cmma_clump
from .risk import mahal, abs_ratio, abs_shift, coherence, delta_coherence
from .janus import Janus, janus_market_index, janus_rs, janus_rs_fractile, janus_delta_rs_fractile, janus_rss, \
    janus_delta_rss, janus_dom, janus_doe, janus_dom_index, janus_rm, janus_rm_fractile, janus_delta_rm_fractile, \
    janus_rs_leader_equity, janus_rs_laggard_equity, janus_rs_ps, janus_rs_leader_advantage, \
    janus_rs_laggard_advantage, janus_rm_leader_equity, janus_rm_laggard_equity, janus_rm_ps, \
    janus_rm_leader_advantage, janus_rm_laggard_advantage, janus_oos_avg, janus_cma_oos, janus_leader_cma_oos


__all__ = [
    # portfolio stats — trend variants
    "trend_rank",
    "trend_median",
    "trend_range",
    "trend_iqr",
    "trend_clump",
    # portfolio stats — cmma variants
    "cmma_rank",
    "cmma_median",
    "cmma_range",
    "cmma_iqr",
    "cmma_clump",
    # risk
    "mahal",
    "abs_ratio",
    "abs_shift",
    "coherence",
    "delta_coherence",
    # janus
    "Janus",
    "janus_market_index",
    "janus_rs",
    "janus_rs_fractile",
    "janus_delta_rs_fractile",
    "janus_rss",
    "janus_delta_rss",
    "janus_dom",
    "janus_doe",
    "janus_dom_index",
    "janus_rm",
    "janus_rm_fractile",
    "janus_delta_rm_fractile",
    "janus_rs_leader_equity",
    "janus_rs_laggard_equity",
    "janus_rs_ps",
    "janus_rs_leader_advantage",
    "janus_rs_laggard_advantage",
    "janus_rm_leader_equity",
    "janus_rm_laggard_equity",
    "janus_rm_ps",
    "janus_rm_leader_advantage",
    "janus_rm_laggard_advantage",
    "janus_oos_avg",
    "janus_cma_oos",
    "janus_leader_cma_oos",
]
