"""
Timothy Masters indicators for financial market prediction.

Source / Attribution
--------------------
All indicators are Python/Numba ports of the C++ implementations published by:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.
    C++ source files: COMP_VAR.CPP, STATS.CPP, LEGENDRE.CPP, INFORM.CPP, FTI.CPP, SPEARMAN.CPP.

Sub-package layout
------------------
single        38 single-market indicators (#1–38)
cross_market   7 paired-market indicators (#1–7: correlation, deviation, purify, trend/cmma diff)
multi_market  40 multi-market indicators (#1–15: portfolio stats, risk; #16–40: JANUS)
utils         Batch computation layer (single_features_computer, cross_features_computer, multi_features_computer)

Live-trading boundary
---------------------
Stateless, side-effect-free — safe to call on a rolling window of OHLCV bars.
"""
