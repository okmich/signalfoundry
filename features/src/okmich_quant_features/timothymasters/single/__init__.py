"""
Timothy Masters single-market indicators (#1–38).

Source / Attribution
--------------------
Python/Numba ports of the C++ implementations published by:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.
    C++ source files: COMP_VAR.CPP, STATS.CPP, LEGENDRE.CPP, INFORM.CPP, FTI.CPP.

Module layout
-------------
_helpers     ATR, variance, normal_cdf_compress, f_cdf_compress
_legendre    Discrete orthonormal Legendre polynomial weights
momentum     Indicators #1–11  (RSI … Reactivity)
trend        Indicators #12–21 (Linear/Quadratic/Cubic Trend & Deviation, ADX, Aroon)
variance     Indicators #22–23 (Price / Change Variance Ratio)
volume       Indicators #24–32 (Intraday Intensity … Volume Momentum)
information  Indicators #33–34 (Entropy, Mutual Information)
fti          Indicators #35–38 (FTI Lowpass, Best Width, Best Period, Best FTI)
"""

from .momentum import (
    rsi, detrended_rsi, stochastic, stoch_rsi, ma_difference, macd, ppo,
    price_change_osc, close_minus_ma, price_intensity, reactivity,
)
from .trend import (
    linear_trend, quadratic_trend, cubic_trend,
    linear_deviation, quadratic_deviation, cubic_deviation,
    adx, aroon_up, aroon_down, aroon_diff,
)
from .variance import (
    price_variance_ratio,
    change_variance_ratio,
)
from .volume import (
    intraday_intensity, money_flow, price_volume_fit, vwma_ratio,
    normalized_obv, delta_obv, normalized_pvi, normalized_nvi,
    volume_momentum,
)
from .information import (
    entropy, mutual_information,
)
from .fti import (
    fti_lowpass, fti_best_width, fti_best_period, fti_best_fti,
)
__all__ = [
    # momentum (#1–11)
    "rsi", "detrended_rsi", "stochastic", "stoch_rsi", "ma_difference",
    "macd", "ppo", "price_change_osc", "close_minus_ma", "price_intensity",
    "reactivity",
    # trend (#12–21)
    "linear_trend", "quadratic_trend", "cubic_trend",
    "linear_deviation", "quadratic_deviation", "cubic_deviation",
    "adx", "aroon_up", "aroon_down", "aroon_diff",
    # variance (#22–23)
    "price_variance_ratio", "change_variance_ratio",
    # volume (#24–32)
    "intraday_intensity", "money_flow", "price_volume_fit", "vwma_ratio",
    "normalized_obv", "delta_obv", "normalized_pvi", "normalized_nvi",
    "volume_momentum",
    # information (#33–34)
    "entropy", "mutual_information",
    # fti (#35–38)
    "fti_lowpass", "fti_best_width", "fti_best_period", "fti_best_fti",
]
