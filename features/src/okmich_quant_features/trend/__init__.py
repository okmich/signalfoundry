"""Trend feature package — lean, orthogonal set.

Directional regime labels
-------------------------
  continuous_trend_labeling      Price-action state machine; +1 / -1 / 0
  trend_persistence_labeling     Vol-normalised drift; +1 / -1 / 0

Continuous oscillators / features
---------------------------------
  cci                            Typical-price vs MAD-normalised SMA
  bollinger_band                 SMA +/- k*stdev (returns bands + %B + width)
  zscore_trend_features          Continuous rolling z-score of log returns + derivative + |z|

Bundle
------
  core_trend_features            Concatenates the orthogonal core into one DataFrame.

Parameter selection: callers pass parameters directly. Anchor to interpretable scales (vol, wall-clock horizon)
rather than fitting — the package no longer ships any parameter optimizers.
"""

import pandas as pd
import numpy as np

from .continous_trend import continuous_trend_labeling
from .misc import bollinger_band, cci
from .trend_persistence import trend_persistence_labeling
from .z_score_trend import zscore_trend_features


def core_trend_features(df: pd.DataFrame,
    bb_window: int = 24, bb_deviation_up: float = 2.0, bb_deviation_down: float = 2.0,
    cci_window: int = 24,
    continuous_omega: float = 0.15,
    persistence_window: int = 20, persistence_smooth: int = 5, persistence_zscore_norm: bool = True,
    zscore_window: int = 30, zscore_deriv_window: int = 5,
    high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]

    bb_upper, bb_middle, bb_lower, bb_percent_b, bb_width = bollinger_band(close_price, window=bb_window,
                                                                           deviation_up=bb_deviation_up,
                                                                           deviation_down=bb_deviation_down)

    safe_close = close_price.replace(0, np.nan)
    result[f"bb_upper_{bb_window}"] = bb_upper / safe_close
    result[f"bb_middle_{bb_window}"] = bb_middle / safe_close
    result[f"bb_lower_{bb_window}"] = bb_lower / safe_close
    result[f"bb_percent_b_{bb_window}"] = bb_percent_b
    result[f"bb_width_{bb_window}"] = bb_width

    result[f"cci_{cci_window}"] = cci(high_price, low_price, close_price, window=cci_window)

    result[f"continuous_trend_{continuous_omega}".replace(".", "_")] = \
        continuous_trend_labeling(close_price, omega=continuous_omega)

    result[f"trend_persistence_{persistence_window}_{persistence_smooth}"] = (
        trend_persistence_labeling(close_price, window=persistence_window, smooth=persistence_smooth,
                                   zscore_norm=persistence_zscore_norm))

    zscore_df = zscore_trend_features(close_price, window=zscore_window, deriv_window=zscore_deriv_window)
    for col in zscore_df.columns:
        result[col] = zscore_df[col]

    return result


__all__ = [
    "core_trend_features",
    # Directional labels
    "continuous_trend_labeling",
    "trend_persistence_labeling",
    # Continuous oscillators / features
    "bollinger_band",
    "cci",
    "zscore_trend_features",
]