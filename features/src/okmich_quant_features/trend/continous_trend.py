"""Continuous Trend Labeling (price-action state machine) and the band-gated 3-class derivation.

Reference: https://www.mdpi.com/1099-4300/22/10/1162

continuous_trend_labeling tracks price extremes and reversals sequentially using a single omega threshold,
emitting a binary {-1, +1} regime label (0 only during pre-trigger warmup). Omega is caller-supplied; pick it
from a vol anchor (e.g. ~ k * median ATR / price) rather than fitting.

The band-gated 3-class section below converts the binary CTL output to a {-1, 0, +1} ternary by gating with an
ATR envelope: bars inside the band are forced to 0 (low-confidence / noise zone). See attach_labels and
apply_3class_labels.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .misc import envelope


def continuous_trend_labeling(prices: Union[pd.Series | np.ndarray], omega=0.15):
    """
    Implements the Continuous Trend Labeling (CTL) method without look-ahead bias.
    Processes data sequentially, only using past information for utils.

    Parameters:
    -----------
    prices : pandas.Series or numpy.ndarray
        Input time series data (prices)
    omega : float, optional (default=0.15)
        Threshold parameter for trend detection (15% as in the paper)

    Returns:
    --------
    numpy.ndarray
        Array of trend: 1 for upward trends, -1 for downward trends, 0 for neutral
    """
    # Convert pandas Series to numpy array
    if isinstance(prices, pd.Series):
        x = prices.values
    else:
        x = np.asarray(prices)

    n = len(x)
    labels = np.zeros(n)  # Initialize trend as neutral (0)

    # Handle empty input
    if n == 0:
        return labels

    # Initialize variables (Algorithm 1 from paper)
    FP = x[0]  # First price
    xH = x[0]  # Current highest price
    HT = 0  # Time of highest price
    xL = x[0]  # Current lowest price
    LT = 0  # Time of lowest price
    Cid = 0  # Current direction (0=neutral, 1=up, -1=down)
    FP_N = 0  # Index of first significant move

    # First pass: Find initial trend direction (sequential)
    for i in range(n):
        if x[i] > FP + x[0] * omega:  # Upward threshold
            xH = x[i]
            HT = i
            FP_N = i
            Cid = 1
            break
        elif x[i] < FP - x[0] * omega:  # Downward threshold
            xL = x[i]
            LT = i
            FP_N = i
            Cid = -1
            break

    # If no significant trend found, return neutral trend
    if Cid == 0:
        return labels

    # Initialize the first segment (neutral until first significant move)
    labels[:FP_N] = 0

    # Second pass: Track trends and reversals (sequential, no look-ahead)
    for i in range(FP_N, n):
        if Cid == 1:  # Current upward trend
            if x[i] > xH:  # Update highest price
                xH = x[i]
                HT = i

            # Label current point as upward trend
            labels[i] = 1

            # Check for downward reversal (using only past information)
            if x[i] < xH - xH * omega and LT <= HT:
                # Switch to downward trend
                xL = x[i]
                LT = i
                Cid = -1
                # Label the reversal point as downward trend
                labels[i] = -1

        elif Cid == -1:  # Current downward trend
            if x[i] < xL:  # Update lowest price
                xL = x[i]
                LT = i

            # Label current point as downward trend
            labels[i] = -1

            # Check for upward reversal (using only past information)
            if x[i] > xL + xL * omega and HT <= LT:
                # Switch to upward trend
                xH = x[i]
                HT = i
                Cid = 1
                # Label the reversal point as upward trend
                labels[i] = 1

    return labels


##############################################################################################################
########################## THREE-CLASS LABEL DERIVATION (BAND-GATED CTL) #####################################
##############################################################################################################
#
# Turns the binary CTL output into a {-1, 0, +1} ternary label by gating CTL labels with an ATR-based envelope band
# (MA +/- k*ATR). Class-0 marks bars where price sits inside the band — the noise / low-confidence zone —
# while +/-1 marks confident directional regimes.
#
# Calibration of (omega, ma_period, atr_period, k_atr) happens upstream; this module
# provides:
#   - compute_band_state — turn an (already-enveloped) close + (upper, lower)
#     into a {-1, 0, +1} ternary state (envelope itself is sourced from .misc),
#   - attach_labels for offline / batch use given an explicit (omega, BandParams),
#   - apply_3class_labels for runtime use that accepts a pre-resolved (omega, band)
#     and rescales bar-count parameters to the input timeframe. Caller is
#     responsible for sourcing the config (typically from a SymbolMetastore block).


@dataclass(frozen=True)
class BandParams:
    """ATR envelope band: MA(ma_period) +/- k_atr * ATR(atr_period)."""
    ma_period: int
    atr_period: int
    k_atr: float

    def __post_init__(self):
        if self.ma_period < 2:
            raise ValueError("ma_period must be >= 2")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if self.k_atr <= 0:
            raise ValueError("k_atr must be > 0")


def compute_band_state(close: pd.Series, upper: pd.Series, lower: pd.Series) -> np.ndarray:
    """Ternary band state: +1 above upper, -1 below lower, 0 inside (or warmup NaN)."""
    if not (len(close) == len(upper) == len(lower)):
        raise ValueError(
            f"compute_band_state: length mismatch — "
            f"close={len(close)}, upper={len(upper)}, lower={len(lower)}"
        )
    state = np.zeros(len(close), dtype=np.int8)
    valid = (upper.notna() & lower.notna()).to_numpy()
    state[(close > upper).to_numpy() & valid] = 1
    state[(close < lower).to_numpy() & valid] = -1
    return state


def emit_three_class(ctl_labels: np.ndarray, band_state: np.ndarray) -> np.ndarray:
    """Quasi-posterior 3-class label: ctl_label when band has signal, else 0."""
    if len(ctl_labels) != len(band_state):
        raise ValueError(
            f"emit_three_class: length mismatch — "
            f"ctl_labels={len(ctl_labels)}, band_state={len(band_state)}"
        )
    return np.where(band_state != 0, ctl_labels, 0).astype(np.int8)


def attach_labels(df: pd.DataFrame, omega: float, band: BandParams,
                  binary_col: str = "ctl_label",
                  ternary_col: str = "ctl_label_3class") -> pd.DataFrame:
    """Compute envelope + binary CTL + 3-class labels and attach to a copy of df.

    Storage convention: the binary CTL label is the source-of-truth staging label; the 3-class column is a
    derived quasi-posterior used by downstream models that want a 'confidence' interpretation.

    Returns df with: 'ma', 'upper', 'lower', 'band_state', binary_col, ternary_col.

    Note: any pre-existing columns named 'ma', 'upper', 'lower', 'band_state', `binary_col`, or `ternary_col` on the
    input df are overwritten in the returned copy without warning.
    """
    out = df.copy()

    out["upper"], out["ma"], out["lower"], _, _ = envelope(
        out["close"], out["high"], out["low"],
        ma_period=band.ma_period, atr_period=band.atr_period, k_atr=band.k_atr,
    )
    ctl = np.asarray(continuous_trend_labeling(out["close"], omega=omega), dtype=np.int8)
    bs = compute_band_state(out["close"], out["upper"], out["lower"])
    out["band_state"] = bs
    out[binary_col] = ctl
    out[ternary_col] = emit_three_class(ctl, bs)
    return out


def _infer_tf_minutes(index: pd.DatetimeIndex) -> Optional[int]:
    """Best-effort bar duration in minutes from a DatetimeIndex.

    Uses index.freq when set, else median bar spacing — the latter is robust to weekend / holiday gaps in market data.
    Returns None when there is not enough information (single bar, non-DatetimeIndex) or when the median spacing is below
    one minute (sub-minute data should pass `tf_minutes` explicitly rather than relying on inference).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    if index.freq is not None:
        try:
            return int(pd.Timedelta(index.freq).total_seconds() / 60)
        except (ValueError, TypeError):
            pass
    diffs = index.to_series().diff().dt.total_seconds().dropna() / 60
    if diffs.empty:
        return None
    median_min = float(diffs.median())
    if median_min < 1.0:
        return None
    return int(round(median_min))


def apply_3class_labels(df: pd.DataFrame, omega: float, band: BandParams,
                        persisted_tf_minutes: int = 15,
                        tf_minutes: Optional[int] = None,
                        binary_col: str = "ctl_label",
                        ternary_col: str = "ctl_label_3class") -> pd.DataFrame:
    """Attach binary CTL + 3-class labels using a pre-resolved (omega, band) config.

    The caller is responsible for sourcing `omega`, `band`, and `persisted_tf_minutes` — typically from the
    SymbolMetastore's `htf_ctl_3class_params` block, but this function has no opinion on the source. It only handles
    label computation and the cross-TF rescaling of band bar-counts.

    Scaling rule:
      `ma_period` and `atr_period` are bar counts; they are scaled by `(persisted_tf_minutes / df_tf_minutes)` so the
      wall-clock window stays constant. Example: persisted `ma_period=480` at 15min = 7200-min window;
      on 5m bars that becomes 1440 bars (still 7200 min).

      `omega` is a percentage threshold and does NOT scale. Note: applying the same omega at finer resolution will produce
      more flips than at the calibration resolution, because finer close-price paths see more intermediate excursions.
      If you need flip locations that match the calibration timeframe exactly, compute on calibration-TF bars and
      forward-fill to your trading TF instead.

    Args:
      df: DataFrame with 'high', 'low', 'close' columns and a DatetimeIndex.
      omega: CTL threshold (dimensionless percentage), as persisted upstream.
      band: BandParams expressed at the persisted (calibration) timeframe.
      persisted_tf_minutes: Bar duration of the calibration venue (e.g., 15
        for "15min"). Defaults to 15 since that's the convention used by the
        upstream optimizer.
      tf_minutes: The bar duration of `df` in minutes. If None, inferred from
        `df.index` via median spacing.
      binary_col / ternary_col: Output column names.

    Returns:
      Copy of df with 'ma', 'upper', 'lower', 'band_state',
      binary_col, ternary_col columns attached.

    Raises:
      ValueError: if `tf_minutes` cannot be inferred and was not passed,
        or if either persisted or input timeframe is non-positive.

    Example caller (with the metastore lookup done outside the function):
        block = metastore.get_property_value(server, 5, symbol, "htf_ctl_3class_params")
        band = BandParams(**block["band"])
        ts_min = int(pd.Timedelta(block["venue_freq"]).total_seconds() / 60)
        labelled = apply_3class_labels(df_5m, omega=block["omega"], band=band,
                                       persisted_tf_minutes=ts_min)
    """
    if persisted_tf_minutes <= 0:
        raise ValueError(f"persisted_tf_minutes must be positive, got {persisted_tf_minutes}")

    if tf_minutes is None:
        tf_minutes = _infer_tf_minutes(df.index)
        if tf_minutes is None:
            raise ValueError("Could not infer tf_minutes from df.index — "
                             "pass tf_minutes explicitly.")
    if tf_minutes <= 0:
        raise ValueError(f"tf_minutes must be positive, got {tf_minutes}")

    scale = persisted_tf_minutes / tf_minutes
    scaled_band = BandParams(
        ma_period=max(2, int(round(band.ma_period * scale))),
        atr_period=max(2, int(round(band.atr_period * scale))),
        k_atr=band.k_atr,
    )
    return attach_labels(df, omega=omega, band=scaled_band,
                         binary_col=binary_col, ternary_col=ternary_col)
