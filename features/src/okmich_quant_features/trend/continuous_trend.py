"""Continuous Trend Labeling (price-action state machine) and the band-gated 3-class derivation.

Reference: https://www.mdpi.com/1099-4300/22/10/1162

continuous_trend_labeling tracks price extremes and reversals sequentially using a single omega threshold,
emitting a binary {-1, +1} regime label (0 only during pre-trigger warmup). Omega is caller-supplied; pick it
from a vol anchor (e.g. ~ k * median ATR / price) rather than fitting.

The band-gated 3-class section below converts the binary CTL output to a {-1, 0, +1} ternary by gating with an
ATR envelope: bars inside the band are forced to 0 (low-confidence / noise zone). See attach_labels and
apply_3class_labels.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .channels import envelope


def continuous_trend_labeling(prices: Union[pd.Series, np.ndarray], omega: float = 0.15) -> Union[pd.Series, np.ndarray]:
    """Continuous Trend Labeling (CTL) — sequential, no look-ahead.

    Implements Algorithm 1 from the reference paper. Tracks running extremes; once price moves omega% from the
    initial price, declares a trend; flips when price retraces omega% from the running extreme.

    Returns float64 with values:
        +1.0  uptrend
        -1.0  downtrend
        NaN   pre-trigger warmup (and the whole series if no significant move ever occurs)

    Return type mirrors input: pd.Series -> pd.Series (index preserved); np.ndarray -> np.ndarray.
    """
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")

    is_series = isinstance(prices, pd.Series)
    x = prices.values if is_series else np.asarray(prices)
    n = len(x)

    labels = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return pd.Series(labels, index=prices.index) if is_series else labels

    # Algorithm 1 from paper; lowercase per PEP 8, paper symbols in comments.
    first_price = x[0]            # paper: FP
    x_high = x[0]                 # paper: xH — running max
    t_high = 0                    # paper: HT — index of running max
    x_low = x[0]                  # paper: xL — running min
    t_low = 0                     # paper: LT — index of running min
    direction = 0                 # paper: Cid (0 = pre-trigger, 1 = up, -1 = down)
    trigger_idx = 0               # paper: FP_N — index of first significant move

    # First pass: find initial trend direction (sequential)
    for i in range(n):
        if x[i] > first_price + first_price * omega:
            x_high = x[i]; t_high = i; trigger_idx = i; direction = 1
            break
        elif x[i] < first_price - first_price * omega:
            x_low = x[i]; t_low = i; trigger_idx = i; direction = -1
            break

    # No significant move ever -> whole series stays NaN (unknown regime).
    if direction == 0:
        return pd.Series(labels, index=prices.index) if is_series else labels

    # Pre-trigger region stays NaN; signal starts at trigger_idx.
    for i in range(trigger_idx, n):
        if direction == 1:
            if x[i] > x_high:
                x_high = x[i]; t_high = i
            labels[i] = 1.0
            # t_low <= t_high guard: ensures we haven't already flipped on this leg.
            # The <= is safe because x_high was just updated above on a same-bar new high,
            # making t_low < t_high strictly; the equality case only arises at the very
            # first trigger when both timestamps are 0 and x_high was set to x[trigger_idx].
            if x[i] < x_high - x_high * omega and t_low <= t_high:
                x_low = x[i]; t_low = i; direction = -1
                labels[i] = -1.0
        else:  # direction == -1
            if x[i] < x_low:
                x_low = x[i]; t_low = i
            labels[i] = -1.0
            if x[i] > x_low + x_low * omega and t_high <= t_low:
                x_high = x[i]; t_high = i; direction = 1
                labels[i] = 1.0

    return pd.Series(labels, index=prices.index) if is_series else labels


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
    """Quasi-posterior 3-class label: ctl_label when band has signal, else 0.

    The band acts as a **confidence gate, not a direction check**. The function takes the CTL label as the source of
    truth for direction and only zeros it out when price sits inside the envelope. This means a CTL label can survive
    a disagreement with band direction:

        ctl_label = +1, band_state = +1  ->  +1   (agree, above upper band)
        ctl_label = +1, band_state = -1  ->  +1   (CTL still says up; price just broke lower band but CTL hasn't flipped yet)
        ctl_label = -1, band_state =  0  ->   0   (price inside band, gated to neutral)
        ctl_label = NaN, band_state = +1 ->   0   (warmup CTL is treated as no-signal)

    If you want sign-agreement semantics (zero out disagreements), do it at the caller.

    Accepts ctl_labels as float64 (with NaN warmup, as produced by continuous_trend_labeling) or int8. Output is int8;
    NaN bars map to 0.
    """
    if len(ctl_labels) != len(band_state):
        raise ValueError(
            f"emit_three_class: length mismatch — "
            f"ctl_labels={len(ctl_labels)}, band_state={len(band_state)}"
        )
    ctl_arr = np.asarray(ctl_labels, dtype=np.float64)
    ctl_safe = np.where(np.isnan(ctl_arr), 0, ctl_arr)
    return np.where(band_state != 0, ctl_safe, 0).astype(np.int8)


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
    ctl_raw = np.asarray(continuous_trend_labeling(out["close"], omega=omega), dtype=np.float64)
    bs = compute_band_state(out["close"], out["upper"], out["lower"])
    out["band_state"] = bs
    # Persist binary CTL as int8 (warmup NaN -> 0) for stable downstream dtype.
    out[binary_col] = np.where(np.isnan(ctl_raw), 0, ctl_raw).astype(np.int8)
    out[ternary_col] = emit_three_class(ctl_raw, bs)
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
      `ma_period` and `atr_period` are bar counts; they are scaled by `(persisted_tf_minutes / df_tf_minutes)` with
      `math.ceil()` so the wall-clock window is never shorter than the calibration window. Example:
      persisted `ma_period=480` at 15min = 7200-min window; on 5m bars that becomes 1440 bars (still 7200 min).
      Edge case: `ma_period=3` at 15min scaled to 10m bars -> 3 * 1.5 = 4.5 -> ceil to 5 (rather than rounding to 4).

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
    # Use ceil rather than round so the scaled wall-clock window is never shorter than the
    # calibrated window. Banker's rounding (round-half-to-even) can shorten the lookback
    # in edge cases (e.g., 4.5 -> 4), which destabilises the band in fine-TF inference.
    scaled_band = BandParams(
        ma_period=max(2, math.ceil(band.ma_period * scale)),
        atr_period=max(2, math.ceil(band.atr_period * scale)),
        k_atr=band.k_atr,
    )
    return attach_labels(df, omega=omega, band=scaled_band,
                         binary_col=binary_col, ternary_col=ternary_col)
