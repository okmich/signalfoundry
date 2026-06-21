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


_NONFINITE_PRICE_MSG = "prices contains NaN or infinite values; clean the data before labeling."


def _as_finite_price_array(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Coerce prices to an ndarray (Series -> .values) and reject non-finite values (NaN/inf).

    Shared by the batch and streaming array entry points so they enforce one input contract: CTL is a
    percentage-of-price state machine, and a NaN/inf bar would otherwise fall through every comparison yet
    still be assigned a directional label (false supervision). Mirrors directional_change.idc_parse.
    """
    x = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)
    if len(x) and not np.isfinite(x).all():
        raise ValueError(_NONFINITE_PRICE_MSG)
    return x


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
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")

    is_series = isinstance(prices, pd.Series)
    x = _as_finite_price_array(prices)  # rejects NaN/inf so missing bars can't get a false +/-1 label
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
############################### STREAMING (O(1) PER-BAR) CTL STATE MACHINE ###################################
##############################################################################################################
#
# Live/online projection of continuous_trend_labeling. The batch function above must replay the whole price history to
# know the CURRENT label with certainty — expensive to recompute every bar. CTLState persists the state machine's running
# extremes so, after a one-time warm-up over a long-enough history, each new bar costs O(1): no history reload, no recompute.
#
# Field names mirror the locals of continuous_trend_labeling so the two can be audited side by side.
# Bar-for-bar equivalence with the batch function is asserted by tests/trend/test_ctl_streaming_equivalence.py
# over CTL's domain: strictly positive prices (omega is a % of price). De-meaned/normalised series that can go
# negative are out of domain for both implementations.
#
# Warm-up / certainty-of-flip: the FSM is only O(1) AFTER burn-in. To match a from-scratch batch run bar-for-bar you must
# warm it up from far enough back that direction != 0 and the running extreme sits on the current leg (the same reason a
# live runner must seed CTL with enough bars before its first confirmed label). Pay that warm-up once at startup via
# ctl_warm_up / ctl_streaming_replay, then advance with ctl_step.


@dataclass
class CTLState:
    """O(1) streaming state for continuous_trend_labeling — a faithful per-bar projection of the batch machine.

    Warm up once (ctl_warm_up / ctl_streaming_replay), then advance one bar at a time with ctl_step. Field names
    mirror the batch function's locals; paper symbols are noted in comments.
    """
    omega: float
    direction: int = 0           # paper: Cid — 0 pre-trigger, +1 up, -1 down
    first_price: float = 0.0     # paper: FP — anchor for the initial trigger
    x_high: float = 0.0          # running max (paper: xH)
    t_high: int = 0              # index of running max (paper: HT)
    x_low: float = 0.0           # running min (paper: xL)
    t_low: int = 0               # index of running min (paper: LT)
    initialized: bool = False

    def __post_init__(self):
        if not math.isfinite(self.omega) or self.omega <= 0:
            raise ValueError(f"omega must be > 0, got {self.omega}")


def ctl_step(state: CTLState, price: float, i: int) -> int:
    """Advance the streaming CTL state by one bar; return the label for bar i. Mutates `state` in place.

    Returns 0/-1/+1. Pre-trigger warmup returns 0 (the batch function emits NaN there — callers that need the NaN
    contract should map 0 -> NaN on the warmup region). Reproduces both the initial-trigger pass (anchored on
    first_price = the first price seen) and the second-pass pivot dynamics of continuous_trend_labeling.

    Non-finite price (NaN/inf): the state is NOT updated and the current label is held (0 while pre-trigger, else the
    running direction), so a single missing/bad tick neither corrupts the running extremes nor forces a spurious
    flat. The array entry points (ctl_warm_up / ctl_streaming_replay) instead reject non-finite input, matching
    continuous_trend_labeling — clean warm-up data is required; only the live single-bar step is lenient.
    """
    if not math.isfinite(price):
        return state.direction
    if not state.initialized:
        state.first_price = price
        state.x_high = price; state.x_low = price
        state.t_high = i; state.t_low = i
        state.initialized = True
        return 0

    omega = state.omega
    if state.direction == 0:
        if price > state.first_price + state.first_price * omega:
            state.x_high = price; state.t_high = i; state.direction = 1
            return 1
        if price < state.first_price - state.first_price * omega:
            state.x_low = price; state.t_low = i; state.direction = -1
            return -1
        return 0

    if state.direction == 1:
        if price > state.x_high:
            state.x_high = price; state.t_high = i
        if price < state.x_high - state.x_high * omega and state.t_low <= state.t_high:
            state.x_low = price; state.t_low = i; state.direction = -1
            return -1
        return 1

    # state.direction == -1
    if price < state.x_low:
        state.x_low = price; state.t_low = i
    if price > state.x_low + state.x_low * omega and state.t_high <= state.t_low:
        state.x_high = price; state.t_high = i; state.direction = 1
        return 1
    return -1


def ctl_warm_up(prices: Union[pd.Series, np.ndarray], omega: float) -> CTLState:
    """Replay a warm-up history and return the LIVE state for incremental stepping.

    Use at startup: pass enough history that the FSM has burned in (direction != 0 with the running extreme on the
    current leg), keep the returned state, then feed each new bar to ctl_step with a monotonically increasing
    index. The first live index should be len(prices) (i.e. continue the warm-up index).
    """
    state = CTLState(omega=float(omega))  # validates omega (finite, > 0)
    x = _as_finite_price_array(prices)
    for i in range(len(x)):
        ctl_step(state, float(x[i]), i)
    return state


def ctl_streaming_replay(prices: Union[pd.Series, np.ndarray], omega: float) -> np.ndarray:
    """Replay a price series through the streaming FSM and return per-bar labels as int64 (0/-1/+1).

    Equivalence harness for the batch continuous_trend_labeling (warmup NaN <-> 0). For live use prefer
    ctl_warm_up (keeps the state) followed by per-bar ctl_step calls.
    """
    state = CTLState(omega=float(omega))  # validates omega (finite, > 0)
    x = _as_finite_price_array(prices)
    out = np.zeros(len(x), dtype=np.int64)
    for i in range(len(x)):
        out[i] = ctl_step(state, float(x[i]), i)
    return out


##############################################################################################################
########################## THREE-CLASS LABEL DERIVATION (BAND-GATED CTL) #####################################
##############################################################################################################
#
# Turns the binary CTL output into a {-1, 0, +1} ternary label by gating CTL labels with an ATR-based envelope band
# (MA +/- k*ATR). Class-0 marks bars where price sits inside the band — the noise / low-confidence zone — while +/-1 marks
# confident directional regimes.
#
# Calibration of (omega, ma_period, atr_period, k_atr) happens upstream; this module
# provides:
#   - compute_band_state — turn an (already-enveloped) close + (upper, lower) into a {-1, 0, +1} ternary state
#       (envelope itself is sourced from .misc),
#   - attach_labels for offline / batch use given an explicit (omega, BandParams),
#   - apply_3class_labels for runtime use that accepts a pre-resolved (omega, band) and rescales bar-count parameters
#       to the input timeframe. Caller is responsible for sourcing the config (typically from a SymbolMetastore block).


@dataclass(frozen=True)
class BandParams:
    """ATR envelope band: MA(ma_period) +/- k_atr * ATR(atr_period)."""
    ma_period: int
    atr_period: int
    k_atr: float

    def __post_init__(self):
        # bool is an int subclass in Python; reject it explicitly so True/False can't pose as a period.
        if isinstance(self.ma_period, bool) or not isinstance(self.ma_period, (int, np.integer)):
            raise ValueError(f"ma_period must be an integer, got {self.ma_period!r}")
        if self.ma_period < 2:
            raise ValueError("ma_period must be >= 2")
        if isinstance(self.atr_period, bool) or not isinstance(self.atr_period, (int, np.integer)):
            raise ValueError(f"atr_period must be an integer, got {self.atr_period!r}")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if not math.isfinite(self.k_atr) or self.k_atr <= 0:
            raise ValueError("k_atr must be > 0")


def compute_band_state(close: pd.Series, upper: pd.Series, lower: pd.Series) -> np.ndarray:
    """Ternary band state: +1 above upper, -1 below lower, 0 inside (or warmup NaN).

    POSITIONAL: the three inputs are compared element-by-element by position, NOT by pandas index. Pass
    same-length, same-order inputs (as attach_labels does); index labels are ignored. This prevents a
    reordered/mismatched index from silently pairing a close with the wrong band level — the previous
    implementation mixed index-aligned comparisons with a positional validity mask.
    """
    if not (len(close) == len(upper) == len(lower)):
        raise ValueError(
            f"compute_band_state: length mismatch — "
            f"close={len(close)}, upper={len(upper)}, lower={len(lower)}"
        )
    close_arr = np.asarray(close, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    state = np.zeros(len(close_arr), dtype=np.int8)
    valid = np.isfinite(upper_arr) & np.isfinite(lower_arr)
    state[(close_arr > upper_arr) & valid] = 1
    state[(close_arr < lower_arr) & valid] = -1
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

    Storage convention: the binary CTL label is the source-of-truth staging label; the 3-class column is a derived
    quasi-posterior used by downstream models that want a 'confidence' interpretation.

    Returns df with: 'ma', 'upper', 'lower', 'band_state', binary_col, ternary_col.

    Note: any pre-existing columns named 'ma', 'upper', 'lower', 'band_state', `binary_col`, or `ternary_col` on the
    input df are overwritten in the returned copy without warning.
    """
    out = df.copy()

    out["upper"], out["ma"], out["lower"], _, _ = envelope(out["close"], out["high"], out["low"],
        ma_period=band.ma_period, atr_period=band.atr_period, k_atr=band.k_atr)
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
