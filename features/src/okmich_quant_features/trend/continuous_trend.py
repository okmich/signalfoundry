"""Continuous Trend Labeling (price-action state machine) and its causal per-bar feature bundle.

Reference: https://www.mdpi.com/1099-4300/22/10/1162

continuous_trend_labeling tracks price extremes and reversals sequentially using a single omega threshold,
emitting a binary {-1, +1} regime label (0 only during pre-trigger warmup). Omega is caller-supplied; pick it
from a vol anchor (e.g. ~ k * median ATR / price) rather than fitting.

The raw +/-1 label is a thin summary of a machine that knows far more. ctl_trend_features (bottom of the file)
replays the same FSM once and projects its running state into a causal, no-look-ahead feature DataFrame — trend
age, retracement toward a flip, realised leg return, and recent flip count — the actually-useful signal the
state machine carries. See ctl_trend_features.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple, Union

import numpy as np
import pandas as pd


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


class CTLFeatures(NamedTuple):
    """One bar of the causal CTL feature bundle — the streaming twin of a ctl_trend_features DataFrame row.

    Field names match ctl_trend_features' columns exactly, so a live per-bar vector and a backtest row are
    directly comparable (bar-for-bar identical by construction — the batch function replays this same stepper).
    Use `._asdict()` for a dict or `list(feats)` for a raw ordered vector.
    """
    ctl_direction: float
    ctl_trend_age: float
    ctl_retrace_frac: float
    ctl_leg_return: float
    ctl_flip_count: float


# Pre-trigger bar: no confirmed leg (direction/age/retrace/leg-return unknown) and no flips counted yet. Shared
# immutable default for every CTLState's held vector; mirrors the warmup rows of ctl_trend_features.
_WARMUP_FEATURES = CTLFeatures(np.nan, np.nan, np.nan, np.nan, 0.0)


@dataclass
class CTLState:
    """O(1) streaming state for continuous_trend_labeling — a faithful per-bar projection of the batch machine.

    Warm up once, then advance one bar at a time — with ctl_step for the label only, or step_features for the full
    causal feature bundle (CTLFeatures). Field names mirror the batch function's locals; paper symbols are noted in
    comments. The leading fields are the label FSM; the `_`-prefixed fields are step_features bookkeeping (untouched
    by ctl_step) and are excluded from repr/eq.
    """
    omega: float
    flip_window: int = 20        # trailing bars for ctl_flip_count (step_features only)
    direction: int = 0           # paper: Cid — 0 pre-trigger, +1 up, -1 down
    first_price: float = 0.0     # paper: FP — anchor for the initial trigger
    x_high: float = 0.0          # running max (paper: xH)
    t_high: int = 0              # index of running max (paper: HT)
    x_low: float = 0.0           # running min (paper: xL)
    t_low: int = 0               # index of running min (paper: LT)
    initialized: bool = False
    # --- step_features bookkeeping (populated per bar; ignored by ctl_step / the label FSM) ---
    _i: int = field(default=-1, init=False, repr=False, compare=False)               # last processed bar index
    _prev_direction: int = field(default=0, init=False, repr=False, compare=False)   # direction before this bar
    _leg_start_idx: int = field(default=0, init=False, repr=False, compare=False)    # index the current leg began
    _flip_indices: deque = field(default_factory=deque, init=False, repr=False, compare=False)  # recent flip bars
    _last_features: CTLFeatures = field(default=_WARMUP_FEATURES, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not math.isfinite(self.omega) or self.omega <= 0:
            raise ValueError(f"omega must be > 0, got {self.omega}")
        if self.flip_window < 1:
            raise ValueError(f"flip_window must be >= 1, got {self.flip_window}")

    def step_features(self, price: float) -> CTLFeatures:
        """Advance one bar and return that bar's full causal feature bundle (live/online use).

        The streaming twin of a ctl_trend_features row: identical values bar-for-bar. The instance owns a monotonic
        bar counter, so live code just calls this once per bar in order — warm up by stepping over enough history
        first (this primes the leg origin and the trailing flip window, the same burn-in the label needs).

        Non-finite price (NaN/inf): the last emitted bundle is returned unchanged and neither the FSM state nor the
        bar counter advances — the same hold-on-bad-tick contract as ctl_step. Feed clean data during warm-up.

        Pre-trigger warmup returns CTLFeatures(nan, nan, nan, nan, 0.0): no confirmed leg, no flips counted.
        """
        if not math.isfinite(price):
            return self._last_features  # hold: FSM + counter untouched, mirroring ctl_step's bad-tick behaviour

        self._i += 1
        i = self._i
        prev_dir = self._prev_direction
        cur_dir = ctl_step(self, price, i)  # advances direction + running extremes in place; returns 0/-1/+1

        if cur_dir == 0:
            feats = _WARMUP_FEATURES
        else:
            if cur_dir != prev_dir:
                self._leg_start_idx = i
                if prev_dir != 0:                 # a reversal — the initial 0 -> +/-1 trigger is not a flip
                    self._flip_indices.append(i)
            cutoff = i - self.flip_window          # drop flips that have aged out of the trailing window
            while self._flip_indices and self._flip_indices[0] <= cutoff:
                self._flip_indices.popleft()
            if cur_dir == 1:
                retrace = (self.x_high - price) / self.x_high / self.omega  # pullback below the running high
                leg_ret = (price - self.x_low) / self.x_low                 # gain from the origin low
            else:
                retrace = (price - self.x_low) / self.x_low / self.omega    # bounce above the running low
                leg_ret = (price - self.x_high) / self.x_high               # drop from the origin high
            feats = CTLFeatures(ctl_direction=float(cur_dir), ctl_trend_age=float(i - self._leg_start_idx),
                                ctl_retrace_frac=retrace, ctl_leg_return=leg_ret,
                                ctl_flip_count=float(len(self._flip_indices)))

        self._prev_direction = cur_dir
        self._last_features = feats
        return feats


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
############################### CAUSAL TREND-FEATURE BUNDLE (FROM THE CTL FSM) ###############################
##############################################################################################################
#
# The raw +/-1 CTL label discards almost everything the state machine computes. ctl_trend_features replays the SAME
# per-bar stepper used live (CTLState.step_features, over ctl_step) once — no separate copy of the algorithm — and
# projects the running state (direction, running extremes, leg origin) into a per-bar feature DataFrame. Every column
# at bar i depends only on prices[:i+1], so the bundle is fully causal (no look-ahead) and safe as a model input.
#
# A "leg" is the stretch between two confirmed pivots; it starts at the initial trigger or at a reversal. The favourable
# extreme of an up-leg is its running high (x_high); its origin is the low it rose from (x_low). Down-legs mirror this.


def ctl_trend_features(prices: Union[pd.Series, np.ndarray], omega: float = 0.15, flip_window: int = 20) -> pd.DataFrame:
    """Causal per-bar trend features projected from the CTL state machine.

    Replays the streaming FSM once and returns a float64 DataFrame (index mirrors the input; RangeIndex for arrays).
    All columns are NaN over the pre-trigger warmup, where the machine has no confirmed leg — except ctl_flip_count,
    a trailing count that is a meaningful 0 there.

    Backtest/live parity: each row is produced by CTLState.step_features, the same per-bar stepper used live, so a
    live CTLFeatures vector equals this function's row for the same bar by construction. For online use, hold a
    CTLState(omega, flip_window) and call state.step_features(price) once per bar.

    Columns:
        ctl_direction    +1 up / -1 down / NaN warmup. Bar-for-bar identical to continuous_trend_labeling.
        ctl_trend_age    Bars since the current leg started (0 on the leg's first bar). Trend persistence / maturity.
        ctl_retrace_frac Pullback from the leg's favourable extreme as a fraction of omega. ~[0, 1): 0 while price
                         hugs the extreme, approaching 1 as it nears the omega retrace that flips the leg (the flip
                         bar resets the new leg to 0). Rarely exceeds 1 only when the pivot-ordering guard in the FSM
                         defers an otherwise-due reversal. A continuous proximity-to-reversal signal.
        ctl_leg_return   Signed % move from the leg's origin extreme to the current close — realised leg magnitude.
        ctl_flip_count   Number of reversals within the trailing flip_window bars — choppiness / whipsaw detector.

    Args:
        prices: Strictly-positive close prices (CTL's domain — omega is a % of price). Series or ndarray.
        omega: CTL threshold (dimensionless percentage); same meaning as in continuous_trend_labeling.
        flip_window: Trailing window (bars) for ctl_flip_count.

    Raises:
        ValueError: non-finite prices, omega <= 0 (via CTLState), or flip_window < 1.
    """
    is_series = isinstance(prices, pd.Series)
    x = _as_finite_price_array(prices)  # strict finite (matches continuous_trend_labeling); only the live step is lenient
    n = len(x)
    index = prices.index if is_series else pd.RangeIndex(n)

    # Replay the SAME per-bar stepper used live, so this DataFrame and a streaming CTLFeatures sequence are identical
    # by construction. CTLState validates omega (finite, > 0) and flip_window (>= 1).
    state = CTLState(omega=float(omega), flip_window=flip_window)
    rows = [state.step_features(float(x[i])) for i in range(n)]

    mat = np.array(rows, dtype=np.float64) if n else np.empty((0, len(CTLFeatures._fields)), dtype=np.float64)
    return pd.DataFrame({name: mat[:, k] for k, name in enumerate(CTLFeatures._fields)}, index=index)
