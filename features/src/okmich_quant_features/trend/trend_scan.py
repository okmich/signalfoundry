"""Causal trend scan: the backward-looking form of López de Prado's trend-scanning method.

At every bar t, fit an OLS line to the (log) price over every trailing window t-L+1 .. t, for L in [min_window,
max_window], and keep the window whose slope has the largest |t-value|. Every value for bar t uses rows <= t only, so
the frame is causal by construction: appending future bars never changes a past row. Rows are taken in the order given;
sort by time first.

Reference: M. López de Prado, Machine Learning for Asset Managers (2020), section 5.4. The published method scans
FORWARD from t and is a training target that reads up to max_window future bars. Only the backward scan, which is
usable as a feature, lives here.

Reading the strength columns. None of them is a significance test, and none is neutral on pure noise:
    * The t-value of a slope fitted to a random walk's LEVELS grows like sqrt(L) (Durlauf & Phillips 1988), so the
      scan's argmax drifts toward max_window. On a driftless random walk with the defaults about 27% of bars pick
      max_window: ts_window is the length that scored best, not how long a trend has lasted.
    * R² does not grow with L the way t does (t / sqrt(L - 2) = r / sqrt(1 - r²)), so ts_r2 / ts_strength are the
      bounded strength measures to use. But the winning window is the best of up to max_window - min_window + 1 fits,
      so its R² is biased upward by that choice: on a driftless random walk the mean ts_r2 is about 0.74-0.78 for
      max_window 24-288, and |ts_strength| > 0.7 on roughly 70% of bars. Calibrate every threshold (including the
      ts_state ones) against a null, e.g. this function on the same returns with shuffled or flipped signs.
"""

import math
import numbers

import numpy as np
import pandas as pd
from numba import njit

_DIRECTION, _T_VALUE, _WINDOW, _R2, _SLOPE, _AGREEMENT, _LINE_GAP = range(7)
_MIN_WINDOW_FLOOR = 5  # 3 residual degrees of freedom: shorter windows have fat-tailed t and win the scan by chance
_MAX_WINDOW_CAP = 1_000_000


@njit(cache=True)
def _scan_kernel(y: np.ndarray, segment: np.ndarray, min_window: int, max_window: int) -> np.ndarray:
    """Per bar: direction, t-value, length, R², slope, sign agreement and line gap of the max-|t| trailing window.

    Windows grow one bar at a time from t backwards, anchored at y[t] for numerical precision (x = -k for bar t-k),
    and stop at a change in `segment`. Rows stay NaN until a window of min_window bars fits. A window straight to
    machine precision has an infinite |t| (it wins the scan) and its reported t-value is NaN (undefined).
    """
    n = y.size
    out = np.full((n, 7), np.nan)
    for t in range(n):
        sx = sxx = sv = svv = sxv = 0.0
        best_abs, best_t, best_m = -1.0, 0.0, 0
        best_beta = best_ssr = best_sx = best_sv = best_cxx = best_cvv = best_cxv = 0.0
        n_pos = n_neg = n_zero = 0
        for k in range(max_window):
            j = t - k
            if j < 0 or segment[j] != segment[t]:
                break
            x = -float(k)
            v = y[j] - y[t]
            sx += x
            sxx += x * x
            sv += v
            svv += v * v
            sxv += x * v
            m = k + 1
            if m < min_window:
                continue
            cxx = sxx - sx * sx / m
            cvv = svv - sv * sv / m
            cxv = sxv - sx * sv / m
            beta = cxv / cxx
            ssr = max(cvv - beta * cxv, 0.0)
            se2 = ssr / (m - 2) / cxx
            if se2 > 0:
                t_value = beta / math.sqrt(se2)
            elif beta != 0:
                t_value = math.copysign(math.inf, beta)     # a window straight to machine precision
            else:
                t_value = 0.0
            if beta > 0:
                n_pos += 1
            elif beta < 0:
                n_neg += 1
            else:
                n_zero += 1
            if abs(t_value) > best_abs:
                best_abs, best_t, best_m = abs(t_value), t_value, m
                best_beta, best_ssr, best_sx, best_sv = beta, ssr, sx, sv
                best_cxx, best_cvv, best_cxv = cxx, cvv, cxv
        if best_abs < 0:
            continue
        sign = 1.0 if best_beta > 0 else (-1.0 if best_beta < 0 else 0.0)
        resid_sd = math.sqrt(best_ssr / (best_m - 2))
        fitted_at_t = best_sv / best_m - best_beta * best_sx / best_m  # anchored at y[t], so the residual is -fitted
        out[t, _DIRECTION] = sign
        out[t, _T_VALUE] = best_t if math.isfinite(best_t) else np.nan
        out[t, _WINDOW] = best_m
        out[t, _R2] = min(best_cxv * best_cxv / (best_cxx * best_cvv), 1.0) if best_cvv > 0 else 0.0
        out[t, _SLOPE] = best_beta
        out[t, _LINE_GAP] = -fitted_at_t / resid_sd if resid_sd > 0 else 0.0
        same = n_pos if sign > 0 else (n_neg if sign < 0 else n_zero)
        out[t, _AGREEMENT] = same / (n_pos + n_neg + n_zero)
    return out


@njit(cache=True)
def _hysteresis_kernel(strength: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """+1 UP / 0 NEUTRAL / -1 DOWN with separate enter and exit thresholds; NaN strength resets to NEUTRAL."""
    n = strength.size
    out = np.full(n, np.nan)
    state = 0.0
    for i in range(n):
        s = strength[i]
        if np.isnan(s):
            state = 0.0
            continue
        if state == 1.0:
            if s <= -enter:
                state = -1.0
            elif s < exit_:
                state = 0.0
        elif state == -1.0:
            if s >= enter:
                state = 1.0
            elif s > -exit_:
                state = 0.0
        elif s >= enter:
            state = 1.0
        elif s <= -enter:
            state = -1.0
        out[i] = state
    return out


def _whole_bars(name: str, value) -> int:
    """A window length: a whole number of bars (int or integral float), never a bool, string or None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a whole number of bars, got {value!r}")
    if not float(value).is_integer() or not 0 < value <= _MAX_WINDOW_CAP:
        raise ValueError(f"{name} must be a whole number of bars in [1, {_MAX_WINDOW_CAP}], got {value!r}")
    return int(value)


def _threshold(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _break_gap(break_gap) -> pd.Timedelta:
    """Parse a duration. A bare number has no unit (pd.Timedelta would read 30 as 30 nanoseconds), so it is refused."""
    if not isinstance(break_gap, np.timedelta64) and (
            isinstance(break_gap, (bool, numbers.Number))
            or (isinstance(break_gap, str) and not any(c.isalpha() for c in break_gap))):
        raise ValueError(f"break_gap needs a unit, e.g. '30min' or pd.Timedelta(minutes=30); got {break_gap!r}, which "
                         "would be read as nanoseconds")
    try:
        gap = pd.Timedelta(break_gap)
    except (TypeError, ValueError) as error:
        raise ValueError(f"break_gap must be a duration such as a Timedelta or '30min', got {break_gap!r}") from error
    if pd.isna(gap) or gap <= pd.Timedelta(0):
        raise ValueError(f"break_gap must be positive, got {break_gap!r}")
    return gap


def _segments(prices: pd.Series | np.ndarray, n: int, break_gap, segment) -> np.ndarray:
    """Segment id per row: a new segment starts wherever ``segment`` changes or a time gap exceeds ``break_gap``."""
    starts = np.zeros(n, dtype=bool)
    if segment is not None:
        labels = np.asarray(segment)
        if labels.ndim != 1 or labels.size != n:
            raise ValueError(f"segment needs one label per price ({n}), got shape {labels.shape}")
        if pd.isna(labels).any():
            raise ValueError("segment contains missing labels")
        starts[1:] |= labels[1:] != labels[:-1]
    if break_gap is not None:
        if not (isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex)):
            raise ValueError("break_gap requires a pd.Series with a DatetimeIndex")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("break_gap needs a time-sorted index; sort the prices first")
        gap = _break_gap(break_gap)
        gaps = prices.index.to_series().diff()                 # unit-safe: the index may be ns, us, s ...
        spacing = gaps.median()
        if pd.notna(spacing) and gap < spacing:
            raise ValueError(f"break_gap {gap} is shorter than the median bar spacing {spacing}: every bar would start "
                             "a new segment")
        starts |= (gaps > gap).to_numpy()
    return np.cumsum(starts).astype(np.int64)


def trend_scan_features(prices: pd.Series | np.ndarray, min_window: int = 6, max_window: int = 72,
                        log_prices: bool = True, break_gap: pd.Timedelta | str | None = None,
                        state_enter: float | None = None, state_exit: float = 0.0,
                        segment: np.ndarray | pd.Series | None = None) -> pd.DataFrame:
    """Causal per-bar trend-scan features: the max-|t| straight line through the trailing prices.

    Returns a float64 DataFrame whose index mirrors the input (RangeIndex for arrays). A row is NaN until min_window
    bars are available within the current segment. Segments split the series so no window ever spans a break: with
    ``break_gap``, every time gap larger than it (a weekend, a session close) starts a new segment; with ``segment``,
    every change of label does (e.g. a daily rollover, which leaves no time gap). Both may be given.

    Columns (read the module docstring before thresholding any of them):
        ts_direction  +1 / -1: sign of the winning slope (0 only for a perfectly flat window).
        ts_strength   ts_direction * ts_r2, in [-1, 1]. Bounded and not inflated by window length the way the
                      t-value is, but biased upward by picking the best window: calibrate thresholds against a null.
        ts_r2         R² of the winning window: how closely the prices hug the line (0 = not at all, 1 = exactly).
        ts_window     Length in bars of the winning window. It piles up at max_window even on noise, because |t|
                      grows with length on price levels; it is not a trend duration.
        ts_t_value    Signed OLS t-value of the winning slope: a heavy-tailed score, not a significance test. NaN when
                      the winning window is straight to machine precision (its t is undefined), e.g. an equal-step
                      price ladder with log_prices=False. Prefer ts_strength as a model input.
        ts_slope      Winning slope per bar (log-price units when log_prices=True): the speed of the trend.
                      Normalise by volatility before comparing instruments or regimes.
        ts_agreement  Share of the window lengths tried at this bar whose slope has the winner's sign. 1.0 = every
                      scale agrees; near 0.5 = short and long scales disagree (a pullback or a turn).
        ts_line_gap   Current price minus the winning line's fitted value at this bar, in residual standard
                      deviations: > 0 stretched above the line, < 0 pulled below it.
        ts_state      Only when state_enter is given. +1 UP / 0 NEUTRAL / -1 DOWN by hysteresis on ts_strength:
                      enter UP at >= state_enter (DOWN at <= -state_enter), leave UP for NEUTRAL below state_exit
                      (DOWN above -state_exit); an opposite reading beyond state_enter flips directly. Restarts from
                      NEUTRAL at every segment start; NaN wherever ts_strength is NaN.

    Live use: every column except ts_state depends only on the last max_window bars of the current segment, so the
    live value equals this function's last row on that trailing slice. ts_state is path-dependent: carry it by
    replaying from a warm start, as with any hysteresis state.

    Args:
        prices: Close prices in time order, finite (and strictly positive when log_prices=True). Series or ndarray.
        min_window: Shortest window tried, >= 5 (at least 3 residual degrees of freedom; a shorter window's t-value is
            so heavy-tailed that it wins the scan by chance).
        max_window: Longest window tried (>= min_window). Anchor it to a wall-clock horizon, e.g. 72 bars of 5m = 6 h.
        log_prices: Fit the line to log(price) (default) or to the raw price.
        break_gap: Optional largest allowed gap between consecutive stamps, with a unit (a Timedelta,
            datetime.timedelta, np.timedelta64 or a string such as "30min"); needs a time-sorted DatetimeIndex Series.
        state_enter: Enables ts_state; strength needed to enter UP or DOWN, in (0, 1].
        state_exit: Strength below which UP (above which DOWN) falls back to NEUTRAL, in [0, state_enter]. Only
            meaningful with state_enter.
        segment: Optional label per price; every change of label starts a new segment.

    Raises:
        ValueError: on invalid windows or thresholds, state_exit without state_enter, input that is not
            one-dimensional, non-finite prices, non-positive prices with log_prices=True, a segment of the wrong
            length, or a break_gap that has no unit, is not positive, is shorter than the bar spacing, or comes
            without a time-sorted DatetimeIndex Series.
    """
    min_window, max_window = _whole_bars("min_window", min_window), _whole_bars("max_window", max_window)
    if min_window < _MIN_WINDOW_FLOOR:
        raise ValueError(f"min_window must be >= {_MIN_WINDOW_FLOOR} (at least 3 residual degrees of freedom), got "
                         f"{min_window}")
    if max_window < min_window:
        raise ValueError(f"max_window must be >= min_window, got {max_window} < {min_window}")
    state_exit = _threshold("state_exit", state_exit)
    if state_enter is None:
        if state_exit != 0.0:
            raise ValueError("state_exit only applies with state_enter; pass state_enter to get ts_state")
    else:
        state_enter = _threshold("state_enter", state_enter)
        if not 0.0 < state_enter <= 1.0:
            raise ValueError(f"state_enter must be in (0, 1], got {state_enter}")
        if not 0.0 <= state_exit <= state_enter:
            raise ValueError(f"state_exit must be in [0, state_enter], got {state_exit}")

    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"prices must be one-dimensional (a Series or 1-D array), got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("prices contains NaN or infinite values")
    if log_prices and (values <= 0).any():
        raise ValueError("log_prices=True needs strictly positive prices")
    index = prices.index if isinstance(prices, pd.Series) else pd.RangeIndex(values.size)
    segments = _segments(prices, values.size, break_gap, segment)

    scan = _scan_kernel(np.log(values) if log_prices else values, segments, min_window, max_window)
    result = pd.DataFrame({"ts_direction": scan[:, _DIRECTION], "ts_strength": scan[:, _DIRECTION] * scan[:, _R2],
                           "ts_r2": scan[:, _R2], "ts_window": scan[:, _WINDOW], "ts_t_value": scan[:, _T_VALUE],
                           "ts_slope": scan[:, _SLOPE], "ts_agreement": scan[:, _AGREEMENT],
                           "ts_line_gap": scan[:, _LINE_GAP]}, index=index)
    if state_enter is not None:
        result["ts_state"] = _hysteresis_kernel(result["ts_strength"].to_numpy(), state_enter, state_exit)
    return result
