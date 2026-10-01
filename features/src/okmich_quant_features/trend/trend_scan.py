"""Causal trend scan: the backward-looking form of López de Prado's trend-scanning method.

At every bar t, fit an OLS line to the (log) price over every trailing window t-L+1 .. t, for L in [min_window,
max_window], and keep the window whose slope has the largest |t-value|. Every value for bar t uses bars <= t only, so
the frame is causal by construction: appending future bars never changes a past row.

Reference: M. López de Prado, Machine Learning for Asset Managers (2020), section 5.4. The published method scans
FORWARD from t and is a training target that reads up to max_window future bars. Only the backward scan, which is
usable as a feature, lives here.

Reading the strength columns: the t-value of a slope fitted to a random walk's LEVELS grows like sqrt(L) (Durlauf &
Phillips 1988). It is therefore a strength score, not a significance test, and its argmax drifts toward max_window, so
ts_window == max_window means "at least max_window". R² carries the same information at a given length without that
inflation (t / sqrt(L - 2) = r / sqrt(1 - r²)), so ts_r2 / ts_strength are the strength measures to prefer.
"""

import math

import numpy as np
import pandas as pd
from numba import njit

_T_VALUE, _WINDOW, _R2, _SLOPE, _AGREEMENT, _LINE_GAP = range(6)


@njit(cache=True)
def _scan_kernel(y: np.ndarray, segment: np.ndarray, min_window: int, max_window: int) -> np.ndarray:
    """Per bar: t-value, length, R², slope, sign agreement and line gap of the max-|t| trailing window.

    Windows grow one bar at a time from t backwards, anchored at y[t] for numerical precision (x = -k for bar t-k),
    and stop at a change in `segment`. Rows stay NaN until a window of min_window bars fits.
    """
    n = y.size
    out = np.full((n, 6), np.nan)
    for t in range(n):
        sx = sxx = sv = svv = sxv = 0.0
        best_abs = -1.0
        best_sign = 0.0
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
                t_value = math.copysign(math.inf, beta)     # an exactly straight window
            else:
                t_value = 0.0
            if beta > 0:
                n_pos += 1
            elif beta < 0:
                n_neg += 1
            else:
                n_zero += 1
            if abs(t_value) > best_abs:
                best_abs = abs(t_value)
                best_sign = 1.0 if beta > 0 else (-1.0 if beta < 0 else 0.0)
                resid_sd = math.sqrt(ssr / (m - 2))
                fitted_at_t = sv / m - beta * sx / m        # in units anchored at y[t], so the residual is -fitted
                out[t, _T_VALUE] = t_value
                out[t, _WINDOW] = m
                out[t, _R2] = min(cxv * cxv / (cxx * cvv), 1.0) if cvv > 0 else 0.0
                out[t, _SLOPE] = beta
                out[t, _LINE_GAP] = -fitted_at_t / resid_sd if resid_sd > 0 else 0.0
        if best_abs >= 0:
            same = n_pos if best_sign > 0 else (n_neg if best_sign < 0 else n_zero)
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


def _segments(prices: pd.Series | np.ndarray, n: int, break_gap) -> np.ndarray:
    if break_gap is None:
        return np.zeros(n, dtype=np.int64)
    if not (isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex)):
        raise ValueError("break_gap requires a pd.Series with a DatetimeIndex")
    try:
        gap = pd.Timedelta(break_gap)
    except (TypeError, ValueError) as error:
        message = f"break_gap must be a duration such as a Timedelta or '30min', got {break_gap!r}"
        raise ValueError(message) from error
    if pd.isna(gap) or gap <= pd.Timedelta(0):
        raise ValueError(f"break_gap must be positive, got {break_gap!r}")
    gaps = prices.index.to_series().diff().to_numpy()       # unit-safe: the index may be ns, us, s ...
    return np.cumsum(gaps > gap.to_timedelta64()).astype(np.int64)


def trend_scan_features(prices: pd.Series | np.ndarray, min_window: int = 6, max_window: int = 72,
                        log_prices: bool = True, break_gap: pd.Timedelta | str | None = None,
                        state_enter: float | None = None, state_exit: float = 0.0) -> pd.DataFrame:
    """Causal per-bar trend-scan features: the max-|t| straight line through the trailing prices.

    Returns a float64 DataFrame whose index mirrors the input (RangeIndex for arrays). A row is NaN until min_window
    bars are available within the current segment; with break_gap set, every gap larger than break_gap (a weekend, a
    session close) starts a new segment, so no window ever spans it.

    Columns:
        ts_direction  +1 / -1: sign of the winning slope (0 only for a perfectly flat window).
        ts_strength   ts_direction * ts_r2, in [-1, 1]. The length-neutral signed strength.
        ts_r2         R² of the winning window: how closely the prices hug the line (0 = not at all, 1 = exactly).
        ts_window     Length in bars of the winning window. Piles up at max_window, which means "at least" that.
        ts_t_value    Signed OLS t-value of the winning slope. A strength score, not a significance test (see the
                      module docstring). It is ±inf only for a window straight to machine precision.
        ts_slope      Winning slope per bar (log-price units when log_prices=True): the speed of the trend.
                      Normalise by volatility before comparing instruments or regimes.
        ts_agreement  Share of the window lengths tried at this bar whose slope has the winner's sign. 1.0 = every
                      scale agrees; near 0.5 = short and long scales disagree (a pullback or a turn).
        ts_line_gap   Current price minus the winning line's fitted value at this bar, in residual standard
                      deviations: > 0 stretched above the line, < 0 pulled below it.
        ts_state      Only when state_enter is given. +1 UP / 0 NEUTRAL / -1 DOWN by hysteresis on ts_strength:
                      enter UP at >= state_enter (DOWN at <= -state_enter), leave UP for NEUTRAL below state_exit
                      (DOWN above -state_exit); an opposite reading beyond state_enter flips directly. Resets to
                      NEUTRAL at a segment start; NaN wherever ts_strength is NaN.

    Live use: every column except ts_state depends only on the last max_window bars of the current segment, so the
    live value equals this function's last row on that trailing slice. ts_state is path-dependent: carry it by
    replaying from a warm start, as with any hysteresis state.

    Args:
        prices: Close prices, finite (and strictly positive when log_prices=True). Series or ndarray.
        min_window: Shortest window tried (>= 3, so the slope has a residual degree of freedom).
        max_window: Longest window tried (>= min_window). Anchor it to a wall-clock horizon, e.g. 72 bars of 5m = 6 h.
        log_prices: Fit the line to log(price) (default) or to the raw price.
        break_gap: Optional largest allowed gap between consecutive stamps (a Timedelta, datetime.timedelta or a
            string such as "30min"); needs a DatetimeIndex Series.
        state_enter: Enables ts_state; strength needed to enter UP or DOWN, in (0, 1].
        state_exit: Strength below which UP (above which DOWN) falls back to NEUTRAL, in [0, state_enter].

    Raises:
        ValueError: on non-integer or invalid windows, invalid thresholds, input that is not one-dimensional,
            non-finite prices, non-positive prices with log_prices=True, or a non-positive break_gap or one given
            without a DatetimeIndex Series.
    """
    for name, window in (("min_window", min_window), ("max_window", max_window)):
        if isinstance(window, bool) or not float(window).is_integer():
            raise ValueError(f"{name} must be a whole number of bars, got {window!r}")
    if min_window < 3:
        raise ValueError(f"min_window must be >= 3, got {min_window}")
    if max_window < min_window:
        raise ValueError(f"max_window must be >= min_window, got {max_window} < {min_window}")
    if state_enter is not None and not (0.0 < state_enter <= 1.0):
        raise ValueError(f"state_enter must be in (0, 1], got {state_enter}")
    if state_enter is not None and not (0.0 <= state_exit <= state_enter):
        raise ValueError(f"state_exit must be in [0, state_enter], got {state_exit}")

    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"prices must be one-dimensional (a Series or 1-D array), got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("prices contains NaN or infinite values")
    if log_prices and (values <= 0).any():
        raise ValueError("log_prices=True needs strictly positive prices")
    index = prices.index if isinstance(prices, pd.Series) else pd.RangeIndex(values.size)
    segment = _segments(prices, values.size, break_gap)

    scan = _scan_kernel(np.log(values) if log_prices else values, segment, int(min_window), int(max_window))
    direction = np.sign(scan[:, _T_VALUE])
    result = pd.DataFrame({"ts_direction": direction, "ts_strength": direction * scan[:, _R2],
                           "ts_r2": scan[:, _R2], "ts_window": scan[:, _WINDOW], "ts_t_value": scan[:, _T_VALUE],
                           "ts_slope": scan[:, _SLOPE], "ts_agreement": scan[:, _AGREEMENT],
                           "ts_line_gap": scan[:, _LINE_GAP]}, index=index)
    if state_enter is not None:
        result["ts_state"] = _hysteresis_kernel(result["ts_strength"].to_numpy(), float(state_enter),
                                                float(state_exit))
    return result
