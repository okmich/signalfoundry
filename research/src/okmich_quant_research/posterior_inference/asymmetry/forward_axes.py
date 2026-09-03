"""Forward-axis outcome builders for posterior-asymmetry validation.

Turns a price frame into the causal, forward-measured outcome series that ``forward_outcome_by_state`` profiles — one
per axis — together with the **trailing baseline** of the same axis used by the incremental-over-baseline gate.

The baseline is uniform across axes: the *trailing* value of the same measure. Residualising the forward outcome on it
asks the single decisive question — *does the state forecast this axis beyond its own persistence?* For volatility that
persistence is clustering; for the directional axis it is return autocorrelation; etc. A state that only re-derives persistence adds
nothing a one-line trailing model doesn't already give.

Axes are ``registry.Axis`` -- the SAME enum the screener dispatches on, so Stage-1 selection and
Stage-2 validation provably measure the same thing rather than merely claiming to. ``DIRECTIONAL``'s
forward outcome is the signed forward log-return, which is exactly the target
``hmm_screener._evaluators._build_forward_log_returns`` scores; there is a test asserting the two agree
numerically, because a shared enum with divergent definitions would be worse than two enums.

A separate ``MOMENTUM`` outcome (forward return signed by the PRIOR move's direction) used to live here
and has been removed. Three reasons: momentum is not an axis (measured, it is trend at half the
lookback -- see ``registry._axis``); nothing referenced it; and it quietly broke this module's own
invariant that every baseline is the TRAILING value of the same measure -- its baseline was
direction-conditioned, which makes the quantity a strategy P&L rather than a market axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...features.registry import Axis
from .profiler import ForwardOutcome

_VOLUME_COLUMNS = ("tick_volume", "volume")


def forward_axis_series(prices: pd.DataFrame, axis: Axis, horizon: int) -> tuple[NDArray, NDArray]:
    """Return ``(forward, baseline)`` for one axis at ``horizon``, full-length and NaN at the unavailable edges.

    ``forward[t]`` is the axis outcome over ``(t, t+horizon]`` (NaN in the trailing ``horizon`` rows); ``baseline[t]`` is
    the trailing-``horizon`` value of the same axis (NaN in the leading warm-up rows). Both are aligned to ``prices.index``.
    """
    if horizon < 1:
        raise ValueError(f"forward_axis_series: horizon must be >= 1, got {horizon}.")
    if "close" not in prices.columns:
        raise ValueError("forward_axis_series: prices must contain a 'close' column.")
    close = prices["close"].astype(float)
    log_close = np.log(close)
    sq_ret = log_close.diff() ** 2

    if axis == Axis.DIRECTIONAL:
        forward = log_close.shift(-horizon) - log_close                       # signed forward log-return
        baseline = log_close - log_close.shift(horizon)                       # trailing return (its own persistence)
    elif axis == Axis.VOLATILITY:
        forward = np.sqrt(sq_ret.rolling(horizon).sum().shift(-horizon) / horizon)
        baseline = np.sqrt(sq_ret.rolling(horizon).sum() / horizon)          # trailing realized vol (clustering)
    elif axis == Axis.PATH_STRUCTURE:
        fwd_path = close.diff().abs().rolling(horizon).sum().shift(-horizon)
        forward = (close.shift(-horizon) - close).abs() / fwd_path.where(fwd_path != 0, np.nan)  # fwd efficiency ratio
        trail_path = close.diff().abs().rolling(horizon).sum()
        baseline = (close - close.shift(horizon)).abs() / trail_path.where(trail_path != 0, np.nan)
    elif axis == Axis.LIQUIDITY:
        vol_col = next((c for c in _VOLUME_COLUMNS if c in prices.columns), None)
        if vol_col is None:
            raise ValueError(f"forward_axis_series: LIQUIDITY axis needs one of {_VOLUME_COLUMNS} in prices.")
        volume = prices[vol_col].astype(float)
        forward = volume.rolling(horizon).sum().shift(-horizon)              # forward cumulative volume
        baseline = volume.rolling(horizon).sum()                            # trailing cumulative volume
    else:  # pragma: no cover - exhaustiveness guard
        raise ValueError(f"forward_axis_series: unknown axis {axis!r}.")

    return np.asarray(forward, dtype=float), np.asarray(baseline, dtype=float)


def build_forward_outcomes(prices: pd.DataFrame, axes: list[Axis], horizons: list[int]) -> dict[str, ForwardOutcome]:
    """Forward outcomes only (no baselines), keyed ``"{axis}_h{horizon}"`` — for direct profiling without the gate."""
    out: dict[str, ForwardOutcome] = {}
    for axis in axes:
        for horizon in horizons:
            forward, _ = forward_axis_series(prices, axis, horizon)
            out[f"{axis.value}_h{horizon}"] = ForwardOutcome(forward, horizon)
    return out
