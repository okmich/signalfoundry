"""Timing-significance & beta/timing decomposition for a traded position series (source-agnostic).

Two complementary primitives for the question *"is this strategy's return TIMING skill, or just structural BETA?"* —
promoted from the posterior-dynamics meta-labeling workstream, where they were the tools that separated a real
(but modest) timing edge from trend-beta.

  * ``circular_shift_null`` — rigidly roll the realised positions in time by random offsets and re-score each
    shifted copy against the *same* returns. The shift preserves EVERYTHING structural (trade count, holding
    durations, long/short balance, turnover, autocorrelation) and breaks ONLY price-alignment, so beating the
    null isolates **timing**. The null's own mean IS the structure/beta benchmark. This is the correct
    significance test for timing — unlike a permutation shuffle, which destroys the hold structure and only
    rewards persistence (min-hold debounce can make random regimes look tradeable against a shuffle null, but
    not against this one). The offsets are distinct and at least ``min_offset`` away from the real alignment
    (in either direction); the p-value is ``(1 + #{null >= real}) / (1 + B)``, so it is never 0.

    A bar roll moves positions to a different time of day, so on intraday data a time-of-day pattern in returns
    (session drift, rollover) can pass for timing skill. Pass the bars' tz-aware ``index`` to rotate on the
    trading-day lattice instead: positions move by whole trading days and every bar keeps its intraday slot, so
    the null holds the same time-of-day exposure and only the day alignment is broken. Use it for intraday data.

  * ``beta_timing_decomposition`` — split the realised return ``pos·r`` into a constant-average-exposure
    (**beta**) component and a vary-the-exposure (**timing**) component:
    ``pos_{t-1}·r_t = mean(pos)·r_t + (pos_{t-1} − mean(pos))·r_t``.

Neither substitutes for the other: the decomposition *measures* the beta/timing split; the null tests whether
the timing part is *significant* against random re-alignments of the same position structure. A beta-harvesting
strategy is expected to *fail* the timing null (its edge is the structure) — that is not a defect, it is the
point; judge such a strategy on risk-adjusted portfolio return, not on this null.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from okmich_quant_labelling.utils.regime_diagnostics.clock import build_clock
from okmich_quant_labelling.utils.regime_diagnostics.null import DayLattice, draw_offsets, p_value


def _checked_positions(positions: NDArray) -> NDArray:
    """Positions as a 1-D float array with NaN → 0 (flat); reject inf (``np.nan_to_num`` would turn it huge)."""
    raw = np.asarray(positions, dtype=float)
    if raw.ndim != 1:
        raise ValueError(f"positions must be 1-D, got shape {raw.shape}.")
    if np.isinf(raw).any():
        raise ValueError("positions contains inf; only finite values (or NaN, treated as flat) are allowed.")
    return np.nan_to_num(raw)


def _checked_spread(spread_frac: NDArray | float, n: int) -> NDArray:
    """Broadcast ``spread_frac`` to a length-``n`` per-bar array; must be finite and non-negative."""
    sf = np.asarray(spread_frac, dtype=float)
    if sf.ndim == 0:
        sf = np.full(n, float(sf))
    elif sf.ndim != 1 or sf.shape[0] != n:
        raise ValueError(f"spread_frac must be a scalar or 1-D of length {n}, got shape {sf.shape}.")
    if not np.isfinite(sf).all():
        raise ValueError("spread_frac must be finite.")
    if (sf < 0).any():
        raise ValueError("spread_frac must be non-negative (a negative half-spread would model a rebate).")
    return sf


def net_bar_returns(positions: NDArray, bar_returns: NDArray, *, spread_frac: NDArray | float = 0.0) -> NDArray:
    """Causal per-bar net strategy return. The position known at bar ``t-1`` earns bar ``t``'s return; a change
    in position pays turnover cost ``|Δposition| × spread_frac`` at the bar it happens.

    ``positions`` and ``bar_returns`` are 1-D, same length, time-ordered. ``spread_frac`` is the per-bar
    fractional half-spread (scalar or per-bar array; finite, non-negative). A NaN *return* contributes 0 to the
    gross P&L, but the **turnover cost on that bar is still charged** — a trade on a missing-data bar is not free;
    the caller should avoid changing position on gap bars if no trade actually occurred. NaN *positions* are
    treated as flat; inf positions are rejected.
    """
    pos = _checked_positions(positions)
    r = np.asarray(bar_returns, dtype=float)
    if pos.shape != r.shape:
        raise ValueError(f"positions {pos.shape} and bar_returns {r.shape} must be the same length.")
    sf = _checked_spread(spread_frac, pos.shape[0])
    lag = np.concatenate(([0.0], pos[:-1]))                     # position applied to bar t = previous bar's position
    turnover = np.abs(np.diff(pos, prepend=0.0))               # |pos_t - pos_{t-1}|
    gross = lag * r
    gross = np.where(np.isfinite(gross), gross, 0.0)           # missing return -> 0 gross contribution ...
    return gross - turnover * sf                                # ... but the turnover cost is always charged


def _ann_sharpe(returns: NDArray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if r.size > 1 and sd > 0 else float("nan")


@dataclass(frozen=True)
class CircularShiftNull:
    """Result of the circular-shift timing null.

    ``real_sharpe`` — annualised Sharpe of the actual positions.
    ``beta_sharpe`` — mean Sharpe across the shifted copies = the structure/beta benchmark (what the same
      position structure earns at random alignment).
    ``percentile`` — fraction of finite null Sharpes strictly below ``real_sharpe``; descriptive (≈ 0.5 when the
      return is pure structure, → 1.0 when timing dominates). It reaches 1.0 against a finite null, so it is not a
      p-value: significance uses ``p_value``. NaN if unscoreable.
    ``null_sharpes`` — the (n_null,) null distribution, one Sharpe per distinct offset.
    ``n_shuffle`` — the number of offsets requested.
    ``p_value`` — one-sided timing p-value ``(1 + #{null >= real}) / (1 + B)`` over the B finite null Sharpes. Never
      0: its floor is ``1 / (1 + B)``. NaN if unscoreable.
    ``n_null`` — distinct offsets actually drawn (``<= n_shuffle``; fewer when the series admits fewer).
    """

    real_sharpe: float
    beta_sharpe: float
    percentile: float
    null_sharpes: NDArray
    n_shuffle: int
    p_value: float
    n_null: int

    def clears(self, level: float = 0.95) -> bool:
        """Whether the real Sharpe beats the null at ``level``: ``p_value <= 1 - level`` (default p <= 0.05) = timing
        skill present. The smallest attainable p is ``1 / (1 + B)``, so below 19 finite null draws 0.95 never clears.
        """
        return bool(np.isfinite(self.p_value) and self.p_value <= 1 - level)


def circular_shift_null(positions: NDArray, bar_returns: NDArray, *, periods_per_year: float, n_shuffle: int = 200,
                        spread_frac: NDArray | float = 0.0, min_offset: int = 200, seed: int = 0,
                        index: pd.DatetimeIndex | None = None) -> CircularShiftNull:
    """Circular-shift timing null (see module docstring). Draws up to ``n_shuffle`` DISTINCT offsets, each at least
    ``min_offset`` bars from the real alignment in either direction, and re-scores the shifted positions against the
    fixed ``bar_returns``. Set ``min_offset`` to at least the holding period so a shifted copy never re-reads its own
    trades; a series too short to honour it raises ``ValueError`` (it is never clamped).

    Bar roll (``index=None``): positions roll by whole bars. The shifts are scored **circularly** — a circular lag
    and a circular turnover array are precomputed once and rolled — so **total turnover is preserved exactly** across
    every shift (a rotation cannot change how many transitions a cyclic sequence has); only the price-alignment
    changes. When ``spread_frac`` is per-bar the cost still varies with alignment because a re-timed trade meets a
    different bar's spread — that is intended. ``real_sharpe`` is the strategy's true **non-circular** return (flat
    start); it differs from the circular nulls only at the single wrap bar, negligible for N ≫ 1.

    Trading-day lattice (``index`` given): ``index`` holds the bars' tz-aware timestamps on the true clock, one per
    position. Every bar is placed on (trading day, intraday slot) with the New York 17:00 roll, and positions rotate
    by whole trading days, slot by slot, so each bar keeps its time of day (the day of week is not kept). A bar whose
    source slot is empty (a short day, a holiday) is flat; many ragged days, or an index off the true clock (partial
    weekend "days"), therefore thin the null's exposure and make the test liberal. Each rotation is scored like the
    real series (``net_bar_returns``, flat start), so turnover is close to, not exactly, the real one. The minimum
    shift is ``ceil(min_offset / median bars per trading day)`` days, at least 1. Use this mode for intraday data:
    under a bar roll, time-of-day seasonality in the returns can pass for timing skill.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}.")
    if int(n_shuffle) < 1:
        raise ValueError(f"n_shuffle must be >= 1, got {n_shuffle}.")
    if int(min_offset) < 1:
        raise ValueError(f"min_offset must be >= 1 (offset 0 is the real alignment), got {min_offset}.")
    pos = _checked_positions(positions)
    r = np.asarray(bar_returns, dtype=float)
    if pos.shape != r.shape:
        raise ValueError(f"positions {pos.shape} and bar_returns {r.shape} must be the same length.")
    n = pos.size
    if n < 4:
        raise ValueError(f"circular_shift_null: need at least 4 bars, got {n}.")
    sf = _checked_spread(spread_frac, n)
    rng = np.random.default_rng(seed)
    if index is None:
        nulls = _bar_roll_nulls(pos, r, sf, int(min_offset), int(n_shuffle), rng, periods_per_year)
    else:
        nulls = _lattice_nulls(pos, r, sf, index, int(min_offset), int(n_shuffle), rng, periods_per_year)

    real = _ann_sharpe(net_bar_returns(pos, r, spread_frac=sf), periods_per_year)
    finite = nulls[np.isfinite(nulls)]
    percentile = float((finite < real).mean()) if finite.size and np.isfinite(real) else float("nan")
    beta_sharpe = float(finite.mean()) if finite.size else float("nan")
    return CircularShiftNull(real, beta_sharpe, percentile, nulls, int(n_shuffle), p_value(real, nulls),
                             int(nulls.size))


def _bar_roll_nulls(pos: NDArray, r: NDArray, sf: NDArray, min_offset: int, n_shuffle: int, rng: np.random.Generator,
                    periods_per_year: float) -> NDArray:
    """Null Sharpes for distinct whole-bar rolls, scored circularly (turnover preserved exactly)."""
    n = pos.size
    offsets = draw_offsets(n, min_offset, n_shuffle, rng)
    if offsets.size == 0:
        raise ValueError(f"circular_shift_null: {n} bars cannot honour min_offset={min_offset} (offsets must lie in "
                         f"[min_offset, n - min_offset]); pass a smaller min_offset or a longer series.")
    r_ok = np.where(np.isfinite(r), r, 0.0)                     # missing return -> 0 gross contribution
    held = np.roll(pos, 1)                                      # circular lag: held[t] = pos[t-1]
    turn = np.abs(pos - held)                                  # circular turnover (sum invariant under rotation)
    nulls = [_ann_sharpe(np.roll(held, o) * r_ok - np.roll(turn, o) * sf, periods_per_year) for o in offsets]
    return np.asarray(nulls, dtype=float)


def _lattice_nulls(pos: NDArray, r: NDArray, sf: NDArray, index: pd.DatetimeIndex, min_offset: int, n_shuffle: int,
                   rng: np.random.Generator, periods_per_year: float) -> NDArray:
    """Null Sharpes for distinct whole-trading-day rotations that keep every bar's intraday slot."""
    if len(index) != pos.size:
        raise ValueError(f"index has {len(index)} stamps but there are {pos.size} positions.")
    clock = build_clock(index, exclude_windows=())
    lattice = DayLattice.build(clock.trading_day, clock.slot, np.arange(pos.size))  # grid cell = that slot's bar
    min_days = max(1, int(np.ceil(min_offset / np.median(np.bincount(lattice.day_pos)))))
    offsets = draw_offsets(lattice.n_days, min_days, n_shuffle, rng)
    if offsets.size == 0:
        raise ValueError(f"circular_shift_null: {lattice.n_days} trading days cannot honour min_offset={min_offset} "
                         f"bars ({min_days} day(s) each way); pass a smaller min_offset or a longer series.")
    nulls = []
    for days in offsets:
        src = lattice.rotate(int(days))                         # source bar per bar, -1 where that slot is empty
        rotated = np.where(src >= 0, pos[src], 0.0)             # an empty source slot is flat
        nulls.append(_ann_sharpe(net_bar_returns(rotated, r, spread_frac=sf), periods_per_year))
    return np.asarray(nulls, dtype=float)


@dataclass(frozen=True)
class BetaTimingDecomposition:
    """Return decomposition. ``total`` is the summed GROSS (pre-cost) return ``Σ pos_{t-1}·r_t``, split into
    ``beta`` (holding the average exposure) + ``timing`` (varying exposure around it); ``beta + timing == total``
    exactly. ``timing_share = timing / total`` (NaN if total ≈ 0). ``mean_exposure`` is the average held position.
    Gross by design — cost is a separate drag, not part of the beta-vs-timing question.
    """

    total: float
    beta: float
    timing: float
    timing_share: float
    mean_exposure: float


def beta_timing_decomposition(positions: NDArray, bar_returns: NDArray) -> BetaTimingDecomposition:
    """Split realised gross return into constant-average-exposure (beta) and vary-the-exposure (timing) parts."""
    pos = _checked_positions(positions)
    r = np.asarray(bar_returns, dtype=float)
    if pos.shape != r.shape:
        raise ValueError(f"positions {pos.shape} and bar_returns {r.shape} must be the same length.")
    lag = np.concatenate(([0.0], pos[:-1]))
    m = np.isfinite(r)
    mean_pos = float(lag[m].mean()) if m.any() else 0.0
    beta = float(np.sum(mean_pos * r[m]))
    timing = float(np.sum((lag[m] - mean_pos) * r[m]))
    total = beta + timing
    timing_share = float(timing / total) if abs(total) > 1e-12 else float("nan")
    return BetaTimingDecomposition(total, beta, timing, timing_share, mean_pos)
