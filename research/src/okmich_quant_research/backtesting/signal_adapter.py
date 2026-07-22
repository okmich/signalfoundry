"""Adapter from a research signal function to a vectorbt portfolio.

The two conditional-performance analyzers (``RegimePerformanceAnalyzer``, ``TemporalPerformanceAnalyzer``) are dual-mode.
In *backtest* mode they consume a ``vbt.Portfolio`` / its trades directly. In *alpha-hunting* mode they accept a raw
values DataFrame plus a **signal function** that maps those values to a signed position series. This module is the
shared bridge for that second path: it turns ``signal_fn(data) -> signed positions`` into a real ``vbt.Portfolio`` so
every downstream metric (returns, Sharpe, drawdown, exposure, trades) comes from vectorbt rather than a hand-rolled
returns engine.

v1 is *sign-based*: only the sign of the position series is used (long / flat / short), matching a ``{-1, 0, +1}`` signal.
Position magnitude (a continuous ``[-1, 1]`` exposure) is intentionally ignored for now; continuous target-sizing
via ``Portfolio.from_orders`` is a deliberate future extension.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import pandas as pd
import vectorbt as vbt

SignalFn = Callable[[pd.DataFrame], pd.Series]


def positions_to_signals(positions: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Convert a signed position series into vectorbt's four boolean signal arrays.

    Returns ``(entries, exits, short_entries, short_exits)`` aligned to ``positions.index``. A bar is a long entry when
    the sign turns positive and a long exit when it leaves positive, and symmetrically for shorts; a direct ``+1 -> -1``
    flip emits a long exit and a short entry on the same bar (vectorbt treats this as a reversal).
    Only the *sign* of each value matters — NaN is flat.
    """
    pos = pd.Series(positions).fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    is_long, is_short = pos > 0, pos < 0
    was_long, was_short = prev > 0, prev < 0
    entries = is_long & ~was_long
    exits = was_long & ~is_long
    short_entries = is_short & ~was_short
    short_exits = was_short & ~is_short
    return entries, exits, short_entries, short_exits


def signal_to_portfolio(data: pd.DataFrame, signal_fn: SignalFn, *, close_col: str = "close",
                        open_col: str = "open", init_cash: float = 10_000, fees: float = 0.0,
                        slippage: float = 0.0, freq: Optional[str] = None, **kwargs) -> vbt.Portfolio:
    """Build a ``vbt.Portfolio`` from a values DataFrame and a signal function.

    ``signal_fn(data)`` must return a signed position series (``{-1, 0, +1}``; v1
    uses only the sign) indexed like ``data``. ``close_col`` supplies the price for
    P&L and ``open_col`` is used when present. Extra ``kwargs`` pass straight
    through to ``vbt.Portfolio.from_signals``.
    """
    if close_col not in data.columns:
        raise ValueError(f"data must contain a '{close_col}' column for P&L; got {list(data.columns)}.")
    positions = signal_fn(data)
    if not isinstance(positions, pd.Series):
        raise TypeError(f"signal_fn must return a pandas Series of signed positions, got {type(positions)!r}.")
    if positions.index.intersection(data.index).empty:
        raise ValueError(
            "signal_fn returned a Series whose index does not overlap data.index; return positions indexed like "
            "`data` (e.g. pd.Series(arr, index=data.index)). A bare pd.Series(np.where(...)) carries a default "
            "RangeIndex and would silently produce zero trades."
        )
    positions = positions.reindex(data.index)
    entries, exits, short_entries, short_exits = positions_to_signals(positions)
    open_ = data[open_col] if open_col in data.columns else None
    return vbt.Portfolio.from_signals(close=data[close_col], open=open_, entries=entries, exits=exits,
                                      short_entries=short_entries, short_exits=short_exits, init_cash=init_cash,
                                      fees=fees, slippage=slippage, freq=freq, **kwargs)
