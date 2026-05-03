"""Live triple-barrier position monitor with side support.

Side semantics:
- side=+1 (long): upper barrier = profit-take above entry, lower = stop below
- side=-1 (short): pt/sl multipliers are conceptually flipped — the price barriers
  are computed by swapping pt and sl in `compute_barrier_levels` so the upper
  barrier sits above entry as the stop and the lower sits below as the target

Returned labels are bet-direction labels: +1 if the bet won, -1 if it lost,
0 on vertical expiry (unless `sign_vertical=True`). Returns are side-adjusted:
`ret = side * log(exit_price / entry_price)`.

Exit price convention (matches offline `get_labels`):
- Horizontal barrier hit -> exit price = barrier price (upper or lower).
  This avoids the wick-touch / close-far-from-barrier pathology where label
  and ret disagree in sign.
- Vertical expiry -> exit price = `price` (the close arg).

Same-bar tie-break: when high >= upper AND low <= lower in the same bar,
`same_bar_policy` decides the label (default WORST_CASE -> lower wins).

Volatility passed in must be unitless return-volatility (see barriers.py).

`on_bar` accepts an optional (high, low) pair so live callers driven by bar
data can match the offline H/L barrier-touch semantics. Tick callers pass
`high=low=close=tick_price` (the default when `high`/`low` are None).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite, log
from typing import Dict, List, Optional

from okmich_quant_features.tbm.barriers import BarrierHit, compute_barrier_levels

logger = logging.getLogger(__name__)


class BarrierTiePolicy(str, Enum):
    UPPER_FIRST = "upper_first"
    LOWER_FIRST = "lower_first"
    WORST_CASE = "worst_case"


def _tz_aware(ts) -> bool:
    return getattr(ts, "tzinfo", None) is not None


@dataclass(frozen=True)
class PositionResult:
    position_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    ret: float
    barrier_hit: BarrierHit
    label: int
    side: int


class PositionMonitor:
    def __init__(self, position_id: str, entry_price: float, entry_time: datetime, volatility: float,
                 pt_multiplier: float, sl_multiplier: float, expiry_time: datetime, side: int = 1,
                 sign_vertical: bool = False,
                 same_bar_policy: BarrierTiePolicy = BarrierTiePolicy.WORST_CASE):
        if not position_id:
            raise ValueError("position_id must be a non-empty string")
        if _tz_aware(entry_time) != _tz_aware(expiry_time):
            raise ValueError("entry_time and expiry_time must have consistent tz-awareness")
        if expiry_time <= entry_time:
            raise ValueError(f"expiry_time {expiry_time} must be > entry_time {entry_time}")
        if side not in (1, -1):
            raise ValueError(f"side must be +1 or -1, got {side}")
        if not isinstance(same_bar_policy, BarrierTiePolicy):
            same_bar_policy = BarrierTiePolicy(same_bar_policy)

        # For shorts, swap pt/sl so the upper-price barrier acts as the stop and
        # the lower-price barrier acts as the profit target.
        pt_geom, sl_geom = (pt_multiplier, sl_multiplier) if side == 1 else (sl_multiplier, pt_multiplier)
        levels = compute_barrier_levels(entry_price, volatility, pt_geom, sl_geom)

        self._position_id = position_id
        self._entry_price = entry_price
        self._entry_time = entry_time
        self._upper: Optional[float] = levels["upper"]
        self._lower: Optional[float] = levels["lower"]
        self._expiry_time = expiry_time
        self._sign_vertical = sign_vertical
        self._side = side
        self._tie_policy = same_bar_policy

    @property
    def position_id(self) -> str:
        return self._position_id

    @property
    def side(self) -> int:
        return self._side

    def on_bar(self, timestamp: datetime, price: float, high: Optional[float] = None,
               low: Optional[float] = None) -> Optional[PositionResult]:
        """Advance the monitor by one bar. `price` is the close, used as exit
        price ONLY on vertical expiry. Horizontal hits exit at the barrier price.

        If `high`/`low` are supplied they are used for the barrier-touch check —
        matching the H/L-aware offline `get_labels` semantics. Tick callers may
        omit them and they default to `price`.

        OHLC consistency is validated: `low <= price <= high`. A swapped or
        corrupt bar (common with weekend FX feeds) raises `ValueError`.
        """
        if _tz_aware(timestamp) != _tz_aware(self._entry_time):
            raise ValueError("timestamp tz-awareness must match entry_time")
        if not isfinite(price) or price <= 0:
            raise ValueError(f"price must be finite and > 0, got {price}")
        h = price if high is None else high
        l = price if low is None else low
        if not (isfinite(h) and isfinite(l)) or h <= 0 or l <= 0:
            raise ValueError(f"high/low must be finite and > 0, got high={h} low={l}")
        if l > price or h < price:
            raise ValueError(f"OHLC inconsistent: low={l}, price={price}, high={h} (require low <= price <= high)")
        if h < l:
            raise ValueError(f"OHLC inconsistent: high={h} < low={l}")

        upper_touch = self._upper is not None and h >= self._upper
        lower_touch = self._lower is not None and l <= self._lower

        if upper_touch and lower_touch:
            # Same-bar ambiguity: dispatch on policy.
            if self._tie_policy == BarrierTiePolicy.UPPER_FIRST:
                return self._build_result(timestamp, self._upper, BarrierHit.UPPER, path_dir=1)
            return self._build_result(timestamp, self._lower, BarrierHit.LOWER, path_dir=-1)
        if upper_touch:
            return self._build_result(timestamp, self._upper, BarrierHit.UPPER, path_dir=1)
        if lower_touch:
            return self._build_result(timestamp, self._lower, BarrierHit.LOWER, path_dir=-1)
        if timestamp >= self._expiry_time:
            return self._build_result(timestamp, price, BarrierHit.VERTICAL, path_dir=0)
        return None

    def _build_result(self, exit_time: datetime, exit_price: float, barrier: BarrierHit,
                      path_dir: int) -> PositionResult:
        if path_dir == 0:
            if self._sign_vertical:
                if exit_price > self._entry_price:
                    label = self._side
                elif exit_price < self._entry_price:
                    label = -self._side
                else:
                    label = 0
            else:
                label = 0
        else:
            label = self._side * path_dir

        ret = self._side * log(exit_price / self._entry_price)
        return PositionResult(
            position_id=self._position_id, entry_time=self._entry_time, exit_time=exit_time,
            entry_price=self._entry_price, exit_price=exit_price, ret=ret, barrier_hit=barrier,
            label=label, side=self._side,
        )


class PositionBook:
    def __init__(self):
        self._positions: Dict[str, PositionMonitor] = {}

    def open_position(self, monitor: PositionMonitor) -> None:
        if monitor.position_id in self._positions:
            raise ValueError(f"position_id already open: {monitor.position_id}")
        self._positions[monitor.position_id] = monitor

    def on_bar(self, timestamp: datetime, prices: Dict[str, float],
               highs: Optional[Dict[str, float]] = None,
               lows: Optional[Dict[str, float]] = None) -> List[PositionResult]:
        """Advance all open positions by one bar. Per-position errors are
        isolated: a bad price/OHLC for one position is logged and skipped,
        leaving the rest of the book to advance normally.
        """
        exits: List[PositionResult] = []
        for pid, monitor in list(self._positions.items()):
            if pid not in prices:
                logger.warning("missing price for position_id=%s; skipping bar", pid)
                continue
            high = None if highs is None else highs.get(pid)
            low = None if lows is None else lows.get(pid)
            try:
                result = monitor.on_bar(timestamp, prices[pid], high=high, low=low)
            except ValueError as exc:
                logger.warning("position_id=%s on_bar error: %s; skipping bar for this position", pid, exc)
                continue
            if result is not None:
                exits.append(result)
        for result in exits:
            del self._positions[result.position_id]
        return exits

    @property
    def open_positions(self) -> Dict[str, PositionMonitor]:
        return dict(self._positions)

    @property
    def n_open(self) -> int:
        return len(self._positions)
