"""Incremental EWMA volatility estimator for live execution.

Updates in O(1) with zero allocations per call. Internal state is three scalars
only — no list, deque, or array accumulation.
"""

from math import isfinite, log, sqrt
from typing import Optional


class EWMAVolatilityEstimator:
    """Streaming EWM standard-deviation of log returns.

    The live default warm-up (`span // 2`) is intentionally more aggressive than
    the offline `get_daily_vol` (~`span` bars). The two will produce slightly
    different estimates for the same data — expected, must be accounted for in
    any live-vs-backtest reconciliation.
    """

    def __init__(self, span: int = 100, warm_up_bars: Optional[int] = None):
        if span < 2:
            raise ValueError(f"span must be >= 2, got {span}")
        if warm_up_bars is not None and warm_up_bars < 0:
            raise ValueError(f"warm_up_bars must be >= 0, got {warm_up_bars}")
        self._alpha = 2.0 / (span + 1.0)
        self._warm_up = span // 2 if warm_up_bars is None else warm_up_bars
        self._prev_price: Optional[float] = None
        self._ewma_var: float = 0.0
        self._n_obs: int = 0

    def update(self, price: float) -> Optional[float]:
        if not isfinite(price) or price <= 0:
            raise ValueError(f"price must be finite and > 0, got {price}")
        if self._prev_price is None:
            self._prev_price = price
            self._n_obs = 1
            return None

        ret = log(price / self._prev_price)
        self._ewma_var = self._alpha * ret * ret + (1.0 - self._alpha) * self._ewma_var
        self._prev_price = price
        self._n_obs += 1

        if self._n_obs <= self._warm_up:
            return None
        return sqrt(self._ewma_var)

    @property
    def current_vol(self) -> Optional[float]:
        if self._n_obs <= self._warm_up:
            return None
        return sqrt(self._ewma_var)

    def reset(self) -> None:
        self._prev_price = None
        self._ewma_var = 0.0
        self._n_obs = 0
