"""Percent-based managers: distances are ``units`` percent of a reference price (entry for SL/TP and the break-even
threshold, current price for trailing - as in the MT5 managers)."""
from typing import Optional

from .base import DistanceBasedPositionManager


class PercentBasedPositionManager(DistanceBasedPositionManager):

    def _distance(self, units: float, reference_price: float) -> Optional[float]:
        return reference_price * units / 100.0
