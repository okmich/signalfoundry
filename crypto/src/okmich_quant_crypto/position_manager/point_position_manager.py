"""Point-based managers: distances are ``units * point_size`` in quote price units (``point_size`` is required)."""
from typing import Optional

from okmich_quant_core import StrategyConfig

from .base import DistanceBasedPositionManager


class PointBasedPositionManager(DistanceBasedPositionManager):

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        point_size = strategy_config.position_manager.point_size
        if not point_size or point_size <= 0:
            raise ValueError(f"{self.manager_type.value} needs an explicit positive point_size in quote price units")
        self.point_size = float(point_size)

    def _distance(self, units: float, reference_price: float) -> Optional[float]:
        return units * self.point_size
