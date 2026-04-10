import logging
from datetime import datetime
from unittest.mock import patch

from okmich_quant_mt5.position_manager import get_position_manager
from okmich_quant_mt5.position_manager.base import BaseMt5PositionManager
from okmich_quant_core.config import PositionManagerConfig, PositionManagerType, StrategyConfig
from . import with_strategy_config

import MetaTrader5 as mt5


def make_concrete_manager(config: PositionManagerConfig) -> BaseMt5PositionManager:
    """Return a concrete subclass instance for testing base methods."""
    return get_position_manager(with_strategy_config(config))


class TestBaseMt5PositionManager:

    def test_init(self):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=50
        )
        manager = with_strategy_config(config)

        assert manager.symbol == "EURUSD"
        assert manager.magic == 12345
        assert manager.name == "TestSystem"

    def test_manage_positions_error_getting_positions(self, mock_mt5_functions, caplog):
        config = PositionManagerConfig(
            type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=50
        )
        manager = get_position_manager(with_strategy_config(config))

        mock_mt5_functions["get_positions"].reset_mock()
        mock_mt5_functions["get_positions"].return_value = None

        with caplog.at_level(logging.ERROR):
            result = manager.manage_positions(run_dt=datetime.now(), flag=True)
            assert result == -1
            assert "Failed to get positions" in caplog.text

    def test_manage_positions_no_positions(self, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=50
        )
        manager = get_position_manager(with_strategy_config(config))

        mock_mt5_functions["get_positions"].return_value = []
        result = manager.manage_positions(run_dt=datetime.now(), flag=True)
        assert result == 0


class TestModifyPositionConduit:
    """
    Verify that BaseMt5PositionManager.modify_position is the single conduit
    to mt5_modify_position — subclasses must never call mt5_modify_position directly.
    """

    def test_modify_position_calls_mt5_with_sl_and_tp(self, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001)
        manager = make_concrete_manager(config)
        position = {"ticket": 999, "symbol": "EURUSD", "volume": 0.1, "price_open": 1.1, "sl": 0.0, "tp": 0.0, "price_current": 1.1}

        result = manager.modify_position(position, sl=1.09, tp=1.12)

        assert result is True
        mock_mt5_functions.modify_base.assert_called_once_with(999, sl=1.09, tp=1.12)

    def test_modify_position_sl_only_passes_zero_tp(self, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001)
        manager = make_concrete_manager(config)
        position = {"ticket": 999, "symbol": "EURUSD", "volume": 0.1, "price_open": 1.1, "sl": 0.0, "tp": 0.0, "price_current": 1.1}

        result = manager.modify_position(position, sl=1.09)

        assert result is True
        mock_mt5_functions.modify_base.assert_called_once_with(999, sl=1.09, tp=0.0)

    def test_modify_position_returns_false_on_exception(self, mock_mt5_functions, caplog):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001)
        manager = make_concrete_manager(config)
        position = {"ticket": 888, "symbol": "EURUSD", "volume": 0.1, "price_open": 1.1, "sl": 0.0, "tp": 0.0, "price_current": 1.1}

        mock_mt5_functions.modify_base.side_effect = RuntimeError("MT5 connection lost")

        with caplog.at_level(logging.ERROR):
            result = manager.modify_position(position, sl=1.09)

        assert result is False
        assert "Failed to modify position" in caplog.text

    def test_modify_position_missing_ticket_returns_false(self, mock_mt5_functions, caplog):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001)
        manager = make_concrete_manager(config)
        position = {"symbol": "EURUSD", "volume": 0.1, "price_open": 1.1, "sl": 0.0, "tp": 0.0}  # no ticket

        with caplog.at_level(logging.ERROR):
            result = manager.modify_position(position, sl=1.09)

        assert result is False
        mock_mt5_functions.modify_base.assert_not_called()
