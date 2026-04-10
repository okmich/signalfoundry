from unittest.mock import patch

import pytest

from okmich_quant_mt5.position_manager import PositionManagerType
from okmich_quant_core.config import PositionManagerConfig


# Mock MT5 functions
@pytest.fixture
def mock_mt5_functions():
    # All SL/TP modifications now route through BaseMt5PositionManager.modify_position
    # which calls base.mt5_modify_position — there is one patch point.
    with patch(
        "okmich_quant_mt5.position_manager.base.get_positions"
    ) as mock_get_positions, patch(
        "okmich_quant_mt5.position_manager.atr_based_position_manager.get_atr"
    ) as mock_get_atr, patch(
        "okmich_quant_mt5.position_manager.base.mt5_close_position"
    ) as mock_close, patch(
        "okmich_quant_mt5.position_manager.base.mt5_modify_position"
    ) as mock_modify_base:
        mock_modify_base.return_value = True

        class UnifiedMocks:
            def __init__(self):
                self.get_positions = mock_get_positions
                self.get_atr = mock_get_atr
                self.close_position = mock_close
                # Single modify conduit — all subclasses route through base.mt5_modify_position
                self.modify_base = mock_modify_base
                self.modify_position = mock_modify_base

            def __getitem__(self, key):
                if key == "modify_position":
                    return self.modify_position
                elif key == "get_positions":
                    return self.get_positions
                elif key == "get_atr":
                    return self.get_atr
                elif key == "close_position":
                    return self.close_position
                raise KeyError(key)

        yield UnifiedMocks()


# Configuration fixtures using Pydantic models
@pytest.fixture
def point_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
    )


@pytest.fixture
def point_trailing_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
        sl=50,
        tp=100,
        trailing=20,
        point_size=0.0001,
    )


@pytest.fixture
def point_break_even_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
        sl=50,
        tp=100,
        break_even=30,
        trailing=10,
        point_size=0.0001,
    )


@pytest.fixture
def point_dynamic_config():
    return PositionManagerConfig(
        type=PositionManagerType.DYNAMIC_POINT,
        sl=50,
        trailing=15,
        break_even=30,
        point_size=0.0001,
    )


@pytest.fixture
def percent_config():
    return PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0)


@pytest.fixture
def percent_trailing_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
        sl=1.0,
        tp=2.0,
        trailing=0.5,
    )


@pytest.fixture
def percent_break_even_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
        sl=1.0,
        tp=2.0,
        break_even=1.0,
        trailing=0.5,
    )


@pytest.fixture
def percent_dynamic_config():
    return PositionManagerConfig(
        type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0, trailing=0.5
    )


@pytest.fixture
def atr_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14
    )


@pytest.fixture
def atr_trailing_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
        sl=1.5,
        tp=3.0,
        trailing=2.0,
        atr_period=14,
    )


@pytest.fixture
def atr_break_even_config():
    return PositionManagerConfig(
        type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
        sl=1.5,
        tp=3.0,
        break_even=1.0,
        trailing=2.0,
        atr_period=14,
    )


@pytest.fixture
def atr_dynamic_config():
    return PositionManagerConfig(
        type=PositionManagerType.DYNAMIC_ATR, sl=1.5, trailing=2.0, atr_period=14
    )


@pytest.fixture
def max_loss_stop_loss_config():
    return PositionManagerConfig(
        type=PositionManagerType.MAX_LOSS_STOP_LOSS,
        max_loss_amount=50.0,
    )


# Mock position data fixtures
@pytest.fixture
def long_position_no_sl_tp():
    return {
        "ticket": 12345,
        "type": 0,  # LONG
        "profit": -50.0,
        "price_open": 1.1000,
        "price_current": 1.0990,
        "symbol": "EURUSD",
        "volume": 0.1,
        "sl": 0.0,
        "tp": 0.0,
    }


@pytest.fixture
def short_position_no_sl_tp():
    return {
        "ticket": 12346,
        "type": 1,  # SHORT
        "profit": -30.0,
        "price_open": 1.1000,
        "price_current": 1.1010,
        "symbol": "EURUSD",
        "volume": 0.1,
        "sl": 0.0,
        "tp": 0.0,
    }


@pytest.fixture
def long_position_with_sl_tp():
    return {
        "ticket": 12347,
        "type": 0,  # LONG
        "profit": 100.0,
        "price_open": 1.1000,
        "price_current": 1.1050,
        "symbol": "EURUSD",
        "volume": 0.1,
        "sl": 1.0950,
        "tp": 1.1100,
    }


@pytest.fixture
def base_manager_kwargs():
    return {"symbol": "EURUSD", "magic": 12345, "system_name": "TestSystem"}
