"""
Tests for MaxLossStopLossPositionManager.

Verifies that the manager correctly reverse-engineers a stop loss price from a
maximum loss amount in account currency and applies it when no SL is set.
"""
import pytest
from unittest.mock import patch, MagicMock

from okmich_quant_core.config import PositionManagerConfig, StrategyConfig
from okmich_quant_mt5.position_manager import (
    MaxLossStopLossPositionManager,
    PositionManagerType,
)
from tests.position_manager import with_strategy_config


def make_strategy_config(max_loss_amount: float) -> StrategyConfig:
    return with_strategy_config(
        PositionManagerConfig(
            type=PositionManagerType.MAX_LOSS_STOP_LOSS,
            max_loss_amount=max_loss_amount,
        )
    )


def make_symbol_info(
    contract_size=100_000,
    tick_size=0.00001,
    tick_value=1.0,
    digits=5,
):
    """Build a mock MT5 symbol_info object."""
    info = MagicMock()
    info.trade_contract_size = contract_size
    info.trade_tick_size = tick_size
    info.trade_tick_value = tick_value
    info.digits = digits
    return info


class TestMaxLossStopLossPositionManager:

    def test_sets_sl_on_long_position_with_no_sl(self, mock_mt5_functions):
        """
        BUY position with no SL: SL should be placed below open_price by
        the price distance that equals max_loss_amount.
        """
        config = make_strategy_config(max_loss_amount=50.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1001,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        symbol_info = make_symbol_info(
            contract_size=100_000, tick_size=0.00001, tick_value=1.0, digits=5
        )

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager.manage_long_position(position, flag=True)

        # SL should have been set (below open price for long)
        assert mock_mt5_functions.modify_base.called
        call_kwargs = mock_mt5_functions.modify_base.call_args
        sl_set = call_kwargs[1]["sl"] if call_kwargs[1] else call_kwargs[0][1]
        assert sl_set < position["price_open"], "SL for BUY should be below open price"

    def test_sets_sl_on_short_position_with_no_sl(self, mock_mt5_functions):
        """SELL position with no SL: SL should be placed above open_price."""
        config = make_strategy_config(max_loss_amount=50.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1002,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        symbol_info = make_symbol_info()

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager.manage_short_position(position, flag=True)

        assert mock_mt5_functions.modify_base.called
        call_kwargs = mock_mt5_functions.modify_base.call_args
        sl_set = call_kwargs[1]["sl"] if call_kwargs[1] else call_kwargs[0][1]
        assert sl_set > position["price_open"], "SL for SELL should be above open price"

    def test_skips_when_sl_already_set(self, mock_mt5_functions):
        """When SL is already set, manager should not modify the position."""
        config = make_strategy_config(max_loss_amount=50.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1003,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 1.09500,  # SL already set
            "tp": 0.0,
        }

        symbol_info = make_symbol_info()

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager.manage_long_position(position, flag=True)

        mock_mt5_functions.modify_base.assert_not_called()

    def test_sl_calculation_correctness(self, mock_mt5_functions):
        """
        Verify the exact SL price calculation.

        Using simple numbers:
            max_loss = $100
            volume = 1.0 lot
            tick_value = 10.0 (i.e. $10 per tick)
            tick_size = 0.0001
            => loss_per_point = 1.0 * (10.0 / 0.0001) = 100_000 per full price unit
            => price_distance = 100 / 100_000 = 0.001
            => sl (long) = 1.10000 - 0.001 = 1.09900
        """
        config = make_strategy_config(max_loss_amount=100.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1004,
            "symbol": "EURUSD",
            "volume": 1.0,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        symbol_info = make_symbol_info(
            contract_size=100_000,
            tick_size=0.0001,
            tick_value=10.0,
            digits=5,
        )

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager.manage_long_position(position, flag=True)

        call_kwargs = mock_mt5_functions.modify_base.call_args
        sl_set = call_kwargs[1]["sl"] if call_kwargs[1] else call_kwargs[0][1]
        assert sl_set == pytest.approx(1.09900, abs=1e-5)

    def test_larger_loss_means_wider_sl(self, mock_mt5_functions):
        """Larger max_loss_amount should produce an SL further from open price."""
        symbol_info = make_symbol_info()

        position = {
            "ticket": 1005,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        # Tight risk
        config_tight = make_strategy_config(max_loss_amount=10.0)
        manager_tight = MaxLossStopLossPositionManager(config_tight)
        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager_tight.manage_long_position(position, flag=True)
        sl_tight = mock_mt5_functions.modify_base.call_args[1]["sl"]

        mock_mt5_functions.modify_base.reset_mock()
        position["sl"] = 0.0  # reset for second manager

        # Wide risk
        config_wide = make_strategy_config(max_loss_amount=100.0)
        manager_wide = MaxLossStopLossPositionManager(config_wide)
        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager_wide.manage_long_position(position, flag=True)
        sl_wide = mock_mt5_functions.modify_base.call_args[1]["sl"]

        assert sl_wide < sl_tight, "Wider risk tolerance should place SL further from open"

    def test_symbol_info_failure_skips_modification(self, mock_mt5_functions):
        """When symbol_info returns None, no modification should happen."""
        config = make_strategy_config(max_loss_amount=50.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1006,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        with patch("MetaTrader5.symbol_info", return_value=None):
            manager.manage_long_position(position, flag=True)

        mock_mt5_functions.modify_base.assert_not_called()

    def test_sl_is_rounded_to_symbol_digits(self, mock_mt5_functions):
        """SL price should be rounded to the symbol's digit precision."""
        config = make_strategy_config(max_loss_amount=33.33)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 1007,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        symbol_info = make_symbol_info(digits=5)

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            manager.manage_long_position(position, flag=True)

        call_kwargs = mock_mt5_functions.modify_base.call_args
        sl_set = call_kwargs[1]["sl"] if call_kwargs[1] else call_kwargs[0][1]
        # Verify it has at most 5 decimal places
        decimal_places = len(str(sl_set).split(".")[-1]) if "." in str(sl_set) else 0
        assert decimal_places <= 5

    def test_factory_creates_correct_manager(self):
        """Factory should return MaxLossStopLossPositionManager for MAX_LOSS_STOP_LOSS type."""
        from okmich_quant_mt5.position_manager import get_position_manager

        config = make_strategy_config(max_loss_amount=50.0)
        manager = get_position_manager(config)
        assert isinstance(manager, MaxLossStopLossPositionManager)

    def test_computed_sl_zero_or_negative_skips_modification(self, mock_mt5_functions, caplog):
        """
        When max_loss_amount is so large that the computed SL would be <= 0,
        no modification should occur and an error should be logged.
        """
        # Use an absurdly large max_loss_amount relative to instrument value
        config = make_strategy_config(max_loss_amount=99_999_999.0)
        manager = MaxLossStopLossPositionManager(config)

        position = {
            "ticket": 2001,
            "symbol": "EURUSD",
            "volume": 0.01,
            "price_open": 1.10000,
            "sl": 0.0,
            "tp": 0.0,
        }

        # With tiny tick_value the computed price_distance > price_open, so sl <= 0
        symbol_info = make_symbol_info(
            contract_size=100_000,
            tick_size=0.00001,
            tick_value=0.00001,  # extremely small tick value
            digits=5,
        )

        with patch("MetaTrader5.symbol_info", return_value=symbol_info):
            with caplog.at_level("ERROR"):
                manager.manage_long_position(position, flag=True)

        assert "Computed SL <= 0" in caplog.text
        mock_mt5_functions.modify_base.assert_not_called()