import logging

from okmich_quant_mt5.position_manager import PositionManagerType, get_position_manager
from okmich_quant_core.config import PositionManagerConfig
from . import with_strategy_config


class TestPercentBasedManagers:
    def test_fixed_sl_tp_calculation(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_no_sl_tp, flag=True)

        # Verify percentage-based SL/TP calculation
        price_open = 1.1000
        expected_sl = price_open * (1 - 1.0 / 100)  # 1.1000 * 0.99 = 1.0890
        expected_tp = price_open * (1 + 2.0 / 100)  # 1.1000 * 1.02 = 1.1220
        mock_mt5_functions["modify_position"].assert_called_once()

        call_args = mock_mt5_functions["modify_position"].call_args[1]
        assert abs(call_args["sl"] - expected_sl) < 0.0001
        assert abs(call_args["tp"] - expected_tp) < 0.0001

    def test_short_position_calculation(self, base_manager_kwargs, short_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_short_position(short_position_no_sl_tp, flag=True)

        # For short: SL goes up, TP goes down
        price_open = 1.1000
        expected_sl = price_open * (1 + 1.0 / 100)  # 1.1000 * 1.01 = 1.1110
        expected_tp = price_open * (1 - 2.0 / 100)  # 1.1000 * 0.98 = 1.0780
        mock_mt5_functions["modify_position"].assert_called_once()

        call_args = mock_mt5_functions["modify_position"].call_args[1]
        assert abs(call_args["sl"] - expected_sl) < 0.0001
        assert abs(call_args["tp"] - expected_tp) < 0.0001

    def test_skips_when_sl_and_tp_already_set(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """Early-exit: both SL and TP already set — no modify_position call."""
        config = PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        mock_mt5_functions["modify_position"].assert_not_called()


class TestPercentTrailingManagers:
    def test_long_trailing(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
            sl=1.0, tp=2.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_current = 1.1060
        position = {
            "ticket": 99, "type": 0, "profit": 100.0,
            "price_open": 1.1000, "price_current": price_current,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0950, "tp": 1.1220,
        }

        manager.manage_long_position(position, flag=True)

        # Trailing is anchored to price_current, not price_open
        trailing_amount = price_current * (0.5 / 100)
        expected_sl = price_current - trailing_amount
        assert expected_sl > 1.0950  # must improve SL
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_short_trailing(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
            sl=1.0, tp=2.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_current = 1.0940
        position = {
            "ticket": 99, "type": 1, "profit": 80.0,
            "price_open": 1.1000, "price_current": price_current,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1110, "tp": 1.0780,
        }

        manager.manage_short_position(position, flag=True)

        # Trailing is anchored to price_current
        trailing_amount = price_current * (0.5 / 100)
        expected_sl = price_current + trailing_amount
        assert expected_sl < 1.1110  # must improve SL for short
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)


class TestPercentBreakEvenManagers:
    def test_long_break_even_activates(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
            sl=1.0, tp=2.0, break_even=1.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        price_current = 1.1120
        position = {
            "ticket": 99, "type": 0, "profit": 100.0,
            "price_open": price_open, "price_current": price_current,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0890, "tp": 1.1220,  # sl below open
        }

        manager.manage_long_position(position, flag=True)

        # break_even_threshold = 1.1000 * 0.01 = 0.011; be_price = 1.111
        # price_current=1.112 >= 1.111 and sl=1.089 < open=1.1 → activate
        # trailing is anchored to price_current
        trailing_amount = price_current * (0.5 / 100)
        expected_sl = max(price_current - trailing_amount, price_open)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_short_break_even_activates(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
            sl=1.0, tp=2.0, break_even=1.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        price_current = 1.0880
        position = {
            "ticket": 99, "type": 1, "profit": 100.0,
            "price_open": price_open, "price_current": price_current,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1110, "tp": 1.0780,  # sl above open
        }

        manager.manage_short_position(position, flag=True)

        # break_even_threshold = 0.011; be_price = 1.089
        # price_current=1.088 <= 1.089 and sl=1.111 > open=1.1 → activate
        # trailing is anchored to price_current
        trailing_amount = price_current * (0.5 / 100)
        expected_sl = min(price_current + trailing_amount, price_open)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_long_break_even_not_activated_before_threshold(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
            sl=1.0, tp=2.0, break_even=1.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 0, "profit": 20.0,
            "price_open": price_open, "price_current": 1.1050,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0890, "tp": 1.1220,
        }

        manager.manage_long_position(position, flag=True)

        # be_price = 1.111; price_current=1.105 < 1.111 → not activated
        mock_mt5_functions["modify_position"].assert_not_called()


class TestDynamicPercentManagers:
    def test_sets_initial_sl_when_absent(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0, trailing=0.5)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_no_sl_tp, flag=True)

        price_open = 1.1000
        expected_sl = price_open * (1 - 1.0 / 100)
        call_args = mock_mt5_functions["modify_position"].call_args[1]
        assert abs(call_args["sl"] - expected_sl) < 1e-9

    def test_trails_when_sl_already_set(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0, trailing=0.5)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        price_current = 1.1070
        position = {
            "ticket": 99, "type": 0, "profit": 80.0,
            "price_open": price_open, "price_current": price_current,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0890, "tp": 0.0,
        }

        manager.manage_long_position(position, flag=True)

        # Trailing is anchored to price_current
        trailing_amount = price_current * (0.5 / 100)
        expected_sl = price_current - trailing_amount
        assert expected_sl > price_open  # guard in dynamic manager
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_skips_when_sl_and_tp_already_set(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """Early-exit: both SL and TP already set and tp_percent == 0."""
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0, trailing=0.5)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        # price_current not past price_open, so no trailing either
        long_position_with_sl_tp["price_current"] = 1.0980  # below open, no trailing

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        mock_mt5_functions["modify_position"].assert_not_called()
