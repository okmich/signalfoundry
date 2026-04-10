import logging
from unittest.mock import patch

from okmich_quant_mt5.position_manager import PositionManagerType, get_position_manager
from okmich_quant_core.config import PositionManagerConfig
from . import with_strategy_config


class TestAtrBasedManagers:
    def test_atr_sl_tp_calculation(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        with patch.object(manager, "_get_current_atr", return_value=0.0010):
            manager.manage_long_position(long_position_no_sl_tp, flag=True)

        price_open = 1.1000
        atr = 0.0010
        expected_sl = price_open - (atr * 1.5)  # 1.0985
        expected_tp = price_open + (atr * 3.0)  # 1.1030
        mock_mt5_functions["modify_position"].assert_called_once()

        call_args = mock_mt5_functions["modify_position"].call_args[1]
        assert abs(call_args["sl"] - expected_sl) < 0.0001
        assert abs(call_args["tp"] - expected_tp) < 0.0001

    def test_atr_calculation_failure(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions, caplog):
        config = PositionManagerConfig(type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        mock_mt5_functions["get_atr"].return_value = None

        with patch.object(manager, "_get_current_atr", return_value=None):
            with caplog.at_level("ERROR"):
                manager.manage_long_position(long_position_no_sl_tp, flag=True)

        assert "Could not get ATR" in caplog.text
        mock_mt5_functions["modify_position"].assert_not_called()

    def test_skips_when_sl_and_tp_already_set(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """Early-exit: both SL and TP already set — no ATR fetch, no modify call."""
        config = PositionManagerConfig(type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        mock_mt5_functions["modify_position"].assert_not_called()

    def test_atr_trailing_logic(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
            sl=1.5, tp=3.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        # Position is profitable
        long_position_with_sl_tp["price_current"] = 1.1040  # 40 pips up

        atr = 0.0008
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(long_position_with_sl_tp, flag=True)

        trailing_amount = atr * 2.0  # 0.0016
        expected_new_sl = 1.1040 - trailing_amount  # 1.1024
        mock_mt5_functions["modify_position"].assert_called_once_with(12347, sl=expected_new_sl, tp=0.0)

    def test_atr_short_trailing_logic(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
            sl=1.5, tp=3.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 1, "profit": 60.0,
            "price_open": price_open, "price_current": 1.0960,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1015, "tp": 1.0970,
        }
        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_short_position(position, flag=True)

        trailing_amount = atr * 2.0  # 0.0020
        expected_sl = 1.0960 + trailing_amount  # 1.0980
        assert expected_sl < 1.1015  # must improve
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_single_atr_fetch_per_cycle(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """ATR should be fetched exactly once per cycle — reused by initial SL/TP and trailing."""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
            sl=1.5, tp=3.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        long_position_with_sl_tp["price_current"] = 1.1040

        atr_call_count = 0

        def counting_get_atr():
            nonlocal atr_call_count
            atr_call_count += 1
            return 0.0010

        with patch.object(manager, "_get_current_atr", side_effect=counting_get_atr):
            manager.manage_long_position(long_position_with_sl_tp, flag=True)

        assert atr_call_count == 1, f"Expected 1 ATR fetch per cycle, got {atr_call_count}"


class TestAtrBreakEvenManagers:
    def test_long_break_even_activates(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 0, "profit": 100.0,
            "price_open": price_open, "price_current": 1.1020,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0985, "tp": 1.1030,
        }
        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(position, flag=True)

        # be_threshold=0.001; be_price=1.101; price_current=1.102 >= 1.101 and sl<open
        trailing_amount = atr * 2.0
        expected_sl = max(1.1020 - trailing_amount, price_open)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_short_break_even_activates(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 1, "profit": 100.0,
            "price_open": price_open, "price_current": 1.0985,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1015, "tp": 1.0970,  # sl above open
        }
        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_short_position(position, flag=True)

        # be_threshold=0.001; be_price=1.099; price_current=1.0985 <= 1.099 and sl>open
        trailing_amount = atr * 2.0
        expected_sl = min(1.0985 + trailing_amount, price_open)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_sl, tp=0.0)

    def test_break_even_not_activated_before_threshold(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 0, "profit": 5.0,
            "price_open": price_open, "price_current": 1.1005,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0985, "tp": 1.1030,
        }
        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(position, flag=True)

        # be_price=1.101; price_current=1.1005 < 1.101 → not activated
        mock_mt5_functions["modify_position"].assert_not_called()

    def test_single_atr_fetch_per_cycle_break_even(self, base_manager_kwargs, mock_mt5_functions):
        """ATR should be fetched exactly once per cycle for break-even managers too."""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        position = {
            "ticket": 99, "type": 0, "profit": 100.0,
            "price_open": price_open, "price_current": 1.1020,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.0985, "tp": 1.1030,
        }

        atr_call_count = 0

        def counting_get_atr():
            nonlocal atr_call_count
            atr_call_count += 1
            return 0.0010

        with patch.object(manager, "_get_current_atr", side_effect=counting_get_atr):
            manager.manage_long_position(position, flag=True)

        assert atr_call_count == 1, f"Expected 1 ATR fetch per cycle, got {atr_call_count}"


class TestDynamicAtrManagers:
    def test_sets_initial_sl_when_absent(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_ATR, sl=1.5, trailing=2.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(long_position_no_sl_tp, flag=True)

        price_open = 1.1000
        expected_sl = price_open - atr * 1.5
        call_args = mock_mt5_functions["modify_position"].call_args[1]
        assert abs(call_args["sl"] - expected_sl) < 1e-9

    def test_dynamic_atr_trails_when_sl_set(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_ATR, sl=1.5, trailing=2.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        price_open = 1.1000
        long_position_with_sl_tp["price_current"] = 1.1060

        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(long_position_with_sl_tp, flag=True)

        expected_sl = 1.1060 - atr * 2.0  # 1.1040
        assert expected_sl > price_open
        mock_mt5_functions["modify_position"].assert_called_once_with(long_position_with_sl_tp["ticket"], sl=expected_sl, tp=0.0)

    def test_single_atr_fetch_per_cycle_dynamic(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """DynamicAtr should also fetch ATR exactly once, passing it to the helper."""
        config = PositionManagerConfig(type=PositionManagerType.DYNAMIC_ATR, sl=1.5, trailing=2.0, atr_period=14)
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        long_position_with_sl_tp["price_current"] = 1.1060

        atr_call_count = 0

        def counting_get_atr():
            nonlocal atr_call_count
            atr_call_count += 1
            return 0.0010

        with patch.object(manager, "_get_current_atr", side_effect=counting_get_atr):
            manager.manage_long_position(long_position_with_sl_tp, flag=True)

        assert atr_call_count == 1, f"Expected 1 ATR fetch per cycle, got {atr_call_count}"
