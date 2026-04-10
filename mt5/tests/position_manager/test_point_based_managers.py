import logging

from okmich_quant_mt5.position_manager import PositionManagerType, get_position_manager
from okmich_quant_core.config import PositionManagerConfig
from . import with_strategy_config


class TestFixedPointBasedPositionManager:

    def test_manage_long_position_sets_sl_tp(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_no_sl_tp, flag=True)

        expected_sl = 1.1000 - (50 * 0.0001)  # 1.0950
        expected_tp = 1.1000 + (100 * 0.0001)  # 1.1100
        mock_mt5_functions["modify_position"].assert_called_once_with(12345, sl=expected_sl, tp=expected_tp)

    def test_manage_short_position_sets_sl_tp(self, base_manager_kwargs, short_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_short_position(short_position_no_sl_tp, flag=True)

        expected_sl = 1.1000 + (50 * 0.0001)  # 1.1050
        expected_tp = 1.1000 - (100 * 0.0001)  # 1.0900
        mock_mt5_functions["modify_position"].assert_called_once_with(12346, sl=expected_sl, tp=expected_tp)

    def test_manage_position_no_flag(self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_no_sl_tp, flag=False)

        mock_mt5_functions["modify_position"].assert_not_called()

    def test_skips_when_sl_and_tp_already_set(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        """Early-exit: both SL and TP already set — no modify_position call."""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        mock_mt5_functions["modify_position"].assert_not_called()

    def test_sl_computed_negative_not_set(self, base_manager_kwargs, mock_mt5_functions, caplog):
        """Silent SL ≤ 0 detection: SL points so large it would go negative."""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=200000, tp=0, point_size=0.0001
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        position = {
            "ticket": 9, "type": 0, "profit": 0.0,
            "price_open": 1.1000, "price_current": 1.1000,
            "symbol": "EURUSD", "volume": 0.1, "sl": 0.0, "tp": 0.0,
        }

        with caplog.at_level(logging.ERROR):
            manager.manage_long_position(position, flag=True)

        assert "SL will not be set" in caplog.text
        mock_mt5_functions["modify_position"].assert_not_called()


class TestFixedSlTpWithTrailingPointBasedPositionManager:
    def test_trailing_profitable_long(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        # Position is profitable (current > open), should trail
        long_position_with_sl_tp["price_current"] = 1.1080  # More profitable

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        # Should trail SL to better level; tp=0.0 because only SL is modified here
        expected_new_sl = 1.1080 - (20 * 0.0001)  # 1.1060
        mock_mt5_functions["modify_position"].assert_called_once_with(12347, sl=expected_new_sl, tp=0.0)


class TestFixedSlTpWithSingleBreakEvenPointBasedPositionManager:
    def test_break_even_logic(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
            sl=50, tp=100, break_even=30, trailing=10, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        # Position has moved 50 points (past 30 point break-even threshold)
        long_position_with_sl_tp["price_current"] = 1.1050  # 50 points up

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        # Should move SL to break-even + trailing
        expected_new_sl = 1.1050 - (10 * 0.0001)  # 1.1040
        mock_mt5_functions["modify_position"].assert_called_once_with(12347, sl=expected_new_sl, tp=0.0)


class TestFixedSlTpWithTrailingPointBasedPositionManager_Short:
    def test_trailing_profitable_short(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        # Short with existing SL above open — profitable and below open
        position = {
            "ticket": 99, "type": 1, "profit": 50.0,
            "price_open": 1.1000, "price_current": 1.0950,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1050, "tp": 1.0900,
        }

        manager.manage_short_position(position, flag=True)

        # new_sl = 1.0950 + 20*0.0001 = 1.0970 < 1.1050 → should trail
        expected_new_sl = 1.0950 + (20 * 0.0001)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_new_sl, tp=0.0)

    def test_no_trailing_when_short_not_profitable(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        position = {
            "ticket": 99, "type": 1, "profit": -10.0,
            "price_open": 1.1000, "price_current": 1.1010,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1050, "tp": 1.0900,
        }

        manager.manage_short_position(position, flag=True)

        mock_mt5_functions["modify_position"].assert_not_called()


class TestFixedSlTpWithSingleBreakEvenPointBasedPositionManager_Short:
    def test_short_break_even_when_price_past_threshold(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
            sl=50, tp=100, break_even=30, trailing=10, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        # Short SL is above open (as expected before break-even)
        position = {
            "ticket": 99, "type": 1, "profit": 80.0,
            "price_open": 1.1000, "price_current": 1.0960,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1050, "tp": 1.0900,
        }

        manager.manage_short_position(position, flag=True)

        # break_even_price = 1.1000 - 0.0030 = 1.0970
        # price_current=1.0960 <= 1.0970 and sl=1.1050 > price_open=1.1000 → activate
        # new_sl = 1.0960 + 0.0010 = 1.0970; min(1.0970, 1.1000) = 1.0970
        expected_new_sl = min(1.0960 + (10 * 0.0001), 1.1000)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_new_sl, tp=0.0)


class TestDynamicPointBasedPositionManager:
    def test_dynamic_continuous_trailing(self, base_manager_kwargs, long_position_with_sl_tp, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.DYNAMIC_POINT,
            sl=50, trailing=15, break_even=30, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)

        # Position is profitable, should trail continuously
        long_position_with_sl_tp["price_current"] = 1.1060  # 60 points up

        manager.manage_long_position(long_position_with_sl_tp, flag=True)

        # Should trail SL to better level
        expected_new_sl = 1.1060 - (15 * 0.0001)  # 1.1045
        mock_mt5_functions["modify_position"].assert_called_once_with(12347, sl=expected_new_sl, tp=0.0)

    def test_dynamic_short_trailing(self, base_manager_kwargs, mock_mt5_functions):
        config = PositionManagerConfig(
            type=PositionManagerType.DYNAMIC_POINT,
            sl=50, trailing=15, point_size=0.0001,
        )
        manager = get_position_manager(with_strategy_config(config), **base_manager_kwargs)
        position = {
            "ticket": 99, "type": 1, "profit": 40.0,
            "price_open": 1.1000, "price_current": 1.0940,
            "symbol": "EURUSD", "volume": 0.1,
            "sl": 1.1050, "tp": 1.0900,
        }

        manager.manage_short_position(position, flag=True)

        # new_sl = 1.0940 + 0.0015 = 1.0955; new_sl < sl (1.0955<1.1050) and new_sl < open → trail
        expected_new_sl = 1.0940 + (15 * 0.0001)
        mock_mt5_functions["modify_position"].assert_called_once_with(99, sl=expected_new_sl, tp=0.0)
