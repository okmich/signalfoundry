from okmich_quant_mt5.position_manager import PositionManagerType
from okmich_quant_mt5.position_manager import get_position_manager
from okmich_quant_core.config import PositionManagerConfig
from . import with_strategy_config


class TestIntegration:

    def test_all_managers_creation_and_basic_functionality(
        self, base_manager_kwargs, long_position_no_sl_tp, mock_mt5_functions
    ):
        """Test that all 12 managers can be created and handle basic operations"""

        configs = [
            # Point-based
            PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
                sl=50,
                tp=100,
                trailing=20,
            ),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
                sl=50,
                tp=100,
                break_even=30,
                trailing=10,
            ),
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_POINT,
                sl=50,
                trailing=15,
                break_even=30,
            ),
            # Percent-based
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0
            ),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
                sl=1.0,
                tp=2.0,
                trailing=0.5,
            ),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
                sl=1.0,
                tp=2.0,
                break_even=1.0,
                trailing=0.5,
            ),
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0, trailing=0.5
            ),
            # ATR-based
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14
            ),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
                sl=1.5,
                tp=3.0,
                trailing=2.0,
                atr_period=14,
            ),
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
                sl=1.5,
                tp=3.0,
                break_even=1.0,
                trailing=2.0,
                atr_period=14,
            ),
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_ATR,
                sl=1.5,
                trailing=2.0,
                atr_period=14,
            ),
            # Loss amount manager
            PositionManagerConfig(
                type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=100
            ),
        ]

        for i, config in enumerate(configs):
            mock_mt5_functions["modify_position"].reset_mock()
            mock_mt5_functions["modify_position"].return_value = True
            mock_mt5_functions["get_atr"].return_value = 0.0010

            manager = get_position_manager(
                with_strategy_config(config), **base_manager_kwargs
            )
            assert manager is not None

            manager.manage_long_position(long_position_no_sl_tp, flag=True)

    def test_all_managers_respect_flag_parameter(
        self,
        point_config,
        base_manager_kwargs,
        long_position_no_sl_tp,
        mock_mt5_functions,
    ):
        """Test that all managers respect the flag parameter"""
        manager = get_position_manager(
            with_strategy_config(point_config), **base_manager_kwargs
        )

        manager.manage_long_position(long_position_no_sl_tp, flag=False)
        mock_mt5_functions["modify_position"].assert_not_called()

        mock_mt5_functions["modify_position"].return_value = True
        manager.manage_long_position(long_position_no_sl_tp, flag=True)
        assert mock_mt5_functions["modify_position"].called
