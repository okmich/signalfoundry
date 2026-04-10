from okmich_quant_mt5.position_manager import *
from okmich_quant_core.config import PositionManagerConfig
from . import with_strategy_config


class TestPositionManagerFactory:

    def test_create_all_point_managers(self, base_manager_kwargs):
        """Test factory creation of all point-based managers"""

        # FIXED_POINT
        config = with_strategy_config(
            PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50, tp=100)
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, FixedPointBasedPositionManager)

        # FIXED_POINT_WITH_TRAILING
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
                sl=50,
                tp=100,
                trailing=20,
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, FixedSlTpWithTrailingPointBasedPositionManager)

        # FIXED_POINT_WITH_BREAK_EVEN
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
                sl=50,
                tp=100,
                break_even=30,
                trailing=10,
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(
            manager, FixedSlTpWithSingleBreakEvenPointBasedPositionManager
        )

        # DYNAMIC_POINT
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_POINT,
                sl=50,
                trailing=15,
                break_even=30,
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, DynamicPointBasedPositionManager)

    def test_create_all_managers(self, base_manager_kwargs):
        """Test factory creation of all percent-based managers"""

        # FIXED_PERCENT
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, FixedPercentBasedPositionManager)

        # FIXED_PERCENT_WITH_TRAILING
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
                sl=1.0,
                tp=2.0,
                trailing=0.5,
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(
            manager, FixedPercentSlTpWithTrailingPercentBasedPositionManager
        )

    def test_create_all_atr_managers(self, base_manager_kwargs):
        """Test factory creation of all ATR-based managers"""

        # FIXED_ATR
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, FixedAtrBasedPositionManager)

        # DYNAMIC_ATR
        config = with_strategy_config(
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_ATR,
                sl=1.5,
                trailing=2.0,
                atr_period=14,
            )
        )
        manager = get_position_manager(config, **base_manager_kwargs)
        assert isinstance(manager, DynamicAtrBasedPositionManager)

    def test_invalid_manager_type(self, base_manager_kwargs):
        """Test factory with unsupported manager type"""
        # Create a config with an invalid type (this would need custom handling)
        pass

    def test_config_to_dict_conversion(
        self, point_config, base_manager_kwargs, mock_mt5_functions
    ):
        """Test that config is properly converted to dict for manager initialization"""
        mock_mt5_functions["modify_position"].return_value = True

        manager = get_position_manager(
            with_strategy_config(point_config), **base_manager_kwargs
        )
        assert hasattr(manager, "sl_points")
        assert manager.sl_points == 50
        assert manager.tp_points == 100
