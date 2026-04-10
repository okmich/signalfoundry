import pytest
from pydantic import ValidationError

from okmich_quant_core.config import PositionManagerConfig, PositionManagerType


class TestPositionManagerConfig:
    def test_valid_point_config(self):
        """Test valid point-based configuration"""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100, point_size=0.0001
        )
        assert config.type == PositionManagerType.FIXED_POINT
        assert config.sl == 50
        assert config.tp == 100
        assert config.point_size == 0.0001

    def test_valid_config(self):
        """Test valid percent-based configuration"""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT, sl=1.0, tp=2.0
        )
        assert config.type == PositionManagerType.FIXED_PERCENT
        assert config.sl == 1.0
        assert config.tp == 2.0

    def test_valid_atr_config(self):
        """Test valid ATR-based configuration"""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=14
        )
        assert config.type == PositionManagerType.FIXED_ATR
        assert config.sl == 1.5
        assert config.tp == 3.0
        assert config.atr_period == 14

    def test_invalid_point_config_missing_fields(self):
        """Test validation error for missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
                sl=50,  # Missing tp and trailing
            )
        assert "tp" in str(exc_info.value)
        assert "trailing" in str(exc_info.value)

    def test_invalid_negative_values(self):
        """Test validation error for negative values"""
        with pytest.raises(ValidationError) as exc_info:
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT, sl=-50, tp=100  # Negative value
            )
        assert "Point/Percent/Atr multiplier values must be positive" in str(
            exc_info.value
        )

    def test_invalid_atr_period(self):
        """Test validation error for invalid ATR period"""
        with pytest.raises(ValidationError) as exc_info:
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=-5
            )
        assert "ATR period must be positive" in str(exc_info.value)

    def test_invalid_max_loss_amount(self):
        with pytest.raises(ValidationError) as exc_info:
            PositionManagerConfig(
                type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=-5
            )
        assert "Max loss amount must be positive" in str(exc_info.value)

    def test_all_manager_types_validation(self):
        """Test that all manager types validate correctly with proper models_research_configs"""
        valid_configs = [
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
                type=PositionManagerType.DYNAMIC_POINT, sl=50, trailing=15
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
            PositionManagerConfig(
                type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=100
            ),
        ]

        # All should validate without errors
        for config in valid_configs:
            assert config.type is not None
