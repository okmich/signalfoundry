import pytest
from pydantic import ValidationError

from okmich_quant_core.config import PositionManagerConfig, PositionManagerType


class TestPointConfigValidation:
    """PositionManagerConfig required-field validation for point-based managers."""

    def test_fixed_point_missing_sl_raises(self):
        with pytest.raises(ValidationError, match="sl"):
            PositionManagerConfig(type=PositionManagerType.FIXED_POINT, tp=100)

    def test_fixed_point_missing_tp_raises(self):
        with pytest.raises(ValidationError, match="tp"):
            PositionManagerConfig(type=PositionManagerType.FIXED_POINT, sl=50)

    def test_fixed_point_with_trailing_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_TRAILING, sl=50, tp=100
            )

    def test_fixed_point_with_break_even_missing_break_even_raises(self):
        with pytest.raises(ValidationError, match="break_even"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
                sl=50,
                tp=100,
                trailing=10,
            )

    def test_fixed_point_with_break_even_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
                sl=50,
                tp=100,
                break_even=30,
            )

    def test_dynamic_point_missing_sl_raises(self):
        with pytest.raises(ValidationError, match="sl"):
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_POINT, trailing=15
            )

    def test_dynamic_point_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(type=PositionManagerType.DYNAMIC_POINT, sl=50)

    def test_valid_fixed_point_config(self):
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT, sl=50, tp=100
        )
        assert config.sl == 50
        assert config.tp == 100

    def test_valid_dynamic_point_config_without_break_even(self):
        """break_even is optional for DYNAMIC_POINT."""
        config = PositionManagerConfig(
            type=PositionManagerType.DYNAMIC_POINT, sl=50, trailing=15
        )
        assert config.sl == 50
        assert config.trailing == 15
        assert config.break_even is None


class TestPercentConfigValidation:
    """PositionManagerConfig required-field validation for percent-based managers."""

    def test_fixed_percent_missing_sl_raises(self):
        with pytest.raises(ValidationError, match="sl"):
            PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, tp=2.0)

    def test_fixed_percent_missing_tp_raises(self):
        with pytest.raises(ValidationError, match="tp"):
            PositionManagerConfig(type=PositionManagerType.FIXED_PERCENT, sl=1.0)

    def test_fixed_percent_with_trailing_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING, sl=1.0, tp=2.0
            )

    def test_fixed_percent_with_break_even_missing_break_even_raises(self):
        with pytest.raises(ValidationError, match="break_even"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
                sl=1.0,
                tp=2.0,
                trailing=0.5,
            )

    def test_fixed_percent_with_break_even_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
                sl=1.0,
                tp=2.0,
                break_even=1.0,
            )

    def test_dynamic_percent_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(type=PositionManagerType.DYNAMIC_PERCENT, sl=1.0)


class TestAtrConfigValidation:
    """PositionManagerConfig required-field validation for ATR-based managers."""

    def test_fixed_atr_missing_atr_period_raises(self):
        with pytest.raises(ValidationError, match="atr_period"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0
            )

    def test_fixed_atr_missing_sl_raises(self):
        with pytest.raises(ValidationError, match="sl"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, tp=3.0, atr_period=14
            )

    def test_fixed_atr_missing_tp_raises(self):
        with pytest.raises(ValidationError, match="tp"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, atr_period=14
            )

    def test_fixed_atr_with_trailing_missing_atr_period_raises(self):
        """Critical fix: FIXED_ATR_WITH_TRAILING must require atr_period."""
        with pytest.raises(ValidationError, match="atr_period"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
                sl=1.5,
                tp=3.0,
                trailing=2.0,
            )

    def test_fixed_atr_with_trailing_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
                sl=1.5,
                tp=3.0,
                atr_period=14,
            )

    def test_fixed_atr_with_break_even_missing_atr_period_raises(self):
        with pytest.raises(ValidationError, match="atr_period"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
                sl=1.5,
                tp=3.0,
                break_even=1.0,
                trailing=2.0,
            )

    def test_fixed_atr_with_break_even_missing_break_even_raises(self):
        with pytest.raises(ValidationError, match="break_even"):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
                sl=1.5,
                tp=3.0,
                trailing=2.0,
                atr_period=14,
            )

    def test_dynamic_atr_missing_atr_period_raises(self):
        with pytest.raises(ValidationError, match="atr_period"):
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_ATR, sl=1.5, trailing=2.0
            )

    def test_dynamic_atr_missing_sl_raises(self):
        with pytest.raises(ValidationError, match="sl"):
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_ATR, trailing=2.0, atr_period=14
            )

    def test_dynamic_atr_missing_trailing_raises(self):
        with pytest.raises(ValidationError, match="trailing"):
            PositionManagerConfig(
                type=PositionManagerType.DYNAMIC_ATR, sl=1.5, atr_period=14
            )

    def test_valid_fixed_atr_with_trailing(self):
        """All four required fields present — must not raise."""
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
            sl=1.5,
            tp=3.0,
            trailing=2.0,
            atr_period=14,
        )
        assert config.atr_period == 14
        assert config.trailing == 2.0


class TestMaxLossConfigValidation:
    def test_max_loss_amount_missing_max_loss_amount_raises(self):
        with pytest.raises(ValidationError, match="max_loss_amount"):
            PositionManagerConfig(type=PositionManagerType.MAX_LOSS_AMOUNT)

    def test_max_loss_stop_loss_missing_max_loss_amount_raises(self):
        with pytest.raises(ValidationError, match="max_loss_amount"):
            PositionManagerConfig(type=PositionManagerType.MAX_LOSS_STOP_LOSS)

    def test_negative_sl_raises(self):
        with pytest.raises(ValidationError):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_POINT, sl=-50, tp=100
            )

    def test_zero_atr_period_raises(self):
        with pytest.raises(ValidationError):
            PositionManagerConfig(
                type=PositionManagerType.FIXED_ATR, sl=1.5, tp=3.0, atr_period=0
            )

    def test_negative_max_loss_amount_raises(self):
        with pytest.raises(ValidationError):
            PositionManagerConfig(
                type=PositionManagerType.MAX_LOSS_AMOUNT, max_loss_amount=-100
            )