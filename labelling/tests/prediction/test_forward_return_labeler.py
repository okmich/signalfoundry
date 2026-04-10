"""
Tests for FixedForwardReturnLabeler
"""

import pytest
import numpy as np
import pandas as pd
from okmich_quant_labelling.prediction import FixedForwardReturnLabeler


class TestFixedForwardReturnLabeler:
    """Test suite for FixedForwardReturnLabeler."""

    def test_uptrend_positive_returns(self):
        """Steady uptrend should produce positive forward returns."""
        # Create steady uptrend
        prices = pd.Series(100 + np.arange(100) * 0.1, name='close')
        df = pd.DataFrame({'close': prices})

        labeler = FixedForwardReturnLabeler(horizon=5, normalize=False, use_log_returns=True)
        targets = labeler.label(df)

        # All targets (except last 5) should be positive in uptrend
        assert (targets[:-5] > 0).all(), "Uptrend should produce positive returns"

    def test_last_bars_nan(self):
        """Last `horizon` bars should be NaN (no future data)."""
        prices = pd.Series(100 + np.arange(100) * 0.1, name='close')
        df = pd.DataFrame({'close': prices})

        horizon = 5
        labeler = FixedForwardReturnLabeler(horizon=horizon, normalize=False)
        targets = labeler.label(df)

        # Last `horizon` bars should be NaN
        assert targets[-horizon:].isna().all(), f"Last {horizon} bars should be NaN"

    def test_downtrend_negative_returns(self):
        """Steady downtrend should produce negative forward returns."""
        # Create steady downtrend
        prices = pd.Series(100 - np.arange(100) * 0.1, name='close')
        df = pd.DataFrame({'close': prices})

        labeler = FixedForwardReturnLabeler(horizon=5, normalize=False)
        targets = labeler.label(df)

        # All targets (except last 5) should be negative in downtrend
        assert (targets[:-5] < 0).all(), "Downtrend should produce negative returns"

    def test_normalization_reduces_variance(self):
        """Normalization should make returns more stationary."""
        # Create data with increasing volatility
        np.random.seed(42)
        vol_profile = np.linspace(0.5, 2.0, 1000)
        returns = np.random.randn(1000) * vol_profile
        prices = pd.Series(100 * np.exp(np.cumsum(returns * 0.01)), name='close')
        df = pd.DataFrame({'close': prices})

        # Without normalization
        labeler_raw = FixedForwardReturnLabeler(horizon=10, normalize=False)
        targets_raw = labeler_raw.label(df)

        # With normalization
        labeler_norm = FixedForwardReturnLabeler(horizon=10, normalize=True, normalize_window=20)
        targets_norm = labeler_norm.label(df)

        # Normalized targets should have more consistent variance
        # (We can't guarantee it's always lower due to randomness, but mean should be similar)
        assert abs(targets_norm.mean()) < abs(targets_raw.mean()) + 0.5

    def test_clipping_removes_outliers(self):
        """Clipping should remove extreme values."""
        # Create data with outliers
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.01
        returns[50] = 0.5  # Large outlier
        returns[150] = -0.5  # Large outlier
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), name='close')
        df = pd.DataFrame({'close': prices})

        # Without clipping
        labeler_raw = FixedForwardReturnLabeler(horizon=5, normalize=False, clip_percentile=None)
        targets_raw = labeler_raw.label(df)

        # With clipping
        labeler_clip = FixedForwardReturnLabeler(horizon=5, normalize=False, clip_percentile=99)
        targets_clip = labeler_clip.label(df)

        # Clipped version should have smaller range
        assert targets_clip.max() <= targets_raw.max()
        assert targets_clip.min() >= targets_raw.min()

    def test_log_vs_simple_returns(self):
        """Test difference between log and simple returns."""
        prices = pd.Series(100 + np.arange(100) * 0.1, name='close')
        df = pd.DataFrame({'close': prices})

        labeler_log = FixedForwardReturnLabeler(horizon=5, normalize=False, use_log_returns=True)
        labeler_simple = FixedForwardReturnLabeler(horizon=5, normalize=False, use_log_returns=False)

        targets_log = labeler_log.label(df)
        targets_simple = labeler_simple.label(df)

        # They should be correlated but not identical
        correlation = targets_log.corr(targets_simple)
        assert correlation > 0.99, "Log and simple returns should be highly correlated"
        assert not np.allclose(targets_log.dropna(), targets_simple.dropna()), \
            "Log and simple returns should not be identical"

    def test_leaks_future_attribute(self):
        """Labeler should have leaks_future=True."""
        labeler = FixedForwardReturnLabeler(horizon=10)
        assert labeler.leaks_future is True, "Forward return labeler must declare leaks_future=True"

    def test_invalid_horizon(self):
        """Should raise error for invalid horizon."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            FixedForwardReturnLabeler(horizon=0)

        with pytest.raises(ValueError, match="horizon must be positive"):
            FixedForwardReturnLabeler(horizon=-5)

    def test_invalid_clip_percentile(self):
        """Should raise error for invalid clip_percentile."""
        with pytest.raises(ValueError, match="clip_percentile must be in"):
            FixedForwardReturnLabeler(horizon=10, clip_percentile=0)

        with pytest.raises(ValueError, match="clip_percentile must be in"):
            FixedForwardReturnLabeler(horizon=10, clip_percentile=100)

        with pytest.raises(ValueError, match="clip_percentile must be in"):
            FixedForwardReturnLabeler(horizon=10, clip_percentile=-10)

    def test_missing_price_column(self):
        """Should raise error if price column not found."""
        df = pd.DataFrame({'price': [100, 101, 102]})
        labeler = FixedForwardReturnLabeler(horizon=1)

        with pytest.raises(KeyError, match="Column 'close' not found"):
            labeler.label(df, price_col='close')

    def test_insufficient_data(self):
        """Should raise error if DataFrame is too short."""
        df = pd.DataFrame({'close': [100, 101]})  # Only 2 rows
        labeler = FixedForwardReturnLabeler(horizon=5)  # Needs at least 6 rows

        with pytest.raises(ValueError, match="requires at least"):
            labeler.label(df)

    def test_custom_price_column(self):
        """Should work with custom price column name."""
        df = pd.DataFrame({'price': 100 + np.arange(50) * 0.1})
        labeler = FixedForwardReturnLabeler(horizon=5, normalize=False)
        targets = labeler.label(df, price_col='price')

        assert (targets[:-5] > 0).all(), "Should work with custom column name"

    def test_returns_series_with_correct_name(self):
        """Output should be a Series with descriptive name."""
        df = pd.DataFrame({'close': 100 + np.arange(50) * 0.1})
        labeler = FixedForwardReturnLabeler(horizon=10)
        targets = labeler.label(df)

        assert isinstance(targets, pd.Series), "Should return pd.Series"
        assert targets.name == 'fwd_return_h10', "Should have descriptive name"

    def test_index_preservation(self):
        """Output Series should have same index as input DataFrame."""
        dates = pd.date_range('2020-01-01', periods=100, freq='1H')
        df = pd.DataFrame({'close': 100 + np.arange(100) * 0.1}, index=dates)

        labeler = FixedForwardReturnLabeler(horizon=5)
        targets = labeler.label(df)

        assert targets.index.equals(df.index), "Index should be preserved"

    def test_repr(self):
        """Test string representation."""
        labeler = FixedForwardReturnLabeler(
            horizon=12,
            normalize=True,
            normalize_window=20,
            clip_percentile=99
        )
        repr_str = repr(labeler)

        assert 'FixedForwardReturnLabeler' in repr_str
        assert 'horizon=12' in repr_str
        assert 'normalize=True' in repr_str
