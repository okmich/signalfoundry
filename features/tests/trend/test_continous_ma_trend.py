import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import continuous_ma_trend_labeling


@pytest.fixture
def sample_price_series():
    """Generate a simple synthetic price series."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = np.linspace(100, 110, 50) + np.random.normal(0, 0.5, 50)
    return pd.Series(prices, index=dates)


@pytest.fixture
def flat_price_series():
    """Generate a flat price series."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = np.full(50, 100.0)
    return pd.Series(prices, index=dates)


def test_continuous_ma_trend_labeling_basic(sample_price_series):
    """Test basic functionality of continuous_ma_trend_labeling."""
    omega = 0.02
    labels = continuous_ma_trend_labeling(sample_price_series, omega=omega)

    assert isinstance(labels, pd.Series)
    assert labels.index.equals(sample_price_series.index)
    assert set(labels.dropna().unique()).issubset({-1, 0, 1})


def test_continuous_ma_trend_labeling_no_omega():
    """Test that ValueError is raised if omega is None."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = np.linspace(100, 110, 10)
    series = pd.Series(prices, index=dates)

    with pytest.raises(ValueError, match="Omega must be provided"):
        continuous_ma_trend_labeling(series, omega=None)


def test_continuous_ma_trend_labeling_constant_series(flat_price_series):
    """Test behavior with constant price (should result in neutral trend)."""
    omega = 0.01
    labels = continuous_ma_trend_labeling(flat_price_series, omega=omega)

    # First few entries should be NaN due to windowing
    min_periods = 20 + 5 - 1
    assert labels.iloc[:min_periods].isna().all()

    # Remaining should be neutral (0)
    assert (labels.iloc[min_periods:] == 0).all()


def test_continuous_ma_trend_labeling_uptrend():
    """Test clear uptrend case."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = np.linspace(100, 150, 50)  # Strong uptrend
    series = pd.Series(prices, index=dates)

    omega = 0.01
    labels = continuous_ma_trend_labeling(series, omega=omega)

    # First few entries should be NaN due to windowing
    min_periods = 20 + 5 - 1
    assert labels.iloc[:min_periods].isna().all()

    # Remaining should mostly be uptrend (1)
    assert (labels.iloc[min_periods:] == 1).mean() > 0.9


def test_continuous_ma_trend_labeling_downtrend():
    """Test clear downtrend case."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = np.linspace(150, 100, 50)  # Strong downtrend
    series = pd.Series(prices, index=dates)

    omega = 0.01
    labels = continuous_ma_trend_labeling(series, omega=omega)

    # First few entries should be NaN due to windowing
    min_periods = 20 + 5 - 1
    assert labels.iloc[:min_periods].isna().all()

    # Remaining should mostly be downtrend (-1)
    assert (labels.iloc[min_periods:] == -1).mean() > 0.9


def test_continuous_ma_trend_labeling_edge_case_small_series():
    """Test behavior with a series smaller than window sizes."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = [100, 101, 102, 103, 104]
    series = pd.Series(prices, index=dates)

    omega = 0.01
    labels = continuous_ma_trend_labeling(
        series, omega=omega, trend_window=10, smooth_window=3
    )

    # All should be NaN because not enough data
    assert labels.isna().all()


def test_continuous_ma_trend_labeling_custom_window_sizes():
    """Test with custom trend and smooth window sizes."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    prices = np.sin(np.linspace(0, 3 * np.pi, 30)) * 10 + 100  # Oscillating prices
    series = pd.Series(prices, index=dates)

    omega = 0.02
    labels = continuous_ma_trend_labeling(
        series, omega=omega, trend_window=10, smooth_window=3
    )

    # Check that trend are generated and contain all categories
    unique_labels = set(labels.dropna().unique())
    assert unique_labels.issubset({-1, 0, 1})


def test_continuous_ma_trend_labeling_empty_series():
    """Test behavior with empty series."""
    series = pd.Series([], dtype=float)
    omega = 0.01

    labels = continuous_ma_trend_labeling(series, omega=omega)

    assert len(labels) == 0
