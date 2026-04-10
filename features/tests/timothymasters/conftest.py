import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    close = np.random.randn(n).cumsum() + 100
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.abs(np.random.randn(n)) * 1000 + 1000

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }


@pytest.fixture
def linear_price_data():
    """Generate linear trending price data."""
    n = 200
    close = np.arange(1, n + 1, dtype=np.float64)
    high = close + 1.0
    low = close - 1.0
    open_ = close + 0.5
    volume = np.ones(n) * 1000

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }


@pytest.fixture
def sine_wave_data():
    """Generate sine wave price data with known period."""
    n = 500
    period = 20
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / period)
    high = close + 1.0
    low = close - 1.0
    open_ = close + 0.5
    volume = np.ones(n) * 1000

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'period': period,
    }
