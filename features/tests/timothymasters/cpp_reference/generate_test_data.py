"""
Generate test data in C++ SINGLE.exe format for reference comparison.

Creates market files in format: YYYYMMDD Open High Low Close Volume
"""

import numpy as np
from pathlib import Path


def generate_market_file(close, high, low, open_, volume, filename):
    """
    Convert numpy arrays to C++ market file format.

    Format: YYYYMMDD Open High Low Close Volume
    """
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        for i in range(len(close)):
            date = 20200101 + i  # Sequential dates starting 2020-01-01
            f.write(f"{date} {open_[i]:.6f} {high[i]:.6f} {low[i]:.6f} {close[i]:.6f} {volume[i]:.0f}\n")

    print(f"Created {filepath} with {len(close)} bars")
    return filepath


def create_random_walk_data(n=200, seed=42):
    """Random walk with realistic OHLCV."""
    np.random.seed(seed)
    close = np.random.randn(n).cumsum() + 100
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.abs(np.random.randn(n)) * 1000 + 1000
    return close, high, low, open_, volume


def create_trending_up_data(n=200):
    """Monotone rising prices."""
    close = np.arange(1, n + 1, dtype=np.float64)
    high = close + 1.0
    low = close - 1.0
    open_ = close + 0.5
    volume = np.ones(n) * 1000
    return close, high, low, open_, volume


def create_trending_down_data(n=200):
    """Monotone falling prices."""
    close = np.arange(n, 0, -1, dtype=np.float64)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    volume = np.ones(n) * 1000
    return close, high, low, open_, volume


def create_sine_wave_data(n=500, period=20):
    """Sine wave with known period for FTI testing."""
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / period)
    high = close + 1.0
    low = close - 1.0
    open_ = close + 0.5
    volume = np.ones(n) * 1000
    return close, high, low, open_, volume


def create_high_volatility_data(n=200, seed=43):
    """High volatility random walk."""
    np.random.seed(seed)
    close = (np.random.randn(n) * 5.0).cumsum() + 100
    high = close + np.abs(np.random.randn(n)) * 2.0
    low = close - np.abs(np.random.randn(n)) * 2.0
    open_ = close + np.random.randn(n) * 1.5
    volume = np.abs(np.random.randn(n)) * 2000 + 500
    return close, high, low, open_, volume


def create_constant_data(n=100):
    """Constant prices for edge case testing."""
    close = np.ones(n) * 100.0
    high = close + 0.5
    low = close - 0.5
    open_ = close
    volume = np.ones(n) * 1000
    return close, high, low, open_, volume


def generate_all_datasets():
    """Generate all test datasets."""
    datasets = {
        'random_walk.txt': create_random_walk_data(200, seed=42),
        'trending_up.txt': create_trending_up_data(200),
        'trending_down.txt': create_trending_down_data(200),
        'sine_wave.txt': create_sine_wave_data(500, period=20),
        'high_volatility.txt': create_high_volatility_data(200, seed=43),
        'constant.txt': create_constant_data(100),
    }

    print("\nGenerating test datasets for C++ reference comparison...\n")

    for filename, (close, high, low, open_, volume) in datasets.items():
        generate_market_file(close, high, low, open_, volume, filename)

    print("\n✓ All test datasets generated successfully!")
    return datasets


if __name__ == "__main__":
    generate_all_datasets()
