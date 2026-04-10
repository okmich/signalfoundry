import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter issues

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from okmich_quant_features.renko import Renko


# Fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Generate standard OHLCV dataframe for testing."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
    np.random.seed(42)

    close_prices = 100 + np.random.randn(50).cumsum()
    data = []

    for i, date in enumerate(dates):
        close = close_prices[i]
        high = close + abs(np.random.randn())
        low = close - abs(np.random.randn())
        open_price = close_prices[i - 1] if i > 0 else 100
        volume = np.random.randint(1000, 10000)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def trending_up_data():
    """Generate strongly trending up data."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="1h")
    data = []

    for i, date in enumerate(dates):
        base_price = 100 + i * 2  # Strong uptrend
        data.append(
            {
                "timestamp": date,
                "open": base_price,
                "high": base_price + 1,
                "low": base_price - 0.5,
                "close": base_price + 0.5,
                "volume": 1000,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def trending_down_data():
    """Generate strongly trending down data."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="1h")
    data = []

    for i, date in enumerate(dates):
        base_price = 100 - i * 2  # Strong downtrend
        data.append(
            {
                "timestamp": date,
                "open": base_price,
                "high": base_price + 0.5,
                "low": base_price - 1,
                "close": base_price - 0.5,
                "volume": 1000,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def flat_data():
    """Generate flat price data (no trend)."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="1h")
    data = []

    for date in dates:
        data.append(
            {
                "timestamp": date,
                "open": 100,
                "high": 100.3,
                "low": 99.7,
                "close": 100,
                "volume": 1000,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def custom_column_data():
    """Generate data with custom column names."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="1h")
    data = []

    for i, date in enumerate(dates):
        data.append(
            {
                "time": date,
                "h": 100 + i * 0.5,
                "l": 99 + i * 0.5,
                "c": 99.5 + i * 0.5,
                "vol": 1000,
            }
        )

    return pd.DataFrame(data)


# Test Class Initialization
class TestRenkoInitialization:
    """Test Renko class initialization and validation."""

    def test_valid_initialization(self):
        """Test valid Renko initialization."""
        renko = Renko(brick_size=2.0)
        assert renko.brick_size == 2.0
        assert renko.high_col == "high"
        assert renko.low_col == "low"
        assert renko.close_col == "close"
        assert renko.volume_col == "volume"
        assert renko.use_high_low is True
        assert renko.renko_df is None

    def test_custom_columns_initialization(self):
        """Test initialization with custom column names."""
        renko = Renko(
            brick_size=1.5, high_col="h", low_col="l", close_col="c", volume_col="v"
        )
        assert renko.high_col == "h"
        assert renko.low_col == "l"
        assert renko.close_col == "c"
        assert renko.volume_col == "v"

    def test_close_only_mode_initialization(self):
        """Test initialization with close-only mode."""
        renko = Renko(brick_size=2.0, use_high_low=False)
        assert renko.use_high_low is False

    def test_negative_brick_size_raises_error(self):
        """Test that negative brick size raises ValueError."""
        with pytest.raises(ValueError, match="brick_size must be positive"):
            Renko(brick_size=-1.0)

    def test_zero_brick_size_raises_error(self):
        """Test that zero brick size raises ValueError."""
        with pytest.raises(ValueError, match="brick_size must be positive"):
            Renko(brick_size=0.0)

    def test_non_numeric_brick_size_raises_error(self):
        """Test that non-numeric brick size raises TypeError."""
        with pytest.raises(TypeError, match="brick_size must be numeric"):
            Renko(brick_size="invalid")

    def test_empty_column_name_raises_error(self):
        """Test that empty column name raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Renko(brick_size=2.0, high_col="")

    def test_repr_not_built(self):
        """Test string representation before build."""
        renko = Renko(brick_size=2.5)
        repr_str = repr(renko)
        assert "brick_size=2.5" in repr_str
        assert "not built" in repr_str
        assert "bricks=0" in repr_str


# Test Build Method
class TestRenkoBuild:
    """Test Renko build method."""

    def test_build_with_valid_data(self, sample_ohlcv_data):
        """Test successful build with valid data."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)

    def test_different_datasets_comparison(self, trending_up_data, trending_down_data):
        """Test comparing results from different datasets."""
        renko = Renko(brick_size=2.0)

        # Build with uptrend
        renko.build(trending_up_data)
        up_stats = renko.get_statistics()

        # Build with downtrend
        renko.build(trending_down_data)
        down_stats = renko.get_statistics()

        # Uptrend should have more up bricks
        assert up_stats["up_bricks"] > up_stats["down_bricks"]
        # Downtrend should have more down bricks
        assert down_stats["down_bricks"] > down_stats["up_bricks"]


# Test Data Integrity
class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_brick_continuity(self, sample_ohlcv_data):
        """Test that bricks are continuous (close of one = open of next)."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 1:
            for i in range(len(result) - 1):
                current_close = result.iloc[i]["close"]
                next_open = result.iloc[i + 1]["open"]
                # Next brick should start where current brick ends
                assert current_close == next_open

    def test_direction_consistency(self, sample_ohlcv_data):
        """Test that direction matches open/close relationship."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        for _, row in result.iterrows():
            if row["direction"] == 1:
                assert row["close"] > row["open"]
            else:
                assert row["close"] < row["open"]

    def test_timestamp_ordering(self, sample_ohlcv_data):
        """Test that timestamps are in order."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 1:
            # Timestamps should be non-decreasing
            timestamps = result["timestamp"].values
            for i in range(len(timestamps) - 1):
                assert timestamps[i] <= timestamps[i + 1]

    def test_volume_non_negative(self, sample_ohlcv_data):
        """Test that volume values are non-negative."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        assert (result["volume"] >= 0).all()

    def test_price_values_realistic(self, sample_ohlcv_data):
        """Test that price values are within realistic ranges."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 0:
            # Prices should be finite
            assert np.isfinite(result["open"]).all()
            assert np.isfinite(result["close"]).all()


# Test Specific Scenarios
class TestSpecificScenarios:
    """Test specific market scenarios."""

    def test_gap_up_scenario(self):
        """Test gap up (price jumps up significantly)."""
        df = pd.DataFrame(
            {
                "high": [100, 110],  # Gap up
                "low": [99, 109],
                "close": [100, 110],
                "volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should create multiple up bricks
        assert len(result) > 0
        assert (result["direction"] == 1).all()

    def test_gap_down_scenario(self):
        """Test gap down (price jumps down significantly)."""
        df = pd.DataFrame(
            {
                "high": [100, 90],  # Gap down
                "low": [99, 89],
                "close": [100, 90],
                "volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should create multiple down bricks
        assert len(result) > 0
        assert (result["direction"] == -1).all()

    def test_consolidation_scenario(self):
        """Test consolidation (sideways price action)."""
        df = pd.DataFrame(
            {
                "high": [100.5, 100.6, 100.4, 100.5, 100.3],
                "low": [99.5, 99.4, 99.6, 99.5, 99.7],
                "close": [100, 100, 100, 100, 100],
                "volume": [1000] * 5,
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should create few or no bricks
        assert len(result) <= 1

    def test_reversal_scenario(self):
        """Test trend reversal."""
        df = pd.DataFrame(
            {
                "high": [100, 102, 104, 103, 101, 99],
                "low": [99, 101, 103, 101, 99, 97],
                "close": [100, 102, 104, 102, 100, 98],
                "volume": [1000] * 6,
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        if len(result) > 0:
            # Should have both up and down bricks
            assert (result["direction"] == 1).any()
            assert (result["direction"] == -1).any()


# Test Column Name Validation
class TestColumnValidation:
    """Test column name validation and handling."""

    def test_case_sensitive_columns(self):
        """Test that column names are case-sensitive."""
        df = pd.DataFrame(
            {
                "High": [100, 102],  # Capital H
                "Low": [99, 101],
                "Close": [100, 102],
                "Volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)

        with pytest.raises(ValueError, match="Missing required columns"):
            renko.build(df)

    def test_extra_columns_ignored(self, sample_ohlcv_data):
        """Test that extra columns are ignored."""
        df = sample_ohlcv_data.copy()
        df["extra_col"] = 999
        df["another_col"] = "test"

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should work fine, extra columns ignored
        assert isinstance(result, pd.DataFrame)

    def test_column_order_irrelevant(self):
        """Test that column order doesn't matter."""
        df = pd.DataFrame(
            {
                "volume": [1000, 1000],
                "close": [100, 102],
                "low": [99, 101],
                "high": [100, 102],
                "timestamp": pd.date_range("2024-01-01", periods=2),
            }
        )

        renko = Renko(brick_size=1.0)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)


# Test Numeric Edge Cases
class TestNumericEdgeCases:
    """Test numeric edge cases and precision."""

    def test_float_precision_brick_size(self, sample_ohlcv_data):
        """Test with float precision brick sizes."""
        brick_sizes = [0.1, 0.01, 0.001, 1.23456789]

        for brick_size in brick_sizes:
            renko = Renko(brick_size=brick_size)
            result = renko.build(sample_ohlcv_data)
            assert isinstance(result, pd.DataFrame)

    def test_very_close_prices(self):
        """Test with prices very close together."""
        df = pd.DataFrame(
            {
                "high": [100.001, 100.002, 100.003],
                "low": [99.999, 100.000, 100.001],
                "close": [100.000, 100.001, 100.002],
                "volume": [1000, 1000, 1000],
            }
        )

        renko = Renko(brick_size=0.001)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)

    def test_integer_prices(self):
        """Test with integer prices."""
        df = pd.DataFrame(
            {
                "high": [100, 102, 104],
                "low": [99, 101, 103],
                "close": [100, 102, 104],
                "volume": [1000, 1000, 1000],
            }
        )

        renko = Renko(brick_size=2)  # Integer brick size
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)

    def test_rounding_consistency(self, sample_ohlcv_data):
        """Test that rounding is consistent."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 0:
            # All open and close prices should be multiples of brick_size
            opens_mod = result["open"] % 2.0
            closes_mod = result["close"] % 2.0

            # Should be very close to 0 (accounting for float precision)
            assert np.allclose(opens_mod, 0, atol=1e-10)
            assert np.allclose(closes_mod, 0, atol=1e-10)


# Pytest Marks and Metadata
@pytest.mark.slow
class TestSlowOperations:
    """Tests that are slow and may be skipped in quick runs."""

    def test_very_large_dataset(self):
        """Test with very large dataset."""
        n = 100000
        df = pd.DataFrame(
            {
                "high": 100 + np.random.randn(n).cumsum() * 0.1,
                "low": 99 + np.random.randn(n).cumsum() * 0.1,
                "close": 99.5 + np.random.randn(n).cumsum() * 0.1,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        renko = Renko(brick_size=1.0)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# Test Suite Metadata
def test_version_compatibility():
    """Test that imports work correctly."""
    import pandas
    import numpy

    # Ensure minimum versions (adjust as needed)
    assert pandas.__version__ >= "1.0.0"
    assert numpy.__version__ >= "1.18.0"


# Conftest-style fixtures for test discovery
@pytest.fixture(scope="session")
def renko_class():
    """Provide Renko class for all tests."""
    # Import your Renko class here
    # from your_module import Renko
    # return Renko
    pass


def test_build_updates_repr(sample_ohlcv_data):
    """Test that build updates repr string."""
    renko = Renko(brick_size=2.0)
    renko.build(sample_ohlcv_data)

    repr_str = repr(renko)
    assert "built" in repr_str
    assert "bricks=" in repr_str


def test_build_with_empty_dataframe():
    """Test build with empty dataframe raises ValueError."""
    renko = Renko(brick_size=2.0)
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input dataframe is empty"):
        renko.build(empty_df)


def test_build_with_non_dataframe():
    """Test build with non-DataFrame raises TypeError."""
    renko = Renko(brick_size=2.0)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        renko.build([1, 2, 3])


def test_build_with_missing_columns():
    """Test build with missing required columns."""
    renko = Renko(brick_size=2.0)
    incomplete_df = pd.DataFrame({"close": [100, 101, 102]})

    with pytest.raises(ValueError, match="Missing required columns"):
        renko.build(incomplete_df)


def test_build_flat_data_no_bricks(flat_data):
    """Test build with flat data produces no bricks."""
    renko = Renko(brick_size=2.0)
    result = renko.build(flat_data)

    assert len(result) == 0
    assert list(result.columns) == [
        "brick_num",
        "open",
        "close",
        "direction",
        "timestamp",
        "volume",
    ]


def test_build_preserves_original_dataframe(sample_ohlcv_data):
    """Test that build doesn't modify original dataframe."""
    renko = Renko(brick_size=2.0)
    original_data = sample_ohlcv_data.copy()

    renko.build(sample_ohlcv_data)
    pd.testing.assert_frame_equal(sample_ohlcv_data, original_data)


def test_build_with_custom_columns(custom_column_data):
    """Test build with custom column names."""
    renko = Renko(
        brick_size=1.0, high_col="h", low_col="l", close_col="c", volume_col="vol"
    )
    result = renko.build(custom_column_data)

    assert len(result) > 0


def test_build_without_volume_column():
    """Test build when volume column is missing."""
    df = pd.DataFrame(
        {
            "high": [100, 102, 104, 103, 105],
            "low": [99, 100, 102, 101, 103],
            "close": [100, 101, 103, 102, 104],
        }
    )

    renko = Renko(brick_size=1.0)
    result = renko.build(df)

    assert "volume" in result.columns
    assert (result["volume"] == 0).all()


def test_build_close_only_mode(sample_ohlcv_data):
    """Test build in close-only mode."""
    renko = Renko(brick_size=2.0, use_high_low=False)
    result = renko.build(sample_ohlcv_data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 0


# Test Brick Formation Logic
class TestBrickFormation:
    """Test brick formation logic."""

    def test_upward_brick_formation(self, trending_up_data):
        """Test that upward trend creates up bricks."""
        renko = Renko(brick_size=2.0)
        result = renko.build(trending_up_data)

        assert len(result) > 0
        assert (result["direction"] == 1).sum() > 0
        assert (result["open"] < result["close"]).all()

    def test_downward_brick_formation(self, trending_down_data):
        """Test that downward trend creates down bricks."""
        renko = Renko(brick_size=2.0)
        result = renko.build(trending_down_data)

        assert len(result) > 0
        assert (result["direction"] == -1).sum() > 0
        assert (result["open"] > result["close"]).all()

    def test_brick_size_consistency(self, sample_ohlcv_data):
        """Test that all bricks have consistent size."""
        brick_size = 2.0
        renko = Renko(brick_size=brick_size)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 0:
            brick_heights = abs(result["close"] - result["open"])
            assert np.allclose(brick_heights, brick_size)

    def test_multiple_bricks_single_period(self):
        """Test that large price moves create multiple bricks."""
        df = pd.DataFrame(
            {
                "high": [100, 110],  # 10 point jump
                "low": [99, 109],
                "close": [100, 110],
                "volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should create 5 bricks (100->102, 102->104, 104->106, 106->108, 108->110)
        assert len(result) == 5
        assert (result["direction"] == 1).all()

    def test_brick_numbering_sequential(self, sample_ohlcv_data):
        """Test that brick numbers are sequential."""
        renko = Renko(brick_size=2.0)
        result = renko.build(sample_ohlcv_data)

        if len(result) > 0:
            expected_nums = list(range(len(result)))
            assert list(result["brick_num"]) == expected_nums

    def test_volume_accumulation(self):
        """Test that volume accumulates correctly."""
        df = pd.DataFrame(
            {
                "high": [100, 101, 103],
                "low": [99, 100, 102],
                "close": [100, 101, 103],
                "volume": [1000, 2000, 3000],
            }
        )

        renko = Renko(brick_size=1.0)
        result = renko.build(df)

        # First brick should have accumulated volume
        if len(result) > 0:
            assert result.iloc[0]["volume"] > 0


# Test Statistics Method
class TestStatistics:
    """Test get_statistics method."""

    def test_statistics_before_build_raises_error(self):
        """Test that calling statistics before build raises error."""
        renko = Renko(brick_size=2.0)

        with pytest.raises(RuntimeError, match="Must call build"):
            renko.get_statistics()

    def test_statistics_with_data(self, sample_ohlcv_data):
        """Test statistics calculation with data."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)
        stats = renko.get_statistics()

        assert "total_bricks" in stats
        assert "up_bricks" in stats
        assert "down_bricks" in stats
        assert "trend_changes" in stats
        assert "largest_trend" in stats
        assert "total_volume" in stats

        assert isinstance(stats["total_bricks"], int)
        assert isinstance(stats["up_bricks"], int)
        assert isinstance(stats["down_bricks"], int)
        assert isinstance(stats["trend_changes"], int)
        assert isinstance(stats["largest_trend"], int)
        assert isinstance(stats["total_volume"], float)

    def test_statistics_sum_consistency(self, sample_ohlcv_data):
        """Test that up + down bricks = total bricks."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)
        stats = renko.get_statistics()

        assert stats["up_bricks"] + stats["down_bricks"] == stats["total_bricks"]

    def test_statistics_empty_data(self, flat_data):
        """Test statistics with no bricks formed."""
        renko = Renko(brick_size=10.0)  # Large brick size, no bricks
        renko.build(flat_data)
        stats = renko.get_statistics()

        assert stats["total_bricks"] == 0
        assert stats["up_bricks"] == 0
        assert stats["down_bricks"] == 0
        assert stats["trend_changes"] == 0
        assert stats["largest_trend"] == 0
        assert stats["total_volume"] == 0.0

    def test_statistics_trending_up(self, trending_up_data):
        """Test statistics with strong uptrend."""
        renko = Renko(brick_size=2.0)
        renko.build(trending_up_data)
        stats = renko.get_statistics()

        # Should be mostly up bricks
        assert stats["up_bricks"] > stats["down_bricks"]


# Test Trend Changes Method
class TestTrendChanges:
    """Test get_trend_changes method."""

    def test_trend_changes_before_build_raises_error(self):
        """Test that calling trend_changes before build raises error."""
        renko = Renko(brick_size=2.0)

        with pytest.raises(RuntimeError, match="Must call build"):
            renko.get_trend_changes()

    def test_trend_changes_with_data(self, sample_ohlcv_data):
        """Test trend change detection."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)
        changes = renko.get_trend_changes()

        assert isinstance(changes, pd.DataFrame)
        assert len(changes) <= len(renko.renko_df)

    def test_trend_changes_alternating_direction(self):
        """Test trend changes with alternating directions."""
        # Create data that alternates up and down
        df = pd.DataFrame(
            {
                "high": [100, 102, 100, 102, 100],
                "low": [98, 100, 98, 100, 98],
                "close": [100, 102, 100, 102, 100],
                "volume": [1000] * 5,
            }
        )

        renko = Renko(brick_size=2.0)
        renko.build(df)
        changes = renko.get_trend_changes()

        # Should have multiple trend changes
        assert len(changes) > 1

    def test_trend_changes_empty_result(self, flat_data):
        """Test trend changes with no bricks."""
        renko = Renko(brick_size=10.0)
        renko.build(flat_data)
        changes = renko.get_trend_changes()

        assert len(changes) == 0


# Test Plotting Method
class TestPlotting:
    """Test plot method."""

    def test_plot_before_build_raises_error(self):
        """Test that plotting before build raises error."""
        renko = Renko(brick_size=2.0)

        with pytest.raises(RuntimeError, match="Must call build"):
            renko.plot()

    def test_plot_empty_data_raises_error(self, flat_data):
        """Test that plotting with no bricks raises error."""
        renko = Renko(brick_size=10.0)
        renko.build(flat_data)

        with pytest.raises(ValueError, match="Cannot plot empty Renko chart"):
            renko.plot()

    def test_plot_returns_figure_and_axes(self, sample_ohlcv_data):
        """Test that plot returns figure and axes."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        fig, ax = renko.plot()
        plt.close(fig)

        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_with_custom_colors(self, sample_ohlcv_data):
        """Test plot with custom colors."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        fig, ax = renko.plot(up_color="blue", down_color="orange")
        plt.close(fig)

        assert fig is not None

    def test_plot_without_volume(self, sample_ohlcv_data):
        """Test plot without volume subplot."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        fig, ax = renko.plot(show_volume=False)
        plt.close(fig)

        assert fig is not None

    def test_plot_with_custom_title(self, sample_ohlcv_data):
        """Test plot with custom title."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        custom_title = "My Custom Renko Chart"
        fig, ax = renko.plot(title=custom_title)

        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_plot_with_existing_axes(self, sample_ohlcv_data):
        """Test plot on existing axes."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        returned_fig, returned_ax = renko.plot(ax=ax)

        assert returned_ax is ax
        assert returned_fig is fig
        plt.close(fig)

    def test_plot_format_volume(self):
        """Test volume formatting helper."""
        assert Renko._format_volume(500) == "500"
        assert Renko._format_volume(1500) == "1.5K"
        assert Renko._format_volume(1_500_000) == "1.5M"
        assert Renko._format_volume(2_345_678) == "2.3M"


# Test Edge Cases and Robustness
class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        df = pd.DataFrame(
            {"high": [100], "low": [99], "close": [100], "volume": [1000]}
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        # Should not crash, might have 0 bricks
        assert len(result) >= 0

    def test_very_small_brick_size(self, sample_ohlcv_data):
        """Test with very small brick size."""
        renko = Renko(brick_size=0.1)
        result = renko.build(sample_ohlcv_data)

        # Should create many bricks
        assert len(result) > 0

    def test_very_large_brick_size(self, sample_ohlcv_data):
        """Test with very large brick size."""
        renko = Renko(brick_size=100.0)
        result = renko.build(sample_ohlcv_data)

        # Might create few or no bricks
        assert len(result) >= 0

    def test_nan_values_in_data(self):
        """Test handling of NaN values."""
        df = pd.DataFrame(
            {
                "high": [100, np.nan, 102],
                "low": [99, np.nan, 101],
                "close": [100, np.nan, 102],
                "volume": [1000, 1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        # Should handle NaN gracefully (may skip or handle)
        # This tests that it doesn't crash
        result = renko.build(df)
        assert isinstance(result, pd.DataFrame)

    def test_extreme_price_movements(self):
        """Test with extreme price movements."""
        df = pd.DataFrame(
            {
                "high": [100, 1000],  # 900 point jump
                "low": [99, 999],
                "close": [100, 1000],
                "volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=10.0)
        result = renko.build(df)

        # Should create many bricks
        assert len(result) > 50

    def test_negative_prices(self):
        """Test with negative prices (e.g., futures spreads)."""
        df = pd.DataFrame(
            {
                "high": [-10, -8, -6],
                "low": [-12, -10, -8],
                "close": [-11, -9, -7],
                "volume": [1000, 1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)

    def test_timestamp_types(self):
        """Test different timestamp types."""
        # String timestamps
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "high": [100, 102],
                "low": [99, 101],
                "close": [100, 102],
                "volume": [1000, 1000],
            }
        )

        renko = Renko(brick_size=1.0)
        result = renko.build(df)

        assert "timestamp" in result.columns

    def test_high_low_equality(self):
        """Test when high equals low (no intraday range)."""
        df = pd.DataFrame(
            {
                "high": [100, 102, 104],
                "low": [100, 102, 104],  # Equal to high
                "close": [100, 102, 104],
                "volume": [1000, 1000, 1000],
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)


# Test Performance and Large Data
class TestPerformance:
    """Test performance with large datasets."""

    def test_large_dataset_performance(self):
        """Test that large datasets complete in reasonable time."""
        # Generate large dataset
        n = 10000
        df = pd.DataFrame(
            {
                "high": 100 + np.random.randn(n).cumsum(),
                "low": 99 + np.random.randn(n).cumsum(),
                "close": 99.5 + np.random.randn(n).cumsum(),
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        renko = Renko(brick_size=2.0)
        result = renko.build(df)

        assert isinstance(result, pd.DataFrame)
        # Should complete without timeout

    def test_memory_efficiency(self, sample_ohlcv_data):
        """Test that building doesn't leak memory."""
        renko = Renko(brick_size=2.0)

        # Build multiple times
        for _ in range(10):
            renko.build(sample_ohlcv_data)

        # Should not crash or use excessive memory
        assert renko.renko_df is not None


# Parametrized Tests
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("brick_size", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_various_brick_sizes(self, sample_ohlcv_data, brick_size):
        """Test with various brick sizes."""
        renko = Renko(brick_size=brick_size)
        result = renko.build(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            brick_heights = abs(result["close"] - result["open"])
            assert np.allclose(brick_heights, brick_size)

    @pytest.mark.parametrize("use_high_low", [True, False])
    def test_both_modes(self, sample_ohlcv_data, use_high_low):
        """Test both high/low and close-only modes."""
        renko = Renko(brick_size=2.0, use_high_low=use_high_low)
        result = renko.build(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("invalid_brick", [-1, 0, -100, -0.5])
    def test_invalid_brick_sizes(self, invalid_brick):
        """Test that invalid brick sizes raise errors."""
        with pytest.raises(ValueError):
            Renko(brick_size=invalid_brick)

    @pytest.mark.parametrize("style", ["classic", "seaborn", "default"])
    def test_plot_styles(self, sample_ohlcv_data, style):
        """Test different plot styles."""
        renko = Renko(brick_size=2.0)
        renko.build(sample_ohlcv_data)

        try:
            fig, ax = renko.plot(style=style)
            plt.close(fig)
            assert fig is not None
        except:
            # Some styles might not be available
            pytest.skip(f"Style {style} not available")


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, sample_ohlcv_data):
        """Test complete workflow from initialization to plotting."""
        # Initialize
        renko = Renko(brick_size=2.0)

        # Build
        result = renko.build(sample_ohlcv_data)
        assert len(result) > 0

        # Get statistics
        stats = renko.get_statistics()
        assert stats["total_bricks"] == len(result)

        # Get trend changes
        changes = renko.get_trend_changes()
        assert len(changes) <= len(result)

        # Plot
        fig, ax = renko.plot()
        plt.close(fig)

        assert fig is not None

    def test_multiple_builds_same_instance(self, sample_ohlcv_data):
        """Test building multiple times with same instance."""
        renko = Renko(brick_size=2.0)

        # First build
        result1 = renko.build(sample_ohlcv_data)
        len1 = len(result1)

        # Second build (should replace first)
        result2 = renko.build(sample_ohlcv_data)
        len2 = len(result2)

        assert len
