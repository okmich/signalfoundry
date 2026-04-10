import numpy as np
import pandas as pd
import pytest

from okmich_quant_research import k_ratio


@pytest.fixture
def mock_equity():
    """Simulate an equity curve with exponential growth and mild noise."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 500)
    equity = 1000 * np.exp(np.cumsum(returns))
    return pd.Series(equity)


def test_basic_k_ratio(mock_equity):
    """Test base K-Ratio returns a valid positive number."""
    k = k_ratio(mock_equity, periods_per_year=252)
    assert np.isfinite(k), "K-Ratio should be finite"
    assert isinstance(k, float), "Output should be a float"


def test_k_ratio_handles_dataframe(mock_equity):
    """Test K-Ratio accepts a DataFrame input."""
    df = pd.DataFrame({"equity": mock_equity})
    k = k_ratio(df, periods_per_year=252)
    assert np.isfinite(k), "K-Ratio should compute correctly for DataFrame input"


def test_k_ratio_with_nans_and_infs(mock_equity):
    """Ensure NaNs and infinities are handled gracefully."""
    equity = mock_equity.copy()
    equity.iloc[::50] = np.nan
    equity.iloc[::100] = np.inf
    k = k_ratio(equity, periods_per_year=252)
    assert np.isfinite(k), "Function should handle NaN/inf gracefully"


def test_k_ratio_normalization(mock_equity):
    """Test normalization scales K-Ratio between 0 and 1."""
    k_norm = k_ratio(mock_equity, periods_per_year=252, normalize=True)
    assert 0 <= k_norm <= 1, "Normalized K-Ratio must be between 0 and 1"


def test_k_ratio_small_sample():
    """Test function raises error for too few data points."""
    with pytest.raises(ValueError):
        k_ratio([100, 101, 102, 103])  # Less than 5 data points


def test_k_ratio_rolling_window(mock_equity):
    """Test rolling K-Ratio returns a Series of expected length."""
    result = k_ratio(mock_equity, periods_per_year=252, window=50)
    assert isinstance(result, pd.Series)
    assert len(result) == len(mock_equity)
    assert (
        result.isna().sum() > 0
    ), "Rolling series should contain NaNs for initial periods"
    assert np.isfinite(
        result.dropna()
    ).all(), "All computed rolling K-Ratios should be finite"


def test_k_ratio_clipping_to_prevent_log_zero():
    """Test that zero equity values are handled gracefully by filtering."""
    # With only 4 valid (positive) values after filtering zero, should return NaN
    equity = pd.Series([100, 100, 0, 105, 110])
    k = k_ratio(equity)
    assert np.isnan(k), "Should return NaN when fewer than 5 positive values remain"

    # With enough valid values, should compute successfully
    equity_valid = pd.Series([100, 101, 0, 105, 110, 115, 120])
    k_valid = k_ratio(equity_valid)
    assert np.isfinite(
        k_valid
    ), "Should compute successfully with enough positive values"


def test_k_ratio_with_custom_periods(mock_equity):
    """Test custom periods_per_year scaling."""
    k_daily = k_ratio(mock_equity, periods_per_year=252)
    k_hourly = k_ratio(mock_equity, periods_per_year=24 * 365)
    assert k_hourly != k_daily, "Changing periods_per_year should change the result"


def test_k_ratio_compare_with_linear_growth():
    """Check monotonic equity produces strong K-Ratio (high value)."""
    equity = pd.Series(np.linspace(100, 200, 500))
    k = k_ratio(equity)
    assert k > 5, f"K-Ratio should be high for perfectly linear growth, got {k:.3f}"
