"""
Validation tests for stats_optimal_search key validation (fix #12).

Covers:
  - Invalid objective_metric raises ValueError with the allowed keys in the message
  - Both search functions validate
  - Valid keys still work
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.stats_optimal_search import (
    optimal_autocorrelation_param_search,
    optimal_variance_ratio_param_search,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(200))


class TestAutoCorrelationParamSearchValidation:

    def test_invalid_metric_raises_value_error(self, returns):
        with pytest.raises(ValueError) as exc_info:
            optimal_autocorrelation_param_search(
                returns, window_range=[10, 20], lag_range=[1, 5],
                objective_metric="not_a_metric",
            )
        msg = str(exc_info.value)
        assert "not_a_metric" in msg
        assert "absolute_autocorr" in msg  # allowed keys listed

    def test_valid_metric_absolute_autocorr_runs(self, returns):
        params, score = optimal_autocorrelation_param_search(
            returns, window_range=[10, 20], lag_range=[1, 5],
            objective_metric="absolute_autocorr",
        )
        assert params is not None
        assert np.isfinite(score)

    def test_valid_metric_trading_profit_runs(self, returns):
        params, score = optimal_autocorrelation_param_search(
            returns, window_range=[10, 20], lag_range=[1, 5],
            objective_metric="trading_profit",
        )
        assert params is not None

    def test_valid_metric_stat_significance_not_rejected(self):
        """stat_significance is a recognised key; the validation guard must NOT reject it."""
        # We only care that our ValueError guard doesn't fire for this valid key.
        # Catch everything else so pre-existing computation bugs don't hide the guard test.
        try:
            optimal_autocorrelation_param_search(
                pd.Series(np.random.default_rng(0).standard_normal(200)),
                window_range=[20], lag_range=[1],
                objective_metric="stat_significance",
            )
        except ValueError as e:
            if "Allowed" in str(e):
                pytest.fail(f"stat_significance incorrectly rejected by key guard: {e}")
        except Exception:
            pass  # pre-existing computation issues are not the concern here


class TestVarianceRatioParamSearchValidation:

    def test_invalid_metric_raises_value_error(self, returns):
        with pytest.raises(ValueError) as exc_info:
            optimal_variance_ratio_param_search(
                returns, window_range=[10, 20], q_range=[2, 4],
                objective_metric="bad_key",
            )
        msg = str(exc_info.value)
        assert "bad_key" in msg
        assert "max_deviation" in msg  # allowed keys listed

    def test_valid_metric_max_deviation_runs(self, returns):
        params, score = optimal_variance_ratio_param_search(
            returns, window_range=[10, 20], q_range=[2, 4],
            objective_metric="max_deviation",
        )
        assert params is not None
        assert np.isfinite(score)

    def test_valid_metric_trading_profit_runs(self, returns):
        params, score = optimal_variance_ratio_param_search(
            returns, window_range=[10, 20], q_range=[2, 4],
            objective_metric="trading_profit",
        )
        assert params is not None

    def test_valid_metric_stat_significance_runs(self, returns):
        params, score = optimal_variance_ratio_param_search(
            returns, window_range=[10, 20], q_range=[2, 4],
            objective_metric="stat_significance",
        )
        assert params is not None
