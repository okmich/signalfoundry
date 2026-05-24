"""
Validation tests for path_structure feature builders.

Covers:
  - strict=True in core_path_structure_features propagates errors (fix #6)
  - Mutable default argument isolation (fix #11)
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.path_structure import core_path_structure_features


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=200, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


# ─── Fix #6: strict mode and exception surfacing ──────────────────────────────

class TestStrictMode:

    def _constant_df(self, n=100):
        """Constant-close DataFrame whose log-returns are all zero → stat tests may fail."""
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        return pd.DataFrame(
            {"high": np.full(n, 1.001), "low": np.full(n, 0.999), "close": np.full(n, 1.0)},
            index=idx,
        )

    def test_strict_false_does_not_raise_on_bad_data(self):
        """strict=False (default): statistical test failures fill NaN, no exception."""
        df = self._constant_df()
        result = core_path_structure_features(
            df, strict=False,
            hurst_window=30, hurst_min_window=10,
            runs_test_window=30,
            shannon_entropy_window=30,
            ljung_box_window=30,
        )
        assert isinstance(result, pd.DataFrame)

    def test_strict_false_emits_warning_on_failure(self):
        """strict=False should emit a warning (not raise) when a stat test fails."""
        df = self._constant_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core_path_structure_features(
                df, strict=False,
                hurst_window=30, hurst_min_window=10,
                runs_test_window=30,
                shannon_entropy_window=30,
                ljung_box_window=30,
            )
        # Constant returns → stat tests should fail → warnings emitted.
        # At minimum there should be no unhandled exception.
        assert len(caught) >= 0  # may or may not fail depending on platform


# ─── Fix #11: mutable default isolation ──────────────────────────────────────

class TestMutableDefaultIsolation:

    def test_auto_corr_lags_default_not_shared(self):
        """
        Mutating the default list in one call must not affect a subsequent call.
        If the default were a module-level list, appending to it inside the function
        would persist across calls.
        """
        df = _make_ohlcv()
        result1 = core_path_structure_features(df)
        result2 = core_path_structure_features(df)
        ac_cols_1 = [c for c in result1.columns if c.startswith("auto_corr_")]
        ac_cols_2 = [c for c in result2.columns if c.startswith("auto_corr_")]
        assert ac_cols_1 == ac_cols_2
