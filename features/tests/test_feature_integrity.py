"""
Feature integrity check (validation plan item 4).

Runs core_path_structure_features on a fixed fixture dataset and asserts:
  - No feature column is entirely NaN or inf
  - After the warmup period, NaN/inf rates per feature are below a threshold
  - Output shape is preserved
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.path_structure import core_path_structure_features


# ─── Fixed fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ohlcv_fixture():
    """Reproducible 500-bar OHLCV fixture used for all integrity checks."""
    rng = np.random.default_rng(2024)
    n = 500
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    high = close * (1 + rng.uniform(0.001, 0.006, n))
    low = close * (1 - rng.uniform(0.001, 0.006, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = rng.uniform(1_000, 10_000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

MAX_NAN_RATE = 0.25   # allow at most 25% NaN after the warmup skip
WARMUP = 60           # skip first 60 bars for warmup


def _check_integrity(df: pd.DataFrame, label: str):
    """Assert integrity constraints on a feature DataFrame."""
    assert not df.empty, f"{label}: result is empty"
    assert len(df) == len(df), f"{label}: length mismatch"

    feature_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    assert len(feature_cols) > 0, f"{label}: no feature columns produced"

    post_warmup = df.iloc[WARMUP:]
    for col in feature_cols:
        series = post_warmup[col]
        inf_count = np.isinf(series.replace([np.nan], [0])).sum()
        assert inf_count == 0, f"{label}[{col}]: contains {inf_count} inf values after warmup"

        nan_rate = series.isna().mean()
        assert nan_rate <= MAX_NAN_RATE, (
            f"{label}[{col}]: NaN rate {nan_rate:.1%} exceeds {MAX_NAN_RATE:.0%} threshold after warmup"
        )


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestCorePathStructureIntegrity:

    def test_output_shape_preserved(self, ohlcv_fixture):
        result = core_path_structure_features(ohlcv_fixture)
        assert len(result) == len(ohlcv_fixture)

    def test_no_all_nan_columns(self, ohlcv_fixture):
        result = core_path_structure_features(ohlcv_fixture)
        for col in result.columns:
            assert not result[col].isna().all(), f"Column '{col}' is entirely NaN"

    def test_post_warmup_nan_rate(self, ohlcv_fixture):
        result = core_path_structure_features(ohlcv_fixture)
        _check_integrity(result, "core_path_structure_features")
