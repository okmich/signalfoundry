"""Regression tests for the bipower / jump-variation decomposition.

These pin the ONE property that makes the decomposition meaningful: on a continuous path,
BPV and RV estimate the same quantity, so RV/BPV sits at 1 and JV collapses to 0. Only a jump
drives RV above BPV.

The bug these guard against: BPV was computed as a rolling MEAN of |r_i||r_{i-1}| while RV was a
rolling SUM of r^2. That makes RV/BPV approximately sqrt(n_bars_in_window) on every bar regardless
of jumps, so JV/RV pins near 1 everywhere and the decomposition carries no jump information at all.
Measured on real 5m FX before the fix: RV/BPV medians of 3.49 / 6.09 / 8.63 for the 60/180/360m
windows, against sqrt(n) of 3.46 / 6.00 / 8.49.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volatility import (realized_volatility_with_bipower_jump_variations,
                                              realized_volatility_window_with_bipower_jump_variations)

BARS = 20_000
SIGMA = 3e-4


def _path(returns: np.ndarray) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(returns), freq="5min")
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)


@pytest.fixture
def continuous() -> pd.Series:
    rng = np.random.default_rng(7)
    return _path(rng.normal(0, SIGMA, BARS))


@pytest.fixture
def with_jumps() -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(7)
    r = rng.normal(0, SIGMA, BARS)
    at = rng.choice(np.arange(200, BARS), size=40, replace=False)
    r[at] += rng.choice([-1, 1], 40) * 8 * SIGMA
    return _path(r), at


class TestBipowerJumpVariation:

    @pytest.mark.parametrize("window", [60, 180, 360])
    def test_rv_over_bpv_is_unity_on_a_continuous_path(self, continuous, window):
        """The decomposition is only meaningful if BPV tracks RV when there are no jumps."""
        out = realized_volatility_window_with_bipower_jump_variations(continuous, windows=[window], annualize=False)
        ratio = (out[f"rv_{window}"] / out[f"bpv_{window}"]).dropna()
        assert 0.9 < ratio.median() < 1.1

    @pytest.mark.parametrize("window", [60, 180, 360])
    def test_ratio_does_not_scale_with_window_length(self, continuous, window):
        """Guards the sum-vs-mean bug directly: the ratio must not grow like sqrt(n)."""
        n_bars = window // 5
        out = realized_volatility_window_with_bipower_jump_variations(continuous, windows=[window], annualize=False)
        ratio = (out[f"rv_{window}"] / out[f"bpv_{window}"]).dropna()
        assert ratio.median() < 0.5 * np.sqrt(n_bars)

    def test_jump_variation_is_mostly_zero_without_jumps(self, continuous):
        out = realized_volatility_window_with_bipower_jump_variations(continuous, windows=[60], annualize=False)
        jv, rv = out["jv_60"], out["rv_60"]
        assert (jv == 0).mean() > 0.3
        assert (jv / rv).median() < 0.1

    def test_jumps_separate_from_clean_windows(self, with_jumps):
        """A window containing an injected jump must read materially higher than one that does not."""
        close, at = with_jumps
        out = realized_volatility_window_with_bipower_jump_variations(close, windows=[60], annualize=False)
        ratio = out["rv_60"] / out["bpv_60"]
        contaminated = pd.Series(False, index=close.index)
        for j in at:
            contaminated.iloc[j:min(j + 12, len(close))] = True
        assert ratio[contaminated].median() > 1.3 * ratio[~contaminated].median()

    def test_single_window_variant_agrees_with_multi_window(self, continuous):
        rv, bpv, jv = realized_volatility_with_bipower_jump_variations(continuous, window=60, annualize=False)
        out = realized_volatility_window_with_bipower_jump_variations(continuous, windows=[60], annualize=False)
        for got, name in ((rv, "rv_60"), (bpv, "bpv_60"), (jv, "jv_60")):
            pd.testing.assert_series_equal(got, out[name].loc[got.index], check_names=False)

    def test_single_window_variant_drops_the_first_bar(self, continuous):
        """Documents CURRENT behaviour, which is inconsistent between the two entry points.

        The single-window variant returns ``pd.concat`` of the dropna'd groups, so its index is the
        input minus the first bar. The multi-window variant assigns into a frame built on
        ``close.index``, so it keeps every bar and leads with NaN. Values agree where both are
        defined, and DatetimeIndex alignment protects pandas-level arithmetic -- but a caller doing
        ``rv.to_numpy()`` against ``close.to_numpy()`` is silently off by one bar. Change this only
        deliberately; it is an API change, not a bug fix.
        """
        rv, _, _ = realized_volatility_with_bipower_jump_variations(continuous, window=60, annualize=False)
        out = realized_volatility_window_with_bipower_jump_variations(continuous, windows=[60], annualize=False)
        assert len(rv) == len(continuous) - 1
        assert len(out) == len(continuous)
        assert rv.index[0] == continuous.index[1]
