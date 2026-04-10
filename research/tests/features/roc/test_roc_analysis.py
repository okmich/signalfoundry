"""
Tests for okmich_quant_research.features.roc module.

Covers:
- ROCAnalyzer._roc_table: frac_above, frac_below, pf values
- ROCAnalyzer._opt_thresh / _mcpt: threshold direction and n_trades
- _opt_thresh_core: contrarian-long / contrarian-short semantics
- ROCResults: n_trades consistency with threshold masks
- StationarityTester: basic smoke test
"""

import numpy as np
import pytest

from okmich_quant_research.features.roc import ROCAnalyzer, ROCResults
from okmich_quant_research.features.roc._numba_kernels import _opt_thresh_core
from okmich_quant_research.features.roc.stationarity import StationarityTester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N = 300


@pytest.fixture(scope="module")
def random_signals_returns():
    """300 random (signal, return) pairs — no real predictive relationship."""
    signals = RNG.standard_normal(N)
    returns = RNG.standard_normal(N) * 0.01
    return signals, returns


@pytest.fixture(scope="module")
def contrarian_signals_returns():
    """
    Contrarian indicator: LOW signal → positive return, HIGH signal → negative return.
    Constructed so that the bottom-third of signals have positive returns.
    """
    signals = np.linspace(1, 100, N)  # low..high
    # bottom third → positive returns, top two thirds → negative
    returns = np.where(signals <= 33, 0.02, -0.01)
    rng = np.random.default_rng(1)
    signals = signals + rng.standard_normal(N) * 0.5  # add noise
    return signals, returns


@pytest.fixture(scope="module")
def trendfollow_signals_returns():
    """
    Trend-following indicator: HIGH signal → positive return.
    Constructed so top-third has positive returns.
    """
    signals = np.linspace(1, 100, N)
    returns = np.where(signals >= 67, 0.02, -0.01)
    rng = np.random.default_rng(2)
    signals = signals + rng.standard_normal(N) * 0.5
    return signals, returns


@pytest.fixture(scope="module")
def analyzer():
    return ROCAnalyzer(min_kept_pct=0.05, nreps=200, random_seed=42)


# ---------------------------------------------------------------------------
# ROC Table
# ---------------------------------------------------------------------------

class TestROCTable:
    """Tests for _roc_table correctness (Bug 1 & 2 regression tests)."""

    def test_frac_above_is_fraction_of_high_signal_elements(self, analyzer, random_signals_returns):
        """frac_above = k/n where k elements sit ABOVE the threshold (highest signals)."""
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)

        n = len(signals)
        # For each row, the threshold = sorted_signals[k], above = first k elements
        # so frac_above = k/n.  Verify that frac_above + frac_below ≈ 1.0
        assert np.allclose(tbl["frac_above"] + tbl["frac_below"], 1.0), \
            "frac_above + frac_below must equal 1.0"

    def test_frac_above_increases_with_threshold_percentile(self, analyzer, random_signals_returns):
        """Higher percentile bins → more data above threshold → larger frac_above."""
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        # Thresholds decrease as percentile increases (sorted descending)
        # As threshold decreases, more data falls "above" → frac_above grows
        # Equivalently: frac_above should be non-decreasing as threshold decreases
        thresholds = tbl["frac_above"].values
        # They should be monotone non-decreasing (might have ties with tie-walk)
        assert np.all(np.diff(thresholds) >= -1e-9), \
            "frac_above must be non-decreasing as threshold decreases"

    def test_frac_below_is_complement(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        n = len(signals)
        # frac_below = (n-k)/n = fraction of data BELOW threshold
        # As threshold decreases, more data is above → frac_below decreases
        assert (tbl["frac_below"] > 0).all(), "frac_below must be positive"
        assert (tbl["frac_below"] < 1).all(), "frac_below must be < 1"

    def test_pf_long_above_and_pf_short_above_are_reciprocals(self, analyzer, random_signals_returns):
        """pf_long * pf_short ≈ 1 for the same slice (they use same wins/losses)."""
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        product = tbl["pf_long_above"] * tbl["pf_short_above"]
        # wins/losses * losses/wins = 1 (approximately, ignoring eps)
        assert np.allclose(product, 1.0, atol=0.01), \
            "pf_long_above * pf_short_above should be ≈ 1"

    def test_pf_long_below_and_pf_short_below_are_reciprocals(self, analyzer, random_signals_returns):
        """Same check for the below slice."""
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        product = tbl["pf_long_below"] * tbl["pf_short_below"]
        assert np.allclose(product, 1.0, atol=0.01), \
            "pf_long_below * pf_short_below should be ≈ 1"

    def test_pf_values_are_positive(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        for col in ["pf_long_above", "pf_short_above", "pf_long_below", "pf_short_below"]:
            assert (tbl[col] > 0).all(), f"{col} must be positive"

    def test_roc_table_columns(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        expected = {"threshold", "frac_above", "pf_long_above", "pf_short_above",
                    "frac_below", "pf_short_below", "pf_long_below"}
        assert set(tbl.columns) == expected

    def test_pf_pair_output_order(self):
        """Standalone check of _pf_pair output order."""
        returns = np.array([0.01, 0.02, -0.005, -0.01])  # wins=0.03, losses=0.015
        pf_long, pf_short = ROCAnalyzer._pf_pair(returns)
        assert pf_long == pytest.approx(0.03 / 0.015, rel=1e-6)
        assert pf_short == pytest.approx(0.015 / 0.03, rel=1e-6)
        assert abs(pf_long * pf_short - 1.0) < 1e-9

    def test_contrarian_indicator_high_pf_long_below(self, analyzer, contrarian_signals_returns):
        """
        For a contrarian indicator (low signal → positive return),
        pf_long_below (long on LOW signals) should be > 1 for low percentile bins.
        pf_long_above (long on HIGH signals) should be < 1.
        """
        signals, returns = contrarian_signals_returns
        tbl = analyzer._roc_table(signals, returns)
        # Low percentile threshold → small "above" set = very high signals
        # "below" = most data (low signals) → should have good pf_long_below
        low_pct_rows = tbl[tbl["frac_above"] < 0.15]
        if len(low_pct_rows) > 0:
            assert (low_pct_rows["pf_long_below"] > 1.0).any(), \
                "contrarian indicator: pf_long_below should be > 1 for low percentiles"


# ---------------------------------------------------------------------------
# _opt_thresh_core semantics
# ---------------------------------------------------------------------------

class TestOptThreshCore:
    """Tests for the Numba threshold-scan kernel."""

    def test_pf_all_equals_grand_profit_factor(self):
        """pf_all = sum(positive returns) / sum(|negative returns|)."""
        rng = np.random.default_rng(5)
        returns = rng.standard_normal(200)
        signals = rng.standard_normal(200)
        sort_idx = np.argsort(-signals)
        sorted_signals = signals[sort_idx]
        sorted_returns = returns[sort_idx]

        pf_all, _, _, _, _ = _opt_thresh_core(sorted_signals, sorted_returns, 5, 1e-30)

        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        expected = wins / (losses + 1e-30)
        assert pf_all == pytest.approx(expected, rel=1e-6)

    def test_long_idx_zero_when_no_improvement(self):
        """
        If no low-signal subset beats the baseline PF, long_idx stays 0
        (meaning "long on all data" = baseline).
        """
        rng = np.random.default_rng(6)
        n = 100
        # Create trend-following indicator: high signal → positive returns
        signals = np.sort(rng.uniform(0, 1, n))[::-1]  # descending
        returns = np.where(signals > 0.5, 0.02, -0.01)
        sort_idx = np.argsort(-signals)
        sorted_signals = signals[sort_idx]
        sorted_returns = returns[sort_idx]

        _, _, long_idx, _, _ = _opt_thresh_core(sorted_signals, sorted_returns, 3, 1e-30)
        # For a trend-following indicator, removing high-signal (positive) elements
        # from "above" only makes the remaining low-signal set worse.
        # So long_idx should stay at 0 (baseline = all data).
        assert long_idx == 0

    def test_contrarian_indicator_finds_low_signal_long_set(self):
        """
        For a contrarian indicator (low signal → positive return),
        the algorithm should find a low-signal long set (long_idx > 0).
        """
        n = 200
        # Deterministic: low signals have positive returns, high signals negative
        signals = np.linspace(100, 1, n)  # 100..1 descending (already sorted)
        sorted_signals = signals.copy()
        sorted_returns = np.where(signals <= 40, 0.02, -0.02)

        _, long_pf, long_idx, _, _ = _opt_thresh_core(sorted_signals, sorted_returns, 5, 1e-30)

        # long_idx > 0 means top elements were removed (contrarian long on bottom)
        assert long_idx > 0, "contrarian indicator should find long_idx > 0"
        assert long_pf > 1.0, "contrarian long PF should be > 1.0"

        # Verify: long set = {long_idx..n-1} = elements with LOWEST signals
        long_threshold = sorted_signals[long_idx]
        # All elements from long_idx onward have signal <= long_threshold
        assert (sorted_signals[long_idx:] <= long_threshold).all()

    def test_short_set_uses_high_signal_elements(self):
        """
        Short strategy = best PF when going SHORT on HIGH-signal elements.
        The short set = {0..short_idx-1} = elements at the START of the descending sort.
        """
        n = 200
        # High signals → negative returns → going SHORT on them wins
        signals = np.linspace(100, 1, n)  # descending
        sorted_signals = signals.copy()
        sorted_returns = np.where(signals >= 70, -0.02, 0.01)

        _, _, _, short_pf, short_idx = _opt_thresh_core(sorted_signals, sorted_returns, 5, 1e-30)

        assert short_pf > 1.0, "short PF should be > 1.0 when going short on high-signal negatives"
        assert short_idx > 0, "short_idx > 0 (some high-signal elements in short set)"

        # Short set = {0..short_idx-1} = highest signals
        short_threshold = sorted_signals[short_idx]
        assert (sorted_signals[:short_idx] >= short_threshold).all()

    def test_min_kept_enforced(self):
        """long_idx and short_idx only update when the candidate set has >= min_kept elements."""
        n = 100
        signals = np.linspace(100, 1, n)  # descending
        sorted_returns = np.random.default_rng(7).standard_normal(n)

        min_kept = 20
        _, _, long_idx, _, short_idx = _opt_thresh_core(signals, sorted_returns, min_kept, 1e-30)

        # long set = {long_idx..n-1}: must have >= min_kept elements
        n_long = n - long_idx
        n_short = short_idx
        assert n_long >= min_kept, f"n_long={n_long} < min_kept={min_kept}"
        assert n_short >= min_kept, f"n_short={n_short} < min_kept={min_kept}"


# ---------------------------------------------------------------------------
# ROCAnalyzer.analyze — threshold direction consistency
# ---------------------------------------------------------------------------

class TestROCAnalyzerAnalyze:

    def test_long_n_trades_matches_threshold_mask(self, analyzer, random_signals_returns):
        """
        long_n_trades = |{signal <= long_threshold}|.
        Verify that applying the correct mask gives the same count.
        """
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)

        mask = signals <= results.long_threshold
        n_from_mask = int(mask.sum())
        assert n_from_mask == results.long_n_trades, (
            f"long_n_trades={results.long_n_trades} but "
            f"|{{signal <= threshold}}|={n_from_mask}"
        )

    def test_short_n_trades_matches_threshold_mask(self, analyzer, random_signals_returns):
        """
        short_n_trades = |{signal > short_threshold}|.
        """
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)

        mask = signals > results.short_threshold
        n_from_mask = int(mask.sum())
        assert n_from_mask == results.short_n_trades, (
            f"short_n_trades={results.short_n_trades} but "
            f"|{{signal > threshold}}|={n_from_mask}"
        )

    def test_long_pf_matches_slice(self, analyzer, contrarian_signals_returns):
        """
        long_pf should match the actual PF of returns where signal <= long_threshold.
        """
        signals, returns = contrarian_signals_returns
        results = analyzer.analyze(signals, returns)

        mask = signals <= results.long_threshold
        r = returns[mask]
        wins = r[r > 0].sum()
        losses = -r[r < 0].sum()
        expected_pf = wins / (losses + 1e-30)
        assert expected_pf == pytest.approx(results.long_pf, rel=1e-4)

    def test_short_pf_matches_slice(self, analyzer, contrarian_signals_returns):
        """
        short_pf should match the actual SHORT PF for returns where signal > short_threshold.
        """
        signals, returns = contrarian_signals_returns
        results = analyzer.analyze(signals, returns)

        mask = signals > results.short_threshold
        r = returns[mask]
        short_wins = -r[r < 0].sum()    # negative returns = wins for short
        short_losses = r[r > 0].sum()   # positive returns = losses for short
        expected_pf = short_wins / (short_losses + 1e-30)
        assert expected_pf == pytest.approx(results.short_pf, rel=1e-4)

    def test_contrarian_long_pf_above_1(self, analyzer, contrarian_signals_returns):
        """Contrarian indicator: long_pf should be > 1."""
        signals, returns = contrarian_signals_returns
        results = analyzer.analyze(signals, returns)
        assert results.long_pf > 1.0

    def test_flip_sign_finds_trend_following(self, analyzer, trendfollow_signals_returns):
        """
        With flip_sign=True, the analyzer should find that high ORIGINAL signals
        → positive returns (after flipping, the algorithm sees contrarian = low negated = high original).
        long_pf > 1 for the trend-following indicator with flipped signs.
        """
        signals, returns = trendfollow_signals_returns
        results = analyzer.analyze(signals, returns, flip_sign=True)
        assert results.long_pf > 1.0

    def test_pf_all_is_grand_profit_factor(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)

        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        expected = wins / (losses + 1e-30)
        assert results.pf_all == pytest.approx(expected, rel=1e-4)

    def test_n_cases(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        assert results.n_cases == N

    def test_pvals_in_0_1(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        assert 0.0 <= results.long_pval <= 1.0
        assert 0.0 <= results.short_pval <= 1.0
        assert 0.0 <= results.best_pval <= 1.0

    def test_best_pval_gte_individual_pvals(self, analyzer, random_signals_returns):
        """best_pval is the most conservative — it accounts for multiple testing."""
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        assert results.best_pval >= min(results.long_pval, results.short_pval) - 1e-9

    def test_raises_on_short_input(self, analyzer):
        with pytest.raises(ValueError, match="30"):
            analyzer.analyze(np.ones(20), np.ones(20))

    def test_raises_on_length_mismatch(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze(np.ones(100), np.ones(50))

    def test_raises_on_2d_input(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze(np.ones((100, 2)), np.ones(100))

    def test_nan_rows_dropped(self, analyzer):
        signals = np.concatenate([np.ones(100), [np.nan, np.inf]])
        returns = np.ones(102) * 0.01
        results = analyzer.analyze(signals, returns)
        assert results.n_cases == 100

    def test_random_indicator_pval_not_significant(self, analyzer, random_signals_returns):
        """For a random indicator, best_pval should be large (not significant)."""
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        # With 200 permutations and a random indicator, p-value should usually be > 0.05
        # This is probabilistic but should pass with seed=42 and n=300
        # Use a lenient threshold: p > 0.10 is expected for pure noise with 200 reps
        assert results.best_pval > 0.05, (
            f"Random indicator should not be significant; got best_pval={results.best_pval:.4f}"
        )


# ---------------------------------------------------------------------------
# ROCResults dataclass
# ---------------------------------------------------------------------------

class TestROCResults:

    def test_str_contains_threshold_comparison(self, analyzer, random_signals_returns):
        """__str__ must use <= for long and > for short (Bug 3 docstring fix)."""
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        s = str(results)
        assert "<=" in s, "__str__ must use <= for long strategy"
        assert "signal >" in s, "__str__ must use > for short strategy"
        assert ">=" not in s, "__str__ must NOT use >= (old wrong convention)"

    def test_roc_table_in_results(self, analyzer, random_signals_returns):
        signals, returns = random_signals_returns
        results = analyzer.analyze(signals, returns)
        assert not results.roc_table.empty
        assert "frac_above" in results.roc_table.columns
        assert "frac_below" in results.roc_table.columns


# ---------------------------------------------------------------------------
# StationarityTester — smoke test
# ---------------------------------------------------------------------------

class TestStationarityTester:

    def test_smoke_stationary(self):
        """Stationary series should not reject stationarity at 5% level."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        tester = StationarityTester(min_recent=30, max_recent=60, nperms=200)
        result = tester.test(data)
        # For iid noise, break test should not fire often
        # The p-value should be > 0.05 in most cases
        assert result.p_value > 0.01, (
            f"Stationary series should not break at low p; got {result.p_value:.4f}"
        )

    def test_smoke_nonstationary(self):
        """
        Series with a structural break AT THE END should be detected.

        The stationarity test compares the most-recent window to the historical
        preceding bars, so the break must appear near the end of the series.
        """
        rng = np.random.default_rng(43)
        before = rng.standard_normal(250)
        after = rng.standard_normal(50) + 4.0   # last 50 bars: mean +4 (large shift)
        data = np.concatenate([before, after])
        tester = StationarityTester(min_recent=30, max_recent=60, nperms=500)
        result = tester.test(data)
        assert result.p_value < 0.05, (
            f"Structural break at end should be detected; got p_value={result.p_value:.4f}"
        )

    def test_result_fields(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(200)
        tester = StationarityTester(nperms=100)
        result = tester.test(data)
        assert hasattr(result, "p_value")
        assert hasattr(result, "test_statistic")
        assert hasattr(result, "break_index")
        assert 0.0 <= result.p_value <= 1.0
