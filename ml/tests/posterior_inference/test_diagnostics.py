import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    CalibrationRecommendation,
    CalibrationReport,
    DynamicsReport,
    SmoothingRecommendation,
    posterior_calibration_report,
    recommend_calibration,
    recommend_smoothing,
    summarize_posterior_dynamics,
)


def _alternating_probs(T: int, K: int = 3) -> np.ndarray:
    p = np.zeros((T, K), dtype=float)
    for t in range(T):
        p[t, t % K] = 0.9
        for k in range(K):
            if k != t % K:
                p[t, k] = 0.1 / (K - 1)
    return p


def _sticky_probs(T: int, K: int = 3, run_length: int = 10) -> np.ndarray:
    p = np.zeros((T, K), dtype=float)
    for t in range(T):
        winner = (t // run_length) % K
        p[t, winner] = 0.9
        for k in range(K):
            if k != winner:
                p[t, k] = 0.1 / (K - 1)
    return p


def _uniform_probs(T: int, K: int = 3) -> np.ndarray:
    return np.full((T, K), 1.0 / K, dtype=float)


# --- summarize_posterior_dynamics ---------------------------------------------

def test_summarize_returns_dynamics_report_with_shape_metadata() -> None:
    probs = _sticky_probs(60, K=3, run_length=15)
    report = summarize_posterior_dynamics(probs, window=10)

    assert isinstance(report, DynamicsReport)
    assert report.n_bars == 60
    assert report.n_states == 3
    assert report.window == 10


def test_summarize_without_transmat_leaves_conditional_fields_none() -> None:
    probs = _sticky_probs(40, K=3, run_length=10)
    report = summarize_posterior_dynamics(probs, window=5)

    assert report.expected_change_rate is None
    assert report.expected_dwell_length is None
    assert report.flip_rate_excess is None
    assert report.dwell_length_ratio is None


def test_summarize_sticky_posterior_has_low_flip_rate() -> None:
    probs = _sticky_probs(100, K=3, run_length=20)
    report = summarize_posterior_dynamics(probs, window=10)

    # 4 flips (at bars 20, 40, 60, 80) out of 99 lag-1 comparisons.
    assert report.mean_flip_rate == pytest.approx(4.0 / 99.0, abs=1e-12)


def test_summarize_alternating_posterior_has_unit_flip_rate() -> None:
    probs = _alternating_probs(30, K=3)
    report = summarize_posterior_dynamics(probs, window=5)

    assert report.mean_flip_rate == pytest.approx(1.0, abs=1e-12)


def test_summarize_transmat_unlocks_expected_fields_for_uniform_chain() -> None:
    probs = _sticky_probs(60, K=3, run_length=10)
    # Uniform transition matrix: stationary = [1/3, 1/3, 1/3], a_ii = 1/3.
    a = np.full((3, 3), 1.0 / 3.0)
    report = summarize_posterior_dynamics(probs, window=10, transmat=a)

    assert report.expected_change_rate == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert report.expected_dwell_length == pytest.approx(1.5, abs=1e-12)
    assert report.flip_rate_excess == pytest.approx(report.mean_flip_rate - 2.0 / 3.0, abs=1e-12)
    assert report.dwell_length_ratio == pytest.approx(report.mean_dwell_length / 1.5, abs=1e-12)


def test_summarize_transmat_stationary_for_sticky_chain() -> None:
    probs = _sticky_probs(60, K=2, run_length=10)
    # Symmetric two-state chain with stickiness 0.95: stationary = [0.5, 0.5], dwell = 1/0.05 = 20.
    a = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    report = summarize_posterior_dynamics(probs, window=10, transmat=a)

    assert report.expected_change_rate == pytest.approx(0.05, abs=1e-12)
    assert report.expected_dwell_length == pytest.approx(20.0, abs=1e-12)


def test_summarize_rejects_absorbing_transmat() -> None:
    probs = _sticky_probs(40, K=2, run_length=10)
    a = np.array([[1.0, 0.0], [0.05, 0.95]], dtype=float)

    with pytest.raises(ValueError, match="absorbing state"):
        summarize_posterior_dynamics(probs, window=5, transmat=a)


def test_summarize_rejects_transmat_with_wrong_shape() -> None:
    probs = _sticky_probs(40, K=3, run_length=10)
    a = np.full((2, 2), 0.5)

    with pytest.raises(ValueError, match=r"must have shape \(3, 3\)"):
        summarize_posterior_dynamics(probs, window=5, transmat=a)


def test_summarize_rejects_non_row_stochastic_transmat() -> None:
    probs = _sticky_probs(40, K=2, run_length=10)
    a = np.array([[0.8, 0.1], [0.1, 0.9]], dtype=float)

    with pytest.raises(ValueError, match="rows must sum to 1"):
        summarize_posterior_dynamics(probs, window=5, transmat=a)


def test_summarize_rejects_short_series() -> None:
    probs = np.array([[0.9, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="at least 2 bars"):
        summarize_posterior_dynamics(probs, window=5)


def test_summarize_rejects_bad_window() -> None:
    probs = _sticky_probs(20, K=2, run_length=5)

    with pytest.raises(ValueError, match="window must be >= 1"):
        summarize_posterior_dynamics(probs, window=0)


# --- posterior_calibration_report --------------------------------------------

def test_calibration_returns_report_with_shape_metadata() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=120)
    y = rng.integers(0, 4, size=120)
    report = posterior_calibration_report(probs, y, n_bins=8)

    assert isinstance(report, CalibrationReport)
    assert report.n_samples == 120
    assert report.n_classes == 4
    assert report.n_bins == 8
    assert report.per_class_ece.shape == (4,)
    assert report.per_class_brier.shape == (4,)
    assert report.reliability_bin_centers.shape == (8,)
    assert report.reliability_predicted_confidence.shape == (8,)
    assert report.reliability_empirical_accuracy.shape == (8,)
    assert report.reliability_bin_counts.shape == (8,)
    assert report.top_prob_histogram.shape == (8,)
    assert report.top_prob_bin_edges.shape == (9,)


def test_calibration_perfect_classifier_has_zero_ece_and_zero_nll() -> None:
    # One-hot posteriors agreeing with the labels — perfect calibration, zero NLL.
    K = 3
    T = 30
    y = np.tile(np.arange(K), T // K)
    probs = np.full((T, K), 1e-12, dtype=float)
    probs[np.arange(T), y] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    report = posterior_calibration_report(probs, y, n_bins=10)

    assert report.ece == pytest.approx(0.0, abs=1e-6)
    assert report.nll == pytest.approx(0.0, abs=1e-9)
    assert report.brier_score == pytest.approx(0.0, abs=1e-12)


def test_calibration_per_class_ece_arrays_immutable() -> None:
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=50)
    y = rng.integers(0, 3, size=50)
    report = posterior_calibration_report(probs, y, n_bins=5)

    with pytest.raises(ValueError):
        report.per_class_ece[0] = -1.0
    with pytest.raises(ValueError):
        report.reliability_bin_counts[0] = 999


def test_calibration_dispersion_is_nonnegative() -> None:
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=80)
    y = rng.integers(0, 3, size=80)
    report = posterior_calibration_report(probs, y, n_bins=5)

    assert report.ece_class_dispersion >= 0.0


def test_calibration_brier_decomposition_recomposes_exactly_when_forecasts_are_bin_constant() -> None:
    # Murphy's decomposition is exact only when forecasts within each bin are constant. Construct a
    # synthetic posterior whose top_prob takes exactly one value per bin so the within-bin forecast
    # variance is zero, then check that reliability - resolution + uncertainty == binary Brier exactly.
    n_bins = 5
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins  # [0.1, 0.3, 0.5, 0.7, 0.9]
    rng = np.random.default_rng(3)
    samples_per_bin = 40
    K = 2
    T = n_bins * samples_per_bin
    probs = np.zeros((T, K), dtype=float)
    y = np.zeros(T, dtype=np.int64)
    for b, center in enumerate(bin_centers):
        idx = slice(b * samples_per_bin, (b + 1) * samples_per_bin)
        probs[idx, 0] = center
        probs[idx, 1] = 1.0 - center
        # Empirical accuracy in bin b chosen to differ from forecast so reliability is non-trivial.
        target_acc = float(np.clip(center - 0.1, 0.0, 1.0))
        n_correct = int(round(target_acc * samples_per_bin))
        y[idx] = 1  # default: argmax (class 0) is wrong
        y[b * samples_per_bin: b * samples_per_bin + n_correct] = 0
    report = posterior_calibration_report(probs, y, n_bins=n_bins)

    top_prob = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    binary_brier = float(np.mean((top_prob - (pred == y).astype(float)) ** 2))
    recomposed = report.brier_reliability - report.brier_resolution + report.brier_uncertainty

    assert recomposed == pytest.approx(binary_brier, abs=1e-12)


def test_calibration_brier_decomposition_components_are_nonnegative() -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(3), size=200)
    y = rng.integers(0, 3, size=200)
    report = posterior_calibration_report(probs, y, n_bins=10)

    assert report.brier_reliability >= 0.0
    assert report.brier_resolution >= 0.0
    assert 0.0 <= report.brier_uncertainty <= 0.25


def test_calibration_rejects_single_class_labels() -> None:
    probs = np.full((20, 3), 1.0 / 3.0, dtype=float)
    y = np.zeros(20, dtype=np.int64)

    with pytest.raises(ValueError, match="only one class"):
        posterior_calibration_report(probs, y, n_bins=5)


def test_calibration_rejects_length_mismatch() -> None:
    probs = np.full((10, 3), 1.0 / 3.0, dtype=float)
    y = np.zeros(11, dtype=np.int64)

    with pytest.raises(ValueError, match="length must equal"):
        posterior_calibration_report(probs, y, n_bins=5)


def test_calibration_rejects_out_of_range_label() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=20)
    y = np.array([0, 1, 2, 5] * 5, dtype=np.int64)

    with pytest.raises(ValueError, match=r"y_idx must be in \[0, K-1\]"):
        posterior_calibration_report(probs, y, n_bins=5)


def test_calibration_rejects_too_few_bins() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=20)
    y = rng.integers(0, 3, size=20)

    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        posterior_calibration_report(probs, y, n_bins=1)


# --- recommend_smoothing ------------------------------------------------------

def test_recommend_smoothing_unknown_without_transmat() -> None:
    probs = _sticky_probs(50, K=2, run_length=10)
    report = summarize_posterior_dynamics(probs, window=5)

    assert recommend_smoothing(report) == SmoothingRecommendation.UNKNOWN


def test_recommend_smoothing_none_when_observed_matches_expected() -> None:
    # Simulate a true 2-state Markov chain so the dwell distribution is geometric, matching the
    # transmat-derived expected dwell formula. Deterministic-length runs would understate per-bar
    # dwell vs. its geometric reference and falsely trigger a smoothing recommendation.
    rng = np.random.default_rng(0)
    T = 2000
    flip_prob = 0.1
    state = 0
    states = np.empty(T, dtype=np.int64)
    for t in range(T):
        if t > 0 and rng.random() < flip_prob:
            state = 1 - state
        states[t] = state
    K = 2
    probs = np.full((T, K), 0.05, dtype=float)
    probs[np.arange(T), states] = 0.95
    a = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    report = summarize_posterior_dynamics(probs, window=10, transmat=a)

    assert recommend_smoothing(report) == SmoothingRecommendation.NONE


def test_recommend_smoothing_heavier_when_observed_flips_more_than_expected() -> None:
    probs = _alternating_probs(60, K=3)  # flip_rate = 1.0
    a = np.full((3, 3), 1.0 / 3.0)        # expected_change_rate = 2/3
    report = summarize_posterior_dynamics(probs, window=5, transmat=a)

    verdict = recommend_smoothing(report, flip_rate_excess_threshold=0.05)
    assert verdict in {SmoothingRecommendation.MODERATE, SmoothingRecommendation.HEAVY}


def test_recommend_smoothing_rejects_non_positive_thresholds() -> None:
    probs = _sticky_probs(50, K=2, run_length=10)
    a = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    report = summarize_posterior_dynamics(probs, window=5, transmat=a)

    with pytest.raises(ValueError, match="flip_rate_excess_threshold"):
        recommend_smoothing(report, flip_rate_excess_threshold=0.0)
    with pytest.raises(ValueError, match="dwell_ratio_threshold"):
        recommend_smoothing(report, dwell_ratio_threshold=0.0)


# --- recommend_calibration ----------------------------------------------------

def test_recommend_calibration_none_for_well_calibrated_posterior() -> None:
    K = 3
    T = 60
    y = np.tile(np.arange(K), T // K)
    probs = np.full((T, K), 1e-12, dtype=float)
    probs[np.arange(T), y] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    report = posterior_calibration_report(probs, y, n_bins=10)

    assert recommend_calibration(report) == CalibrationRecommendation.NONE


def test_recommend_calibration_temperature_when_uniform_miscalibration() -> None:
    # Over-confident across classes uniformly: predict 0.9 for correct, 0.05 for others; accuracy is only 50%.
    K = 2
    T = 200
    y = np.zeros(T, dtype=np.int64)
    y[T // 2:] = 1
    probs = np.empty((T, K), dtype=float)
    rng = np.random.default_rng(0)
    correct = rng.random(T) < 0.5
    for t in range(T):
        winner = int(y[t]) if correct[t] else int(1 - y[t])
        probs[t, winner] = 0.9
        probs[t, 1 - winner] = 0.1
    report = posterior_calibration_report(probs, y, n_bins=10)

    verdict = recommend_calibration(report, ece_threshold=0.05, dispersion_threshold=0.3)
    assert verdict == CalibrationRecommendation.TEMPERATURE


def test_recommend_calibration_platt_when_class_dependent_miscalibration() -> None:
    # Construct a report directly to exercise the dispersion branch.
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=100)
    y = rng.integers(0, 3, size=100)
    report = posterior_calibration_report(probs, y, n_bins=10)
    # Synthetically construct a report with high overall ECE AND high dispersion via dataclasses.replace pattern.
    from dataclasses import replace
    synthetic = replace(report, ece=0.20, ece_class_dispersion=0.10)

    assert recommend_calibration(synthetic, ece_threshold=0.05, dispersion_threshold=0.05) == \
        CalibrationRecommendation.PLATT


def test_recommend_calibration_rejects_non_positive_thresholds() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=30)
    y = rng.integers(0, 3, size=30)
    report = posterior_calibration_report(probs, y, n_bins=5)

    with pytest.raises(ValueError, match="ece_threshold"):
        recommend_calibration(report, ece_threshold=0.0)
    with pytest.raises(ValueError, match="dispersion_threshold"):
        recommend_calibration(report, dispersion_threshold=0.0)


# --- Review fixes -------------------------------------------------------------

def test_dynamics_flip_autocorr_finite_when_lagged_slice_constant() -> None:
    # argmax sequence [0, 1, 1, 1, 1, 1, ...] yields flips = [1, 0, 0, 0, 0, ...] of length T-1.
    # left  = flips[:-1] = [1, 0, 0, 0, ...]  (not constant)
    # right = flips[1:]  = [0, 0, 0, 0, ...]  (constant)
    # The original full-series std guard would not trip — np.corrcoef returns NaN. The fixed guard
    # checks both slices independently and returns 0 instead.
    T = 15
    K = 2
    probs = np.full((T, K), 0.1, dtype=float)
    probs[0, 0] = 0.9
    probs[1:, 1] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    report = summarize_posterior_dynamics(probs, window=5)

    assert np.isfinite(report.flip_autocorr_lag1)
    assert report.flip_autocorr_lag1 == 0.0


def test_calibration_rejects_float_y_idx_to_prevent_truncation() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=30)
    y_float = np.array([0.9, 1.1] * 15, dtype=float)

    with pytest.raises(ValueError, match="must be an integer array"):
        posterior_calibration_report(probs, y_float, n_bins=5)


def test_dynamics_rejects_row_sum_drift() -> None:
    # Non-negative, finite, but not a simplex: rows sum to 1.5.
    bad_probs = np.full((20, 2), 0.75, dtype=float)

    with pytest.raises(ValueError, match="must sum to 1"):
        summarize_posterior_dynamics(bad_probs, window=5)


def test_calibration_rejects_row_sum_drift() -> None:
    bad_probs = np.full((20, 3), 0.5, dtype=float)
    y = np.tile(np.arange(3), 7)[:20]

    with pytest.raises(ValueError, match="must sum to 1"):
        posterior_calibration_report(bad_probs, y, n_bins=5)


def test_summarize_rejects_reducible_transmat() -> None:
    probs = _sticky_probs(40, K=2, run_length=10)
    # Identity matrix: two closed communicating classes, eigenvalue 1 has multiplicity 2.
    a = np.eye(2, dtype=float)

    # Absorbing-state check fires first under the current ordering — both are correct rejections, but
    # for the multiplicity-of-1 path we need a non-absorbing reducible chain. Use a 4-state block
    # matrix: two independent sticky 2-state chains.
    a = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=float,
    )
    probs4 = _sticky_probs(40, K=4, run_length=10)

    with pytest.raises(ValueError, match="reducible"):
        summarize_posterior_dynamics(probs4, window=5, transmat=a)
