import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    PosteriorHealthBaselines,
    PosteriorHealthReport,
    entropy_staleness,
    fit_posterior_health_baselines,
    flip_rate_drift,
    score_posterior_health,
    state_occupancy_drift,
)


def _uniform_probs(T: int, K: int) -> np.ndarray:
    return np.full((T, K), 1.0 / K, dtype=float)


def _random_simplex(T: int, K: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(K), size=T)


# --- entropy_staleness --------------------------------------------------------

def test_entropy_staleness_returns_zero_when_series_matches_baseline_mean() -> None:
    e = np.full(50, 1.0, dtype=float)
    out = entropy_staleness(e, baseline_mean=1.0, baseline_std=0.1, window=10)

    assert np.isnan(out[:9]).all()
    np.testing.assert_allclose(out[9:], 0.0, atol=1e-12)


def test_entropy_staleness_scales_by_baseline_std() -> None:
    e = np.full(20, 2.0, dtype=float)
    out = entropy_staleness(e, baseline_mean=1.0, baseline_std=0.5, window=5)

    np.testing.assert_allclose(out[4:], 2.0, atol=1e-12)


def test_entropy_staleness_warmup_mask_length_equals_window_minus_one() -> None:
    e = np.arange(30, dtype=float)
    out = entropy_staleness(e, baseline_mean=10.0, baseline_std=1.0, window=7)

    assert np.isnan(out[:6]).all()
    assert not np.isnan(out[6:]).any()


def test_entropy_staleness_window_one_emits_no_warmup() -> None:
    e = np.array([1.0, 2.0, 3.0], dtype=float)
    out = entropy_staleness(e, baseline_mean=0.0, baseline_std=1.0, window=1)

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]), atol=1e-12)


def test_entropy_staleness_rejects_non_positive_baseline_std() -> None:
    e = np.ones(10, dtype=float)
    with pytest.raises(ValueError, match="baseline_std must be positive"):
        entropy_staleness(e, baseline_mean=1.0, baseline_std=0.0, window=3)


def test_entropy_staleness_rejects_non_1d_series() -> None:
    e = np.ones((5, 3), dtype=float)
    with pytest.raises(ValueError, match="must be 1-D"):
        entropy_staleness(e, baseline_mean=1.0, baseline_std=1.0, window=3)


def test_entropy_staleness_rejects_bad_window() -> None:
    e = np.ones(5, dtype=float)
    with pytest.raises(ValueError, match="window must be >= 1"):
        entropy_staleness(e, baseline_mean=1.0, baseline_std=1.0, window=0)


# --- state_occupancy_drift ----------------------------------------------------

def test_state_occupancy_drift_returns_zero_when_observed_matches_baseline() -> None:
    probs = _uniform_probs(40, 3)
    baseline = np.full(3, 1.0 / 3, dtype=float)
    out = state_occupancy_drift(probs, baseline, window=8)

    assert np.isnan(out[:7]).all()
    np.testing.assert_allclose(out[7:], 0.0, atol=1e-12)


def test_state_occupancy_drift_computes_l1_correctly_in_steady_state() -> None:
    T = 15
    probs = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (T, 1))
    baseline = np.array([0.5, 0.5, 0.0], dtype=float)
    out = state_occupancy_drift(probs, baseline, window=5)

    np.testing.assert_allclose(out[4:], 1.0, atol=1e-12)


def test_state_occupancy_drift_warmup_mask_length_equals_window_minus_one() -> None:
    probs = _random_simplex(30, 3, seed=1)
    baseline = probs.mean(axis=0)
    out = state_occupancy_drift(probs, baseline, window=10)

    assert np.isnan(out[:9]).all()
    assert not np.isnan(out[9:]).any()


def test_state_occupancy_drift_rejects_baseline_shape_mismatch() -> None:
    probs = _uniform_probs(10, 3)
    bad = np.array([0.5, 0.5], dtype=float)
    with pytest.raises(ValueError, match=r"must have shape \(3,\)"):
        state_occupancy_drift(probs, bad, window=3)


def test_state_occupancy_drift_rejects_negative_baseline_components() -> None:
    probs = _uniform_probs(10, 3)
    bad = np.array([0.6, 0.5, -0.1], dtype=float)
    with pytest.raises(ValueError, match="must be non-negative"):
        state_occupancy_drift(probs, bad, window=3)


def test_state_occupancy_drift_rejects_baseline_not_summing_to_one() -> None:
    probs = _uniform_probs(10, 3)
    bad = np.array([0.5, 0.3, 0.3], dtype=float)
    with pytest.raises(ValueError, match="must sum to 1"):
        state_occupancy_drift(probs, bad, window=3)


def test_state_occupancy_drift_handles_empty_input() -> None:
    probs = np.zeros((0, 3), dtype=float)
    baseline = np.full(3, 1.0 / 3, dtype=float)

    out = state_occupancy_drift(probs, baseline, window=3)

    assert out.shape == (0,)


# --- flip_rate_drift ----------------------------------------------------------

def test_flip_rate_drift_returns_zero_when_observed_matches_baseline() -> None:
    probs = np.tile(np.array([0.7, 0.2, 0.1], dtype=float), (20, 1))
    out = flip_rate_drift(probs, baseline_flip_rate=0.0, window=5)

    assert np.isnan(out[:4]).all()
    np.testing.assert_allclose(out[4:], 0.0, atol=1e-12)


def test_flip_rate_drift_signed_when_observed_exceeds_baseline() -> None:
    probs = np.array([[0.6, 0.4], [0.4, 0.6]] * 10, dtype=float)
    out = flip_rate_drift(probs, baseline_flip_rate=0.2, window=5)

    assert out[-1] == pytest.approx(0.8)


def test_flip_rate_drift_rejects_baseline_below_zero() -> None:
    probs = _uniform_probs(10, 3)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        flip_rate_drift(probs, baseline_flip_rate=-0.1, window=3)


def test_flip_rate_drift_rejects_baseline_above_one() -> None:
    probs = _uniform_probs(10, 3)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        flip_rate_drift(probs, baseline_flip_rate=1.5, window=3)


# --- fit_posterior_health_baselines -------------------------------------------

def test_fit_posterior_health_baselines_returns_dataclass_with_expected_shape() -> None:
    probs = _random_simplex(200, 3, seed=2)

    b = fit_posterior_health_baselines(probs, window=10)

    assert isinstance(b, PosteriorHealthBaselines)
    assert b.window == 10
    assert b.occupancy.shape == (3,)
    np.testing.assert_allclose(b.occupancy.sum(), 1.0, atol=1e-6)
    assert b.entropy_std > 0.0
    assert 0.0 <= b.flip_rate <= 1.0


def test_fit_posterior_health_baselines_occupancy_equals_column_mean() -> None:
    probs = _random_simplex(100, 4, seed=3)

    b = fit_posterior_health_baselines(probs, window=5)

    np.testing.assert_allclose(b.occupancy, probs.mean(axis=0), atol=1e-12)


def test_fit_posterior_health_baselines_rejects_too_few_rows() -> None:
    probs = _uniform_probs(10, 3)

    with pytest.raises(ValueError, match=r"at least window\+1"):
        fit_posterior_health_baselines(probs, window=10)


def test_fit_posterior_health_baselines_is_frozen() -> None:
    probs = _random_simplex(50, 3, seed=4)
    b = fit_posterior_health_baselines(probs, window=5)

    with pytest.raises(Exception):
        b.window = 99  # type: ignore[misc]


def test_fit_posterior_health_baselines_occupancy_is_read_only() -> None:
    probs = _random_simplex(50, 3, seed=10)
    b = fit_posterior_health_baselines(probs, window=5)

    with pytest.raises(ValueError, match="read-only|assignment destination"):
        b.occupancy[0] = 0.99


def test_fit_posterior_health_baselines_rejects_zero_entropy_variance() -> None:
    # Constant posteriors -> constant entropy -> zero stdev of rolling-mean entropy.
    # Must raise at fit so the failure surfaces with the upstream model, not later
    # at the first score_posterior_health call.
    constant = np.tile(np.array([0.33, 0.33, 0.34], dtype=float), (200, 1))

    with pytest.raises(ValueError, match="zero entropy variance"):
        fit_posterior_health_baselines(constant, window=10)


# --- score_posterior_health ---------------------------------------------------

def test_score_posterior_health_fit_then_score_passes_on_training_data() -> None:
    probs = _random_simplex(3000, 3, seed=4)
    b = fit_posterior_health_baselines(probs, window=30)

    r = score_posterior_health(probs, b, max_entropy_abs_z=5.0, max_occupancy_drift_l1=0.5,
                               max_flip_rate_drift_abs=0.3)

    assert isinstance(r, PosteriorHealthReport)
    assert r.overall_ok is True
    assert not np.isnan(r.entropy_staleness_z)
    assert not np.isnan(r.occupancy_drift_l1)
    assert not np.isnan(r.flip_rate_drift_signed)


def test_score_posterior_health_detects_occupancy_shift() -> None:
    train = _random_simplex(500, 3, seed=5)
    b = fit_posterior_health_baselines(train, window=20)
    bad = np.tile(np.array([0.9, 0.05, 0.05], dtype=float), (50, 1))

    r = score_posterior_health(bad, b)

    assert r.occupancy_drift_ok is False
    assert r.overall_ok is False


def test_score_posterior_health_detects_flip_rate_collapse() -> None:
    train = _random_simplex(500, 3, seed=6)
    b = fit_posterior_health_baselines(train, window=20)
    # Frozen-argmax inference series (no flips) should diverge from random train flip-rate
    frozen = np.tile(np.array([0.6, 0.3, 0.1], dtype=float), (50, 1))

    r = score_posterior_health(frozen, b)

    assert r.flip_rate_drift_ok is False
    assert r.flip_rate_drift_signed < 0.0


def test_score_posterior_health_rejects_too_few_rows() -> None:
    probs = _random_simplex(200, 3, seed=7)
    b = fit_posterior_health_baselines(probs, window=20)
    too_short = probs[:10]

    with pytest.raises(ValueError, match="at least window=20"):
        score_posterior_health(too_short, b)


def test_score_posterior_health_overall_ok_is_and_of_components() -> None:
    probs = _random_simplex(500, 3, seed=8)
    b = fit_posterior_health_baselines(probs, window=20)

    r = score_posterior_health(probs, b)

    expected = r.entropy_staleness_ok and r.occupancy_drift_ok and r.flip_rate_drift_ok
    assert r.overall_ok == expected


def test_score_posterior_health_uses_final_row_of_inputs() -> None:
    # Append a sharp outlier as the last row; the final-row metrics should react.
    rng = np.random.default_rng(9)
    train = rng.dirichlet(np.ones(3), size=500)
    b = fit_posterior_health_baselines(train, window=20)
    # Build a 30-row inference frame: 29 in-distribution + 1 sharp outlier at the end
    in_dist = rng.dirichlet(np.ones(3), size=29)
    tail = np.array([[0.97, 0.02, 0.01]], dtype=float)
    probe = np.vstack([in_dist, tail])

    r_with_tail = score_posterior_health(probe, b)
    r_without_tail = score_posterior_health(in_dist, b)

    # The outlier should push the final-row occupancy further from baseline.
    assert r_with_tail.occupancy_drift_l1 >= r_without_tail.occupancy_drift_l1
