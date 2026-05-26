import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    FeatureHealthBaselines,
    FeatureHealthReport,
    LoglikDriftBaselines,
    LoglikDriftReport,
    PosteriorHealthBaselines,
    PosteriorHealthReport,
    RefitAuditReport,
    RefitMetricVerdict,
    audit_refit_metrics,
    entropy_staleness,
    feature_ks_drift,
    fit_feature_health_baselines,
    fit_loglik_drift_baselines,
    fit_posterior_health_baselines,
    flip_rate_drift,
    log_likelihood_drift,
    score_feature_health,
    score_loglik_health,
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
    # window=10 requires at least 12 rows (window + 2 for post-warmup stdev to be meaningful).
    probs = _uniform_probs(11, 3)

    with pytest.raises(ValueError, match=r"at least window\+2"):
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


def test_posterior_health_baselines_round_trip_via_dict() -> None:
    probs = _random_simplex(500, 3, seed=300)
    original = fit_posterior_health_baselines(probs, window=20)

    payload = original.to_dict()
    restored = PosteriorHealthBaselines.from_dict(payload)

    assert restored.window == original.window
    assert restored.entropy_mean == pytest.approx(original.entropy_mean, abs=1e-15)
    assert restored.entropy_std == pytest.approx(original.entropy_std, abs=1e-15)
    assert restored.flip_rate == pytest.approx(original.flip_rate, abs=1e-15)
    np.testing.assert_allclose(restored.occupancy, original.occupancy, atol=1e-15)


def test_posterior_health_baselines_to_dict_emits_plain_python_types() -> None:
    probs = _random_simplex(500, 3, seed=301)
    b = fit_posterior_health_baselines(probs, window=20)

    payload = b.to_dict()

    assert isinstance(payload["window"], int)
    assert isinstance(payload["entropy_mean"], float)
    assert isinstance(payload["entropy_std"], float)
    assert isinstance(payload["flip_rate"], float)
    assert isinstance(payload["occupancy"], list)
    assert all(isinstance(x, float) for x in payload["occupancy"])


def test_posterior_health_baselines_from_dict_rejects_missing_keys() -> None:
    payload = {"window": 20, "entropy_mean": 1.0, "entropy_std": 0.1, "flip_rate": 0.05}  # no occupancy
    with pytest.raises(ValueError, match=r"missing required keys \['occupancy'\]"):
        PosteriorHealthBaselines.from_dict(payload)


def test_posterior_health_baselines_from_dict_occupancy_is_read_only() -> None:
    payload = {
        "window": 20,
        "entropy_mean": 1.0,
        "entropy_std": 0.1,
        "occupancy": [0.33, 0.33, 0.34],
        "flip_rate": 0.05,
    }

    b = PosteriorHealthBaselines.from_dict(payload)

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


# --- fit_feature_health_baselines ---------------------------------------------

def _gaussian_features(T: int, n_features: int, seed: int = 0, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc=loc, scale=scale, size=(T, n_features))


def test_fit_feature_health_baselines_returns_dataclass_with_expected_shape() -> None:
    X = _gaussian_features(1000, 3, seed=20)

    b = fit_feature_health_baselines(X)

    assert isinstance(b, FeatureHealthBaselines)
    assert b.samples.shape == (1000, 3)
    assert b.feature_names == ("feature_0", "feature_1", "feature_2")


def test_fit_feature_health_baselines_accepts_custom_feature_names() -> None:
    X = _gaussian_features(100, 3, seed=21)

    b = fit_feature_health_baselines(X, feature_names=["tsi", "dbl_smoothed_log_rets", "smoothed_atr"])

    assert b.feature_names == ("tsi", "dbl_smoothed_log_rets", "smoothed_atr")


def test_fit_feature_health_baselines_rejects_mismatched_feature_names_length() -> None:
    X = _gaussian_features(100, 3, seed=22)

    with pytest.raises(ValueError, match="feature_names has length 2 but X has 3 columns"):
        fit_feature_health_baselines(X, feature_names=["a", "b"])


def test_fit_feature_health_baselines_rejects_too_few_rows() -> None:
    X = _gaussian_features(1, 3, seed=23)

    with pytest.raises(ValueError, match="at least 2 rows"):
        fit_feature_health_baselines(X)


def test_fit_feature_health_baselines_rejects_zero_columns() -> None:
    X = np.zeros((100, 0), dtype=float)

    with pytest.raises(ValueError, match="n_features >= 1"):
        fit_feature_health_baselines(X)


def test_fit_feature_health_baselines_rejects_nan() -> None:
    X = _gaussian_features(100, 3, seed=24)
    X[10, 1] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_feature_health_baselines(X)


def test_fit_feature_health_baselines_is_frozen() -> None:
    X = _gaussian_features(100, 2, seed=25)
    b = fit_feature_health_baselines(X)

    with pytest.raises(Exception):
        b.samples = np.zeros((1, 1))  # type: ignore[misc]


def test_fit_feature_health_baselines_samples_are_read_only() -> None:
    X = _gaussian_features(100, 2, seed=26)
    b = fit_feature_health_baselines(X)

    with pytest.raises(ValueError, match="read-only|assignment destination"):
        b.samples[0, 0] = 99.0


def test_fit_feature_health_baselines_copies_input() -> None:
    # Mutating the caller's array after fit must not affect the stored baseline.
    X = _gaussian_features(100, 2, seed=27)
    b = fit_feature_health_baselines(X)
    original = b.samples[0, 0]
    # The stored array is read-only; mutating the caller's array should not change it.
    X[0, 0] = 999.0

    assert b.samples[0, 0] == original


# --- feature_ks_drift ---------------------------------------------------------

def test_feature_ks_drift_returns_expected_shapes() -> None:
    base = _gaussian_features(2000, 4, seed=30)
    window = _gaussian_features(200, 4, seed=31)
    b = fit_feature_health_baselines(base)

    stats, pvals = feature_ks_drift(window, b)

    assert stats.shape == (4,)
    assert pvals.shape == (4,)


def test_feature_ks_drift_small_when_window_drawn_from_baseline_distribution() -> None:
    base = _gaussian_features(5000, 3, seed=40)
    window = _gaussian_features(500, 3, seed=41)
    b = fit_feature_health_baselines(base)

    stats, pvals = feature_ks_drift(window, b)

    # Two independent draws from N(0, 1) should produce small KS statistics.
    assert (stats < 0.1).all()


def test_feature_ks_drift_detects_location_shift() -> None:
    base = _gaussian_features(5000, 3, seed=50)
    # Shift feature 1 by 2 standard deviations; features 0 and 2 unchanged.
    window = _gaussian_features(500, 3, seed=51)
    window[:, 1] += 2.0
    b = fit_feature_health_baselines(base)

    stats, pvals = feature_ks_drift(window, b)

    # Shifted feature must show large KS statistic and tiny p-value.
    assert stats[1] > 0.5
    assert pvals[1] < 1e-10
    # Unshifted features should remain small.
    assert stats[0] < 0.1
    assert stats[2] < 0.1


def test_feature_ks_drift_rejects_column_count_mismatch() -> None:
    base = _gaussian_features(500, 3, seed=60)
    bad_window = _gaussian_features(100, 4, seed=61)
    b = fit_feature_health_baselines(base)

    with pytest.raises(ValueError, match="X_window has 4 columns but baselines has 3"):
        feature_ks_drift(bad_window, b)


def test_feature_ks_drift_rejects_window_with_fewer_than_two_rows() -> None:
    base = _gaussian_features(500, 2, seed=70)
    b = fit_feature_health_baselines(base)
    too_short = _gaussian_features(1, 2, seed=71)

    with pytest.raises(ValueError, match="X_window must have at least 2 rows"):
        feature_ks_drift(too_short, b)


# --- score_feature_health -----------------------------------------------------

def test_score_feature_health_passes_when_window_drawn_from_baseline() -> None:
    base = _gaussian_features(5000, 3, seed=80)
    window = _gaussian_features(500, 3, seed=81)
    b = fit_feature_health_baselines(base, feature_names=["f0", "f1", "f2"])

    r = score_feature_health(window, b)

    assert isinstance(r, FeatureHealthReport)
    assert r.overall_ok is True
    assert r.per_feature_ok.all()
    assert r.feature_names == ("f0", "f1", "f2")


def test_score_feature_health_detects_when_both_effect_and_significance_fire() -> None:
    base = _gaussian_features(5000, 3, seed=90)
    window = _gaussian_features(500, 3, seed=91)
    # Large shift on feature 0 — significant AND large effect.
    window[:, 0] += 1.5
    b = fit_feature_health_baselines(base)

    r = score_feature_health(window, b, max_ks_statistic=0.1, alpha=0.01)

    assert r.overall_ok is False
    assert r.per_feature_ok[0] is np.False_ or not bool(r.per_feature_ok[0])
    assert bool(r.per_feature_ok[1])
    assert bool(r.per_feature_ok[2])


def test_score_feature_health_does_not_alarm_on_tiny_effect_even_when_significant() -> None:
    # With huge sample sizes on both sides, KS p-values can be tiny for trivial
    # distributional differences. The effect-size gate must filter these out.
    rng = np.random.default_rng(100)
    base = rng.normal(loc=0.0, scale=1.0, size=(50000, 2))
    # Microscopic location shift — KS statistic will be small but p-value can be tiny.
    window = rng.normal(loc=0.001, scale=1.0, size=(10000, 2))
    b = fit_feature_health_baselines(base)

    r = score_feature_health(window, b, max_ks_statistic=0.1, alpha=0.01)

    # Effect size should be well under 0.1 → gate passes regardless of p-value.
    assert (r.ks_statistics < 0.1).all()
    assert r.overall_ok is True


def test_score_feature_health_does_not_alarm_on_underpowered_window() -> None:
    # With tiny windows, even visibly different distributions can fail to reach
    # significance. The p-value gate must filter these out (no alarm when not
    # significant), even if the empirical KS statistic exceeds the effect-size
    # threshold by chance.
    rng = np.random.default_rng(110)
    base = rng.normal(loc=0.0, scale=1.0, size=(50000, 2))
    tiny_window = rng.normal(loc=0.0, scale=1.0, size=(5, 2))
    b = fit_feature_health_baselines(base)

    r = score_feature_health(tiny_window, b, max_ks_statistic=0.05, alpha=1e-6)

    # alpha=1e-6 ensures no feature reaches significance with n=5 → all pass.
    assert r.overall_ok is True


def test_score_feature_health_overall_ok_is_strict_and_of_per_feature() -> None:
    base = _gaussian_features(5000, 4, seed=120)
    window = _gaussian_features(500, 4, seed=121)
    # Shift only feature 2.
    window[:, 2] += 1.5
    b = fit_feature_health_baselines(base)

    r = score_feature_health(window, b)

    assert r.overall_ok is False
    assert bool(r.per_feature_ok.all()) == r.overall_ok
    assert int((~r.per_feature_ok).sum()) == 1


def test_score_feature_health_rejects_bad_max_ks_statistic() -> None:
    base = _gaussian_features(500, 2, seed=130)
    window = _gaussian_features(100, 2, seed=131)
    b = fit_feature_health_baselines(base)

    with pytest.raises(ValueError, match=r"max_ks_statistic must be in \[0, 1\]"):
        score_feature_health(window, b, max_ks_statistic=1.5)


def test_score_feature_health_rejects_bad_alpha() -> None:
    base = _gaussian_features(500, 2, seed=140)
    window = _gaussian_features(100, 2, seed=141)
    b = fit_feature_health_baselines(base)

    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        score_feature_health(window, b, alpha=0.0)


def test_score_feature_health_report_arrays_are_read_only() -> None:
    base = _gaussian_features(500, 2, seed=150)
    window = _gaussian_features(100, 2, seed=151)
    b = fit_feature_health_baselines(base)

    r = score_feature_health(window, b)

    with pytest.raises(ValueError, match="read-only|assignment destination"):
        r.ks_statistics[0] = 99.0
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        r.p_values[0] = 99.0
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        r.per_feature_ok[0] = False


# --- log_likelihood_drift -----------------------------------------------------

def _gaussian_loglik(T: int, seed: int = 0, loc: float = -2.0, scale: float = 0.5) -> np.ndarray:
    """Synthetic per-bar log-likelihood series: normal-distributed around ``loc``."""
    return np.random.default_rng(seed).normal(loc=loc, scale=scale, size=T)


def test_log_likelihood_drift_returns_zero_when_series_matches_baseline_mean() -> None:
    s = np.full(50, -1.5, dtype=float)
    out = log_likelihood_drift(s, baseline_mean=-1.5, baseline_std=0.1, window=10)

    assert np.isnan(out[:9]).all()
    np.testing.assert_allclose(out[9:], 0.0, atol=1e-12)


def test_log_likelihood_drift_scales_by_baseline_std() -> None:
    s = np.full(20, -3.0, dtype=float)
    out = log_likelihood_drift(s, baseline_mean=-2.0, baseline_std=0.5, window=5)

    # (-3.0 - (-2.0)) / 0.5 = -2.0
    np.testing.assert_allclose(out[4:], -2.0, atol=1e-12)


def test_log_likelihood_drift_warmup_mask_length_equals_window_minus_one() -> None:
    s = np.arange(30, dtype=float)
    out = log_likelihood_drift(s, baseline_mean=10.0, baseline_std=1.0, window=7)

    assert np.isnan(out[:6]).all()
    assert not np.isnan(out[6:]).any()


def test_log_likelihood_drift_window_one_emits_no_warmup() -> None:
    s = np.array([-1.0, -2.0, -3.0], dtype=float)
    out = log_likelihood_drift(s, baseline_mean=0.0, baseline_std=1.0, window=1)

    np.testing.assert_allclose(out, np.array([-1.0, -2.0, -3.0]), atol=1e-12)


def test_log_likelihood_drift_accepts_positive_values_for_continuous_densities() -> None:
    # Continuous-density emissions can produce positive log-density values.
    # No sign assumption should be enforced.
    s = np.full(20, 2.5, dtype=float)
    out = log_likelihood_drift(s, baseline_mean=2.0, baseline_std=0.25, window=5)

    np.testing.assert_allclose(out[4:], 2.0, atol=1e-12)


def test_log_likelihood_drift_rejects_non_positive_baseline_std() -> None:
    s = np.ones(10, dtype=float)
    with pytest.raises(ValueError, match="baseline_std must be positive"):
        log_likelihood_drift(s, baseline_mean=1.0, baseline_std=0.0, window=3)


def test_log_likelihood_drift_rejects_non_1d_series() -> None:
    s = np.ones((5, 3), dtype=float)
    with pytest.raises(ValueError, match="must be 1-D"):
        log_likelihood_drift(s, baseline_mean=1.0, baseline_std=1.0, window=3)


def test_log_likelihood_drift_rejects_nan() -> None:
    s = np.ones(10, dtype=float)
    s[3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        log_likelihood_drift(s, baseline_mean=1.0, baseline_std=1.0, window=3)


def test_log_likelihood_drift_rejects_bad_window() -> None:
    s = np.ones(5, dtype=float)
    with pytest.raises(ValueError, match="window must be >= 1"):
        log_likelihood_drift(s, baseline_mean=1.0, baseline_std=1.0, window=0)


# --- fit_loglik_drift_baselines -----------------------------------------------

def test_fit_loglik_drift_baselines_returns_dataclass_with_expected_shape() -> None:
    s = _gaussian_loglik(500, seed=200)

    b = fit_loglik_drift_baselines(s, window=20)

    assert isinstance(b, LoglikDriftBaselines)
    assert b.window == 20
    assert b.loglik_std > 0.0


def test_fit_loglik_drift_baselines_rejects_too_few_rows() -> None:
    s = _gaussian_loglik(11, seed=201)

    with pytest.raises(ValueError, match=r"at least window\+2"):
        fit_loglik_drift_baselines(s, window=10)


def test_fit_loglik_drift_baselines_rejects_near_zero_variance() -> None:
    constant = np.full(200, -2.0, dtype=float)

    with pytest.raises(ValueError, match="near-zero variance"):
        fit_loglik_drift_baselines(constant, window=10)


def test_fit_loglik_drift_baselines_is_frozen() -> None:
    s = _gaussian_loglik(200, seed=202)
    b = fit_loglik_drift_baselines(s, window=10)

    with pytest.raises(Exception):
        b.window = 99  # type: ignore[misc]


def test_loglik_drift_baselines_round_trip_via_dict() -> None:
    s = _gaussian_loglik(2000, seed=400)
    original = fit_loglik_drift_baselines(s, window=30)

    payload = original.to_dict()
    restored = LoglikDriftBaselines.from_dict(payload)

    assert restored.window == original.window
    assert restored.loglik_mean == pytest.approx(original.loglik_mean, abs=1e-15)
    assert restored.loglik_std == pytest.approx(original.loglik_std, abs=1e-15)


def test_loglik_drift_baselines_to_dict_emits_plain_python_types() -> None:
    s = _gaussian_loglik(2000, seed=401)
    b = fit_loglik_drift_baselines(s, window=30)

    payload = b.to_dict()

    assert isinstance(payload["window"], int)
    assert isinstance(payload["loglik_mean"], float)
    assert isinstance(payload["loglik_std"], float)


def test_loglik_drift_baselines_from_dict_rejects_missing_keys() -> None:
    payload = {"window": 30, "loglik_mean": -2.5}  # no loglik_std
    with pytest.raises(ValueError, match=r"missing required keys \['loglik_std'\]"):
        LoglikDriftBaselines.from_dict(payload)


def test_fit_loglik_drift_baselines_rejects_nan() -> None:
    s = _gaussian_loglik(200, seed=203)
    s[5] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_loglik_drift_baselines(s, window=10)


# --- score_loglik_health ------------------------------------------------------

def test_score_loglik_health_passes_on_training_data() -> None:
    s = _gaussian_loglik(2000, seed=210)
    b = fit_loglik_drift_baselines(s, window=30)

    r = score_loglik_health(s, b, max_abs_z=5.0)

    assert isinstance(r, LoglikDriftReport)
    assert r.overall_ok is True
    assert r.loglik_drift_ok is True
    assert not np.isnan(r.loglik_drift_z)


def test_score_loglik_health_detects_drop_in_likelihood() -> None:
    train = _gaussian_loglik(2000, seed=220, loc=-2.0, scale=0.3)
    b = fit_loglik_drift_baselines(train, window=30)
    # Sustained drop: per-bar log-lik shifted to a much worse value.
    dropped = _gaussian_loglik(200, seed=221, loc=-5.0, scale=0.3)

    r = score_loglik_health(dropped, b, max_abs_z=3.0)

    assert r.loglik_drift_ok is False
    assert r.overall_ok is False
    # Drop should manifest as a strongly negative z.
    assert r.loglik_drift_z < -3.0


def test_score_loglik_health_detects_spike_in_likelihood() -> None:
    # Two-sided gate: spikes should also fail (over-fit / mode collapse).
    train = _gaussian_loglik(2000, seed=230, loc=-2.0, scale=0.3)
    b = fit_loglik_drift_baselines(train, window=30)
    spiked = _gaussian_loglik(200, seed=231, loc=0.5, scale=0.3)

    r = score_loglik_health(spiked, b, max_abs_z=3.0)

    assert r.loglik_drift_ok is False
    assert r.overall_ok is False
    assert r.loglik_drift_z > 3.0


def test_score_loglik_health_overall_ok_mirrors_component_ok() -> None:
    # Single-component report: overall_ok == loglik_drift_ok by construction.
    s = _gaussian_loglik(1000, seed=240)
    b = fit_loglik_drift_baselines(s, window=20)

    r = score_loglik_health(s, b)

    assert r.overall_ok == r.loglik_drift_ok


def test_score_loglik_health_rejects_too_few_rows() -> None:
    s = _gaussian_loglik(500, seed=250)
    b = fit_loglik_drift_baselines(s, window=30)
    too_short = s[:10]

    with pytest.raises(ValueError, match="at least window=30"):
        score_loglik_health(too_short, b)


def test_score_loglik_health_uses_final_row_of_input() -> None:
    # Append a sharp drop at the end of an in-distribution series; the
    # final-row z should be more negative than the in-distribution-only score.
    rng = np.random.default_rng(260)
    train = rng.normal(loc=-2.0, scale=0.3, size=2000)
    b = fit_loglik_drift_baselines(train, window=30)
    in_dist = rng.normal(loc=-2.0, scale=0.3, size=50)
    tail = rng.normal(loc=-6.0, scale=0.1, size=10)
    probe = np.concatenate([in_dist, tail])

    r_with_tail = score_loglik_health(probe, b)
    r_without_tail = score_loglik_health(in_dist, b)

    assert r_with_tail.loglik_drift_z < r_without_tail.loglik_drift_z


# --- audit_refit_metrics ------------------------------------------------------

def test_audit_refit_metrics_passes_on_identical_metrics() -> None:
    old = {"sharpe_ratio": 0.32, "total_return": 2.5}
    new = {"sharpe_ratio": 0.32, "total_return": 2.5}
    thresholds = {"sharpe_ratio": (0.1, 0.2), "total_return": (0.5, 0.3)}

    r = audit_refit_metrics(old, new, thresholds)

    assert isinstance(r, RefitAuditReport)
    assert r.overall_ok is True
    assert all(v.ok for v in r.per_metric)


def test_audit_refit_metrics_detects_absolute_drift() -> None:
    old = {"sharpe_ratio": 0.30}
    new = {"sharpe_ratio": 0.60}
    # |0.60 - 0.30| = 0.30 > max_abs=0.20; relative gate disabled.
    thresholds = {"sharpe_ratio": (0.20, None)}

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is False
    v = r.get("sharpe_ratio")
    assert v.ok is False
    assert v.absolute_delta == pytest.approx(0.30)


def test_audit_refit_metrics_detects_relative_drift() -> None:
    old = {"total_return": 2.0}
    new = {"total_return": 3.0}
    # |(3.0 - 2.0)/2.0| = 0.5 > max_rel=0.30; absolute gate disabled.
    thresholds = {"total_return": (None, 0.30)}

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is False
    v = r.get("total_return")
    assert v.ok is False
    assert v.relative_delta == pytest.approx(0.5)


def test_audit_refit_metrics_either_bound_independently_trips_gate() -> None:
    old = {"a": 1.0, "b": 1.0}
    new = {"a": 1.05, "b": 1.30}
    # 'a': abs=0.05 (under 0.10), rel=0.05 (under 0.10) → ok
    # 'b': abs=0.30 (over 0.10) OR rel=0.30 (over 0.10) → fail
    thresholds = {"a": (0.10, 0.10), "b": (0.10, 0.10)}

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is False
    assert r.get("a").ok is True
    assert r.get("b").ok is False


def test_audit_refit_metrics_passes_when_within_both_bounds() -> None:
    old = {"sharpe_ratio": 0.30}
    new = {"sharpe_ratio": 0.32}
    # abs=0.02 < 0.10; rel=0.067 < 0.20
    thresholds = {"sharpe_ratio": (0.10, 0.20)}

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is True
    assert r.get("sharpe_ratio").ok is True


def test_audit_refit_metrics_relative_delta_none_when_old_near_zero() -> None:
    old = {"win_rate": 0.0}
    new = {"win_rate": 0.4}
    # Absolute gate fires (|0.4| > 0.1); relative gate skipped because |old| < 1e-12.
    thresholds = {"win_rate": (0.1, 0.5)}

    r = audit_refit_metrics(old, new, thresholds)

    v = r.get("win_rate")
    assert v.relative_delta is None
    assert v.ok is False  # absolute gate still fired


def test_audit_refit_metrics_relative_only_gate_is_no_op_when_old_near_zero() -> None:
    # Documented caveat: relative-only gate with old=0 silently cannot fire.
    # Callers should set max_abs_delta as a safety net for metrics that can be 0.
    old = {"win_rate": 0.0}
    new = {"win_rate": 0.4}
    thresholds = {"win_rate": (None, 0.5)}  # relative-only, no abs safety net

    r = audit_refit_metrics(old, new, thresholds)

    v = r.get("win_rate")
    assert v.relative_delta is None
    assert v.ok is True  # gate vacuous — documented behaviour
    assert r.overall_ok is True


def test_audit_refit_metrics_verdict_carries_expected_fields() -> None:
    old = {"sharpe_ratio": 0.5}
    new = {"sharpe_ratio": 0.7}
    thresholds = {"sharpe_ratio": (1.0, 1.0)}

    r = audit_refit_metrics(old, new, thresholds)
    v = r.get("sharpe_ratio")

    assert v.metric_name == "sharpe_ratio"
    assert v.old_value == 0.5
    assert v.new_value == 0.7
    assert v.absolute_delta == pytest.approx(0.2)
    assert v.relative_delta == pytest.approx(0.4)
    assert v.ok is True


def test_audit_refit_metrics_overall_ok_is_strict_and() -> None:
    old = {"a": 1.0, "b": 1.0, "c": 1.0}
    new = {"a": 1.0, "b": 1.0, "c": 5.0}  # only 'c' drifts
    thresholds = {"a": (0.1, None), "b": (0.1, None), "c": (0.1, None)}

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is False
    assert r.get("a").ok is True
    assert r.get("b").ok is True
    assert r.get("c").ok is False


def test_audit_refit_metrics_preserves_threshold_iteration_order() -> None:
    old = {"x": 1.0, "y": 1.0, "z": 1.0}
    new = {"x": 1.0, "y": 1.0, "z": 1.0}
    thresholds = {"z": (0.1, None), "x": (0.1, None), "y": (0.1, None)}

    r = audit_refit_metrics(old, new, thresholds)

    assert tuple(v.metric_name for v in r.per_metric) == ("z", "x", "y")


def test_audit_refit_report_get_raises_for_unknown_metric() -> None:
    old = {"a": 1.0}
    new = {"a": 1.0}
    r = audit_refit_metrics(old, new, {"a": (0.1, None)})

    with pytest.raises(KeyError, match="not in audit report"):
        r.get("nonexistent")


def test_audit_refit_report_and_verdict_are_frozen() -> None:
    old = {"a": 1.0}
    new = {"a": 1.0}
    r = audit_refit_metrics(old, new, {"a": (0.1, None)})

    with pytest.raises(Exception):
        r.overall_ok = False  # type: ignore[misc]
    with pytest.raises(Exception):
        r.per_metric[0].ok = False  # type: ignore[misc]


def test_audit_refit_metrics_rejects_empty_thresholds() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {})


def test_audit_refit_metrics_rejects_malformed_threshold_tuple() -> None:
    with pytest.raises(ValueError, match=r"must be a \(max_abs_delta, max_rel_delta\) tuple"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": 0.5})  # type: ignore[dict-item]


def test_audit_refit_metrics_rejects_both_bounds_none() -> None:
    with pytest.raises(ValueError, match="at least one must be set"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": (None, None)})


def test_audit_refit_metrics_rejects_negative_absolute_bound() -> None:
    with pytest.raises(ValueError, match="max_abs_delta must be > 0"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": (-0.1, None)})


def test_audit_refit_metrics_rejects_zero_absolute_bound() -> None:
    # max_*_delta == 0 with strict > would be vacuous (never fires); with >= would
    # fire on every non-identical pair. Reject upfront.
    with pytest.raises(ValueError, match="max_abs_delta must be > 0"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": (0.0, None)})


def test_audit_refit_metrics_rejects_negative_relative_bound() -> None:
    with pytest.raises(ValueError, match="max_rel_delta must be > 0"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": (None, -0.1)})


def test_audit_refit_metrics_rejects_zero_relative_bound() -> None:
    with pytest.raises(ValueError, match="max_rel_delta must be > 0"):
        audit_refit_metrics({"a": 1.0}, {"a": 1.0}, {"a": (None, 0.0)})


def test_audit_refit_metrics_rejects_missing_old_metric() -> None:
    with pytest.raises(ValueError, match="missing from old_metrics"):
        audit_refit_metrics({}, {"a": 1.0}, {"a": (0.1, None)})


def test_audit_refit_metrics_rejects_missing_new_metric() -> None:
    with pytest.raises(ValueError, match="missing from new_metrics"):
        audit_refit_metrics({"a": 1.0}, {}, {"a": (0.1, None)})


def test_audit_refit_metrics_rejects_non_finite_old_value() -> None:
    with pytest.raises(ValueError, match="old metric 'a' is not finite"):
        audit_refit_metrics({"a": float("nan")}, {"a": 1.0}, {"a": (0.1, None)})


def test_audit_refit_metrics_rejects_non_finite_new_value() -> None:
    with pytest.raises(ValueError, match="new metric 'a' is not finite"):
        audit_refit_metrics({"a": 1.0}, {"a": float("inf")}, {"a": (0.1, None)})


def test_audit_refit_metrics_handles_sharpe_drop_realistic_scenario() -> None:
    # Realistic refit scenario: Sharpe drops from 0.32 to 0.18 — should fail.
    old = {"sharpe_ratio": 0.32, "net_return_after_costs": 2.54, "win_rate": 0.56}
    new = {"sharpe_ratio": 0.18, "net_return_after_costs": 1.10, "win_rate": 0.55}
    thresholds = {
        "sharpe_ratio": (0.10, 0.25),
        "net_return_after_costs": (0.50, 0.30),
        "win_rate": (0.05, 0.10),
    }

    r = audit_refit_metrics(old, new, thresholds)

    assert r.overall_ok is False
    assert r.get("sharpe_ratio").ok is False  # abs delta 0.14 > 0.10
    assert r.get("net_return_after_costs").ok is False  # abs delta 1.44 > 0.50
    assert r.get("win_rate").ok is True  # abs 0.01 < 0.05, rel 0.018 < 0.10
