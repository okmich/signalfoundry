import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    EmaPosteriorTransformer,
    MaturationAlignTransformer,
    PlattScalingTransformer,
    RollingMeanPosteriorTransformer,
    TemperatureScalingTransformer,
    entropy,
)


def _nll(probs: np.ndarray, y_idx: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))


def test_ema_transformer_preserves_shape_and_simplex() -> None:
    probs = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.15, 0.15, 0.70],
            [0.70, 0.20, 0.10],
        ],
        dtype=float,
    )
    transformer = EmaPosteriorTransformer(alpha=0.20)

    transformed = transformer.transform(probs)

    assert transformed.shape == probs.shape
    np.testing.assert_allclose(transformed.sum(axis=1), np.ones(len(probs)), atol=1e-12)
    np.testing.assert_allclose(transformed[0], probs[0], atol=1e-12)


def test_ema_transformer_alpha_one_is_identity() -> None:
    probs = np.array(
        [
            [0.10, 0.60, 0.30],
            [0.70, 0.20, 0.10],
            [0.25, 0.25, 0.50],
        ],
        dtype=float,
    )
    transformer = EmaPosteriorTransformer(alpha=1.0)

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed, probs, atol=1e-12)


def test_temperature_scaling_t_one_is_identity() -> None:
    probs = np.array(
        [
            [0.92, 0.05, 0.03],
            [0.30, 0.55, 0.15],
            [0.02, 0.08, 0.90],
        ],
        dtype=float,
    )
    transformer = TemperatureScalingTransformer(temperature=1.0)

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed, probs, atol=1e-12)


def test_temperature_scaling_higher_temperature_increases_entropy() -> None:
    probs = np.array(
        [
            [0.97, 0.02, 0.01],
            [0.85, 0.10, 0.05],
            [0.75, 0.20, 0.05],
            [0.60, 0.30, 0.10],
        ],
        dtype=float,
    )
    transformer = TemperatureScalingTransformer(temperature=4.0)

    transformed = transformer.transform(probs)

    base_entropy = float(np.mean(entropy(probs)))
    scaled_entropy = float(np.mean(entropy(transformed)))
    assert scaled_entropy > base_entropy
    np.testing.assert_allclose(transformed.sum(axis=1), np.ones(len(probs)), atol=1e-12)


def test_rolling_mean_window_one_is_identity() -> None:
    probs = np.array(
        [
            [0.10, 0.60, 0.30],
            [0.70, 0.20, 0.10],
            [0.25, 0.25, 0.50],
        ],
        dtype=float,
    )
    transformer = RollingMeanPosteriorTransformer(window=1)

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed, probs, atol=1e-12)


def test_rolling_mean_preserves_shape_and_simplex() -> None:
    probs = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.20, 0.20, 0.60],
            [0.70, 0.20, 0.10],
            [0.05, 0.15, 0.80],
        ],
        dtype=float,
    )
    transformer = RollingMeanPosteriorTransformer(window=3)

    transformed = transformer.transform(probs)

    assert transformed.shape == probs.shape
    np.testing.assert_allclose(transformed.sum(axis=1), np.ones(len(probs)), atol=1e-12)


def test_rolling_mean_is_causal_and_matches_trailing_average() -> None:
    probs = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.20, 0.20, 0.60],
            [0.70, 0.20, 0.10],
            [0.05, 0.15, 0.80],
            [0.40, 0.40, 0.20],
        ],
        dtype=float,
    )
    transformer = RollingMeanPosteriorTransformer(window=3)

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed[0], probs[0], atol=1e-12)
    np.testing.assert_allclose(transformed[1], probs[:2].mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(transformed[2], probs[:3].mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(transformed[3], probs[1:4].mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(transformed[5], probs[3:6].mean(axis=0), atol=1e-12)

    truncated_last = transformer.transform(probs[:4])
    np.testing.assert_allclose(truncated_last[-1], transformed[3], atol=1e-12)


def test_rolling_mean_window_larger_than_history_averages_all_available() -> None:
    probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=float,
    )
    transformer = RollingMeanPosteriorTransformer(window=10)

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed[0], probs[0], atol=1e-12)
    np.testing.assert_allclose(transformed[1], probs.mean(axis=0), atol=1e-12)


def test_rolling_mean_reduces_variance_on_noisy_posteriors() -> None:
    rng = np.random.default_rng(13)
    n_rows = 400
    n_classes = 3
    signal = np.tile(np.array([0.70, 0.20, 0.10]), (n_rows, 1))
    noise = rng.dirichlet(alpha=np.ones(n_classes), size=n_rows) - 1.0 / n_classes
    noisy = signal + 0.5 * noise
    noisy = np.clip(noisy, 1e-6, None)
    noisy = noisy / noisy.sum(axis=1, keepdims=True)
    transformer = RollingMeanPosteriorTransformer(window=20)

    smoothed = transformer.transform(noisy)

    assert smoothed.var(axis=0).sum() < noisy.var(axis=0).sum()
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(n_rows), atol=1e-10)


def test_rolling_mean_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="window"):
        RollingMeanPosteriorTransformer(window=0)
    with pytest.raises(ValueError, match="eps"):
        RollingMeanPosteriorTransformer(window=3, eps=0.0)


def test_rolling_mean_handles_empty_input() -> None:
    empty = np.zeros((0, 3), dtype=float)
    transformer = RollingMeanPosteriorTransformer(window=5)

    transformed = transformer.transform(empty)

    assert transformed.shape == (0, 3)


def test_platt_scaling_without_fit_is_identity() -> None:
    probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.30, 0.60],
            [0.33, 0.33, 0.34],
        ],
        dtype=float,
    )
    transformer = PlattScalingTransformer()

    transformed = transformer.transform(probs)

    np.testing.assert_allclose(transformed, probs, atol=1e-10)


def test_platt_scaling_fit_improves_nll_on_overconfident_predictions() -> None:
    rng = np.random.default_rng(11)
    n_rows = 600
    n_classes = 3
    y_idx = rng.integers(0, n_classes, size=n_rows, endpoint=False)

    probs = np.full((n_rows, n_classes), 0.01, dtype=float)
    predicted = y_idx.copy()
    wrong_mask = rng.random(n_rows) < 0.30
    noise = rng.integers(1, n_classes, size=wrong_mask.sum(), endpoint=False)
    predicted[wrong_mask] = (predicted[wrong_mask] + noise) % n_classes
    probs[np.arange(n_rows), predicted] = 0.98

    baseline_nll = _nll(probs, y_idx)

    transformer = PlattScalingTransformer()
    transformer.fit(probs, y_idx)
    transformed = transformer.transform(probs)
    improved_nll = _nll(transformed, y_idx)

    assert transformer.a_ is not None and transformer.b_ is not None
    assert improved_nll < baseline_nll
    np.testing.assert_allclose(transformed.sum(axis=1), np.ones(n_rows), atol=1e-10)


def test_platt_scaling_rejects_mismatched_class_count_after_fit() -> None:
    rng = np.random.default_rng(2)
    n_rows = 50
    probs = rng.dirichlet(alpha=np.ones(3), size=n_rows)
    y_idx = rng.integers(0, 3, size=n_rows, endpoint=False)
    transformer = PlattScalingTransformer().fit(probs, y_idx)

    mismatched = rng.dirichlet(alpha=np.ones(4), size=10)
    with pytest.raises(ValueError, match="Fitted parameters"):
        transformer.transform(mismatched)


def test_platt_scaling_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="eps"):
        PlattScalingTransformer(eps=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        PlattScalingTransformer(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        PlattScalingTransformer(tol=0.0)


def test_platt_scaling_rejects_malformed_y_idx() -> None:
    probs = np.array([[0.50, 0.30, 0.20], [0.10, 0.80, 0.10]], dtype=float)
    transformer = PlattScalingTransformer()

    with pytest.raises(ValueError, match="y_idx must be 1D"):
        transformer.fit(probs, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="y_idx length"):
        transformer.fit(probs, np.array([0, 1, 2]))
    with pytest.raises(ValueError, match=r"y_idx must be in \[0"):
        transformer.fit(probs, np.array([0, 3]))
    with pytest.raises(ValueError, match=r"y_idx must be in \[0"):
        transformer.fit(probs, np.array([-1, 0]))


def test_platt_scaling_fit_transform_on_held_out_produces_valid_simplex() -> None:
    rng = np.random.default_rng(17)
    train = rng.dirichlet(alpha=np.ones(3), size=200)
    y_idx = rng.integers(0, 3, size=200, endpoint=False)
    transformer = PlattScalingTransformer().fit(train, y_idx)

    held_out = rng.dirichlet(alpha=np.ones(3), size=50)
    transformed = transformer.transform(held_out)

    assert transformed.shape == held_out.shape
    np.testing.assert_allclose(transformed.sum(axis=1), np.ones(50), atol=1e-10)
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0 + 1e-10)


def test_platt_scaling_preserves_argmax_when_fit_with_identity_signal() -> None:
    rng = np.random.default_rng(19)
    n_rows = 300
    n_classes = 3
    y_idx = rng.integers(0, n_classes, size=n_rows, endpoint=False)
    probs = np.full((n_rows, n_classes), 0.05, dtype=float)
    probs[np.arange(n_rows), y_idx] = 0.90
    probs = probs / probs.sum(axis=1, keepdims=True)

    transformer = PlattScalingTransformer().fit(probs, y_idx)
    transformed = transformer.transform(probs)

    np.testing.assert_array_equal(np.argmax(transformed, axis=1), np.argmax(probs, axis=1))


def test_temperature_scaling_fit_improves_nll_on_overconfident_predictions() -> None:
    rng = np.random.default_rng(7)
    n_rows = 500
    n_classes = 3
    y_idx = rng.integers(0, n_classes, size=n_rows, endpoint=False)

    probs = np.full((n_rows, n_classes), 0.01, dtype=float)
    predicted = y_idx.copy()
    wrong_mask = rng.random(n_rows) < 0.25
    noise = rng.integers(1, n_classes, size=wrong_mask.sum(), endpoint=False)
    predicted[wrong_mask] = (predicted[wrong_mask] + noise) % n_classes
    probs[np.arange(n_rows), predicted] = 0.98

    baseline_nll = _nll(probs, y_idx)

    transformer = TemperatureScalingTransformer(temperature=1.0)
    transformer.fit(probs, y_idx)
    transformed = transformer.transform(probs)
    improved_nll = _nll(transformed, y_idx)

    assert transformer.temperature > 1.0
    assert improved_nll < baseline_nll


def test_maturation_align_zero_lag_is_identity() -> None:
    probs = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.50, 0.30],
            [0.10, 0.10, 0.80],
        ],
        dtype=float,
    )
    out = MaturationAlignTransformer(lag=0).transform(probs)

    np.testing.assert_allclose(out, probs, atol=1e-12)


def test_maturation_align_shifts_rows_and_prepends_uniform() -> None:
    probs = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
            [0.30, 0.40, 0.30],
        ],
        dtype=float,
    )
    out = MaturationAlignTransformer(lag=2).transform(probs)

    assert out.shape == probs.shape
    # First `lag` rows are the uniform prior.
    np.testing.assert_allclose(out[0], np.full(3, 1.0 / 3), atol=1e-12)
    np.testing.assert_allclose(out[1], np.full(3, 1.0 / 3), atol=1e-12)
    # Subsequent rows are the input shifted down by `lag`.
    np.testing.assert_allclose(out[2], probs[0], atol=1e-12)
    np.testing.assert_allclose(out[3], probs[1], atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-12)


def test_maturation_align_lag_exceeding_T_fills_all_uniform() -> None:
    probs = np.array([[0.90, 0.10], [0.20, 0.80]], dtype=float)
    out = MaturationAlignTransformer(lag=10).transform(probs)

    np.testing.assert_allclose(out, np.full((2, 2), 0.5), atol=1e-12)


def test_maturation_align_preserves_simplex_on_renormalized_input() -> None:
    rng = np.random.default_rng(23)
    probs = rng.dirichlet(alpha=np.ones(4), size=6)
    out = MaturationAlignTransformer(lag=1).transform(probs)

    np.testing.assert_allclose(out.sum(axis=1), np.ones(6), atol=1e-12)
    assert np.all(out > 0.0)


def test_maturation_align_rejects_negative_lag() -> None:
    with pytest.raises(ValueError, match="lag must be >= 0"):
        MaturationAlignTransformer(lag=-1)


def test_maturation_align_rejects_malformed_input() -> None:
    transformer = MaturationAlignTransformer(lag=1)
    with pytest.raises(ValueError, match="shape"):
        transformer.transform(np.array([0.5, 0.5], dtype=float))
    with pytest.raises(ValueError, match="NaN or Inf"):
        transformer.transform(np.array([[0.5, np.nan]], dtype=float))


def test_maturation_align_preserves_exact_zero_values() -> None:
    # Alignment must be a pure rearrangement — it must not clip zeros or renormalize rows.
    # Downstream calibration transformers clip as they need log-safe input, but
    # MaturationAlign is a semantic shift and should hand back the exact input.
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    shifted = MaturationAlignTransformer(lag=1).transform(probs)

    # Prepended row is the uniform prior; remaining rows must be the exact input.
    np.testing.assert_array_equal(shifted[0], np.full(3, 1.0 / 3))
    np.testing.assert_array_equal(shifted[1:], probs[:-1])


def test_maturation_align_zero_lag_does_not_clip() -> None:
    probs = np.array([[1.0, 0.0]], dtype=float)
    out = MaturationAlignTransformer(lag=0).transform(probs)

    np.testing.assert_array_equal(out, probs)
