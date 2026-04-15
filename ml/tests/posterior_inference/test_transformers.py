import numpy as np

from okmich_quant_ml.posterior_inference import EmaPosteriorTransformer, TemperatureScalingTransformer, entropy


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
