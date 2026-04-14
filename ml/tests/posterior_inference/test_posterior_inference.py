import numpy as np

from okmich_quant_ml.posterior_inference import ArgmaxInferer, PosteriorPipeline, entropy, margin


class ReverseColumnsTransformer:
    def transform(self, probs: np.ndarray) -> np.ndarray:
        return probs[..., ::-1]


class SquareNormalizeTransformer:
    def transform(self, probs: np.ndarray) -> np.ndarray:
        squared = probs ** 2
        return squared / np.sum(squared, axis=-1, keepdims=True)


def test_argmax_inferer_returns_expected_shape_and_values() -> None:
    probs = np.array(
        [
            [0.10, 0.70, 0.20],
            [0.60, 0.25, 0.15],
            [0.05, 0.15, 0.80],
            [0.90, 0.05, 0.05],
        ],
        dtype=float,
    )

    inferer = ArgmaxInferer()
    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.array([1, 0, 2, 0]))
    assert labels.shape == (4,)
    assert inferer.get_metadata() == {}


def test_margin_returns_expected_values_on_canonical_inputs() -> None:
    probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.34, 0.33, 0.33],
            [1.00, 0.00, 0.00],
        ],
        dtype=float,
    )

    expected = np.array([0.50, 0.01, 1.00], dtype=float)
    np.testing.assert_allclose(margin(probs), expected, atol=1e-12)


def test_entropy_handles_uniform_and_one_hot_cases() -> None:
    uniform = np.full((3, 4), 0.25, dtype=float)
    one_hot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    uniform_entropy = entropy(uniform)
    one_hot_entropy = entropy(one_hot)

    np.testing.assert_allclose(uniform_entropy, np.full(3, np.log(4.0)), atol=1e-10)
    np.testing.assert_allclose(one_hot_entropy, np.zeros(2), atol=1e-10)


def test_pipeline_composes_transformers_then_inferer() -> None:
    probs = np.array(
        [
            [0.20, 0.60, 0.20],
            [0.80, 0.10, 0.10],
            [0.30, 0.25, 0.45],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(
        transformers=[ReverseColumnsTransformer(), SquareNormalizeTransformer()],
        inferer=ArgmaxInferer(),
    )

    actual_labels = pipeline.run(probs)

    expected_probs = probs[..., ::-1]
    expected_probs = expected_probs ** 2
    expected_probs = expected_probs / np.sum(expected_probs, axis=-1, keepdims=True)
    expected_labels = np.argmax(expected_probs, axis=-1)

    np.testing.assert_array_equal(actual_labels, expected_labels)


def test_margin_works_for_k2() -> None:
    probs = np.array(
        [
            [0.80, 0.20],
            [0.55, 0.45],
            [0.00, 1.00],
        ],
        dtype=float,
    )

    expected = np.array([0.60, 0.10, 1.00], dtype=float)
    np.testing.assert_allclose(margin(probs), expected, atol=1e-12)


def test_pipeline_empty_transformers_returns_probs_unchanged() -> None:
    probs = np.array(
        [
            [0.30, 0.50, 0.20],
            [0.10, 0.10, 0.80],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(transformers=[])

    result = pipeline.run(probs)

    np.testing.assert_array_equal(result, probs)


def test_pipeline_handles_single_row_input() -> None:
    probs = np.array([[0.15, 0.65, 0.20]], dtype=float)
    pipeline = PosteriorPipeline(transformers=[], inferer=ArgmaxInferer())

    labels = pipeline.run(probs)

    np.testing.assert_array_equal(labels, np.array([1]))
    assert labels.shape == (1,)


def test_pipeline_without_inferer_returns_final_posterior_shape_preserved() -> None:
    probs = np.array(
        [
            [0.25, 0.25, 0.50],
            [0.70, 0.20, 0.10],
            [0.33, 0.33, 0.34],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(transformers=[SquareNormalizeTransformer()])

    transformed = pipeline.run(probs)
    expected = probs ** 2
    expected = expected / np.sum(expected, axis=-1, keepdims=True)

    assert transformed.shape == probs.shape
    np.testing.assert_allclose(transformed, expected, atol=1e-12)
