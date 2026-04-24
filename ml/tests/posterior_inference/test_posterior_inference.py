import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    AbstainMode,
    ArgmaxInferer,
    CompositeGateInferer,
    EmaPosteriorTransformer,
    EntropyGateInferer,
    MarginGateInferer,
    PosteriorPipeline,
    RollingMeanPosteriorTransformer,
    dwell_length,
    entropy,
    margin,
    rolling_entropy_std,
    rolling_flip_rate,
    rolling_max_prob_std,
    step_kl,
)


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


def test_pipeline_with_ema_transformer_and_margin_gate_inferer_runs_end_to_end() -> None:
    probs = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.55, 0.35, 0.10],
            [0.10, 0.80, 0.10],
            [0.45, 0.45, 0.10],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(
        transformers=[EmaPosteriorTransformer(alpha=0.30)],
        inferer=MarginGateInferer(theta_top=0.70, theta_margin=0.20, abstain_mode=AbstainMode.HOLD_LAST, abstain_label=2),
    )

    labels = pipeline.run(probs)

    assert labels.shape == (len(probs),)
    assert labels.dtype.kind in {"i", "u"}


def test_pipeline_with_rolling_mean_and_composite_gate_runs_end_to_end() -> None:
    probs = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.70, 0.20, 0.10],
            [0.05, 0.10, 0.85],
            [0.10, 0.80, 0.10],
            [0.33, 0.34, 0.33],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(
        transformers=[RollingMeanPosteriorTransformer(window=2)],
        inferer=CompositeGateInferer(theta_top=0.55, theta_margin=0.20, theta_entropy=0.80,
                                     abstain_mode=AbstainMode.FLAT, abstain_label=-1),
    )

    labels = pipeline.run(probs)

    assert labels.shape == (len(probs),)
    assert labels.dtype.kind in {"i"}


def test_pipeline_with_entropy_gate_only_runs_end_to_end() -> None:
    probs = np.array(
        [
            [0.92, 0.05, 0.03],
            [0.40, 0.30, 0.30],
            [0.02, 0.96, 0.02],
        ],
        dtype=float,
    )
    pipeline = PosteriorPipeline(
        transformers=[],
        inferer=EntropyGateInferer(theta_entropy=0.40, abstain_mode=AbstainMode.FLAT, abstain_label=-1),
    )

    labels = pipeline.run(probs)

    np.testing.assert_array_equal(labels, np.array([0, -1, 1], dtype=np.int64))


def test_step_kl_is_zero_on_identical_rows() -> None:
    probs = np.tile(np.array([0.3, 0.4, 0.3], dtype=float), (6, 1))
    out = step_kl(probs)

    assert out.shape == (6,)
    assert out[0] == 0.0
    np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)


def test_step_kl_grows_with_magnitude_of_shift() -> None:
    probs = np.array(
        [
            [0.50, 0.30, 0.20],
            [0.50, 0.30, 0.20],
            [0.45, 0.35, 0.20],
            [0.10, 0.10, 0.80],
        ],
        dtype=float,
    )
    out = step_kl(probs)

    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.0, abs=1e-12)
    assert out[2] > 0.0
    assert out[3] > out[2]


def test_step_kl_handles_zero_probability_in_prior_row() -> None:
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
        ],
        dtype=float,
    )
    out = step_kl(probs)

    assert np.isfinite(out).all()
    assert out[1] > 0.0


def test_step_kl_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="step_kl requires"):
        step_kl(np.array([0.5, 0.5], dtype=float))


def test_rolling_flip_rate_is_zero_on_stable_argmax() -> None:
    probs = np.tile(np.array([0.7, 0.2, 0.1], dtype=float), (8, 1))
    out = rolling_flip_rate(probs, window=5)

    assert out.shape == (8,)
    np.testing.assert_allclose(out, 0.0)


def test_rolling_flip_rate_saturates_on_alternating_argmax() -> None:
    probs = np.array([[0.6, 0.4], [0.4, 0.6]] * 10, dtype=float)
    out = rolling_flip_rate(probs, window=5)

    # Steady state: every step flips, so the trailing flip rate saturates at 1.0
    # once the window has slid past the initial zero-flip row.
    assert out[-1] == pytest.approx(1.0)


def test_rolling_flip_rate_rejects_bad_window() -> None:
    probs = np.tile(np.array([0.5, 0.5], dtype=float), (5, 1))
    with pytest.raises(ValueError, match="window must be >= 1"):
        rolling_flip_rate(probs, window=0)


def test_rolling_max_prob_std_is_zero_on_constant_top_prob() -> None:
    probs = np.tile(np.array([0.7, 0.3], dtype=float), (10, 1))
    out = rolling_max_prob_std(probs, window=4)

    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_rolling_max_prob_std_is_positive_when_top_prob_varies() -> None:
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.9, 0.1], [0.6, 0.4], [0.9, 0.1]], dtype=float)
    out = rolling_max_prob_std(probs, window=5)

    assert out[-1] > 0.0


def test_rolling_entropy_std_is_zero_on_flat_posterior_sequence() -> None:
    probs = np.tile(np.array([0.5, 0.5], dtype=float), (6, 1))
    out = rolling_entropy_std(probs, window=3)

    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_dwell_length_increments_during_stable_run() -> None:
    probs = np.tile(np.array([0.7, 0.3], dtype=float), (5, 1))
    out = dwell_length(probs)

    np.testing.assert_array_equal(out, np.array([1, 2, 3, 4, 5], dtype=np.int64))


def test_dwell_length_resets_on_argmax_change() -> None:
    probs = np.array(
        [
            [0.7, 0.3],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.3, 0.7],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    out = dwell_length(probs)

    np.testing.assert_array_equal(out, np.array([1, 2, 1, 2, 1], dtype=np.int64))


def test_dwell_length_handles_empty_input() -> None:
    out = dwell_length(np.zeros((0, 3), dtype=float))

    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_dynamic_features_reject_nan_input() -> None:
    bad = np.array([[0.5, 0.5], [0.5, np.nan]], dtype=float)

    with pytest.raises(ValueError, match="NaN or Inf"):
        step_kl(bad)
    with pytest.raises(ValueError, match="NaN or Inf"):
        rolling_flip_rate(bad, window=2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        rolling_max_prob_std(bad, window=2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        rolling_entropy_std(bad, window=2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        dwell_length(bad)
