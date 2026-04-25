import numpy as np
import pytest

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel
from okmich_quant_ml.posterior_inference import (
    AbstainMode,
    EmaPosteriorTransformer,
    MarginGateInferer,
    PosteriorPipeline,
)


class FixedLogPredModel:
    def __init__(self, rows: list[np.ndarray]) -> None:
        self.rows = rows
        self.idx = 0
        self.reset_calls: list[int] = []
        self.updates: list[float] = []

    def reset(self, r_max: int) -> None:
        self.reset_calls.append(r_max)
        self.idx = 0
        self.updates = []

    def log_pred_probs(self, x: float) -> np.ndarray:
        row = self.rows[min(self.idx, len(self.rows) - 1)]
        return np.log(row).astype(np.float64)

    def update(self, x: float) -> None:
        self.updates.append(float(x))
        self.idx += 1


def test_update_recursion_matches_manual_constant_hazard_calculation() -> None:
    model = FixedLogPredModel(
        [
            np.array([0.20, 0.40, 0.80], dtype=np.float64),
            np.array([0.50, 0.25, 0.10], dtype=np.float64),
        ]
    )
    detector = BayesianOnlineChangepointDetector(model, hazard_rate=0.25, r_max=3)

    first = detector.update(10.0)
    second = detector.update(11.0)

    np.testing.assert_allclose(first, np.array([0.25, 0.75, 0.0]), atol=1e-12)
    np.testing.assert_allclose(second, np.array([0.25, 0.30, 0.45]), atol=1e-12)
    assert detector.changepoint_prob == pytest.approx(second[0])
    assert detector.map_run_length == 2
    assert model.updates == [10.0, 11.0]


def test_batch_is_stateful_and_equivalent_to_online_updates() -> None:
    xs = np.array([0.10, -0.20, 0.15, 1.80, 2.10, 1.90], dtype=np.float64)
    model_a = NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0)
    model_b = NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0)
    detector_a = BayesianOnlineChangepointDetector(model_a, hazard_rate=0.10, r_max=10)
    detector_b = BayesianOnlineChangepointDetector(model_b, hazard_rate=0.10, r_max=10)

    full = detector_a.batch(xs)
    split_first = detector_b.batch(xs[:3])
    split_second = detector_b.batch(xs[3:])

    assert full.shape == (len(xs), 10)
    np.testing.assert_allclose(np.vstack([split_first, split_second]), full, atol=1e-12)
    np.testing.assert_allclose(detector_b.run_length_posterior_, full[-1], atol=1e-12)


def test_batch_empty_is_no_op() -> None:
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.20, r_max=5)
    before = detector.run_length_posterior_.copy()

    out = detector.batch(np.array([], dtype=np.float64))

    assert out.shape == (0, 5)
    np.testing.assert_array_equal(detector.run_length_posterior_, before)


def test_reset_restores_detector_and_observation_model_prior_state() -> None:
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.20, r_max=5)
    detector.batch(np.array([0.1, 0.2, 0.3], dtype=np.float64))

    detector.reset()

    np.testing.assert_array_equal(detector.run_length_posterior_, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert detector.changepoint_prob == 1.0
    assert detector.map_run_length == 0


def test_detector_rejects_invalid_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="hazard_rate"):
        BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.0, r_max=5)
    with pytest.raises(ValueError, match="r_max"):
        BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=1)

    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=5)
    with pytest.raises(ValueError, match="finite"):
        detector.update(float("nan"))
    # batch now coerces any numeric dtype to float64 rather than rejecting — verify
    # the coercion path produces the same result as an explicit float64 call.
    int_out = detector.batch(np.array([1, 2, 3], dtype=np.int64))
    detector.reset()
    float_out = detector.batch(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(int_out, float_out)
    with pytest.raises(ValueError, match="one-dimensional"):
        detector.batch(np.ones((2, 2), dtype=np.float64))


def test_posterior_pipeline_consumes_run_length_posterior_without_adapter() -> None:
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=8)
    posterior = detector.batch(np.array([0.0, 0.1, 0.2, 2.5, 2.7, 2.6], dtype=np.float64))
    pipeline = PosteriorPipeline(
        transformers=[EmaPosteriorTransformer(alpha=0.50)],
        inferer=MarginGateInferer(theta_top=0.30, theta_margin=0.02, abstain_mode=AbstainMode.FLAT, abstain_label=-1),
    )

    labels = pipeline.run(posterior)

    assert posterior.shape == (6, 8)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
    assert labels.shape == (6,)
    assert labels.dtype == np.int64
