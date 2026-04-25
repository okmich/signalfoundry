import numpy as np
from scipy.special import logsumexp

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel


def _reference_detector(xs: np.ndarray, hazard_rate: float, r_max: int) -> np.ndarray:
    model = NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0)
    model.reset(r_max)
    posterior = np.zeros(r_max, dtype=np.float64)
    posterior[0] = 1.0
    out = np.empty((len(xs), r_max), dtype=np.float64)
    log_h = np.log(hazard_rate)
    log_g = np.log1p(-hazard_rate)
    for i, x in enumerate(xs):
        log_pred = model.log_pred_probs(float(x))
        log_prev = np.full(r_max, -np.inf, dtype=np.float64)
        positive = posterior > 0.0
        log_prev[positive] = np.log(posterior[positive])
        joint = log_prev + log_pred
        next_log = np.full(r_max, -np.inf, dtype=np.float64)
        next_log[0] = logsumexp(joint + log_h)
        next_log[1:] = joint[:-1] + log_g
        next_log[-1] = np.logaddexp(next_log[-1], joint[-1] + log_g)
        posterior = np.exp(next_log - logsumexp(next_log))
        posterior /= posterior.sum()
        out[i] = posterior
        model.update(float(x))
    return out


def test_normal_inverse_gamma_detector_matches_reference_recursion() -> None:
    xs = np.array([0.05, -0.10, 0.00, 0.15, 2.20, 2.00, 2.10, 1.90], dtype=np.float64)
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=12)

    actual = detector.batch(xs)
    expected = _reference_detector(xs, hazard_rate=0.10, r_max=12)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_posterior_mean_run_length_drops_after_clean_mean_shift() -> None:
    xs = np.concatenate([np.zeros(60, dtype=np.float64), np.full(20, 5.0, dtype=np.float64)])
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=1.0 / 30.0, r_max=120)

    posterior = detector.batch(xs)
    run_lengths = np.arange(posterior.shape[1], dtype=np.float64)
    expected_run_length = posterior @ run_lengths

    assert expected_run_length[58] > 20.0
    assert np.min(expected_run_length[60:66]) < 10.0
