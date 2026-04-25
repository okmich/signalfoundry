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


def test_truncated_detector_agrees_with_high_r_max_when_fold_mass_is_negligible() -> None:
    """§10.1: when no posterior mass has reached the truncation bin, a small-``r_max``
    detector and a much-larger-``r_max`` detector produce identical posteriors over
    the columns the smaller one represents.

    Construction: ``T = 50`` updates with ``r_max_small = 60`` and ``r_max_large = 200``.
    Because the recursion only advances mass by one column per step (plus the cp jump
    to column 0), after T updates posterior mass is supported on columns ``0..T``;
    columns ``> T`` are exactly zero. This means the truncation bin of the smaller
    detector has not yet been touched, so both detectors must agree on every column
    of the smaller one.
    """
    rng = np.random.default_rng(seed=7)
    xs = rng.standard_normal(50).astype(np.float64)

    small = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=60)
    large = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=0.10, r_max=200)

    small_post = small.batch(xs)
    large_post = large.batch(xs)

    np.testing.assert_allclose(small_post, large_post[:, :60], atol=1e-12)


def test_posterior_mean_run_length_drops_after_clean_mean_shift() -> None:
    xs = np.concatenate([np.zeros(60, dtype=np.float64), np.full(20, 5.0, dtype=np.float64)])
    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=1.0 / 30.0, r_max=120)

    posterior = detector.batch(xs)
    run_lengths = np.arange(posterior.shape[1], dtype=np.float64)
    expected_run_length = posterior @ run_lengths

    assert expected_run_length[58] > 20.0
    assert np.min(expected_run_length[60:66]) < 10.0
