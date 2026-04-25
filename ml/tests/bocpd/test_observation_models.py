import numpy as np
import pytest
from scipy.special import gammaln

from okmich_quant_ml.bocpd import GammaExponentialModel, GaussianKnownVarianceModel, NormalInverseGammaModel


def _nig_closed_form(xs: np.ndarray, mu_0: float, kappa_0: float,
                     alpha_0: float, beta_0: float) -> tuple[float, float, float, float]:
    mu = mu_0
    kappa = kappa_0
    alpha = alpha_0
    beta = beta_0
    for x in xs:
        next_kappa = kappa + 1.0
        next_mu = (kappa * mu + x) / next_kappa
        next_alpha = alpha + 0.5
        next_beta = beta + kappa * ((x - mu) ** 2) / (2.0 * next_kappa)
        mu, kappa, alpha, beta = next_mu, next_kappa, next_alpha, next_beta
    return mu, kappa, alpha, beta


def test_gaussian_known_variance_prior_predictive_and_update() -> None:
    model = GaussianKnownVarianceModel(mu_0=1.0, sigma0_sq=4.0, sigma_obs_sq=9.0)
    model.reset(5)

    log_pred = model.log_pred_probs(2.0)
    expected = -0.5 * (np.log(2.0 * np.pi * 13.0) + ((2.0 - 1.0) ** 2) / 13.0)
    assert log_pred[0] == pytest.approx(expected)

    model.update(2.0)
    posterior_var = 1.0 / (1.0 / 4.0 + 1.0 / 9.0)
    posterior_mu = posterior_var * (1.0 / 4.0 + 2.0 / 9.0)
    assert model.sigma_sq_[1] == pytest.approx(posterior_var)
    assert model.mu_[1] == pytest.approx(posterior_mu)


def test_normal_inverse_gamma_prior_predictive_and_closed_form_update() -> None:
    model = NormalInverseGammaModel(mu_0=0.0, kappa_0=2.0, alpha_0=3.0, beta_0=4.0)
    model.reset(6)
    x = 0.25

    log_pred = model.log_pred_probs(x)
    df = 6.0
    scale_sq = 4.0 * 3.0 / (3.0 * 2.0)
    expected = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * (np.log(df * np.pi) + np.log(scale_sq))
        - ((df + 1.0) / 2.0) * np.log1p((x ** 2) / (df * scale_sq))
    )
    assert log_pred[0] == pytest.approx(expected)

    xs = np.array([0.25, -0.10, 0.30], dtype=np.float64)
    for value in xs:
        model.update(float(value))
    mu, kappa, alpha, beta = _nig_closed_form(xs, 0.0, 2.0, 3.0, 4.0)

    actual = [model.mu_[3], model.kappa_[3], model.alpha_[3], model.beta_[3]]
    np.testing.assert_allclose(actual, [mu, kappa, alpha, beta])


def test_gamma_exponential_prior_predictive_and_update() -> None:
    model = GammaExponentialModel(alpha_0=2.0, beta_0=3.0)
    model.reset(5)

    log_pred = model.log_pred_probs(4.0)
    expected = np.log(2.0) + 2.0 * np.log(3.0) - 3.0 * np.log(7.0)
    assert log_pred[0] == pytest.approx(expected)

    model.update(4.0)

    assert model.alpha_[1] == pytest.approx(3.0)
    assert model.beta_[1] == pytest.approx(7.0)


def test_saturated_final_slot_advances_from_itself_after_first_population() -> None:
    model = NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0)
    model.reset(3)

    for value in (1.0, 2.0, 3.0):
        model.update(value)

    mu, kappa, alpha, beta = _nig_closed_form(np.array([1.0, 2.0, 3.0]), 0.0, 1.0, 1.0, 1.0)
    actual = [model.mu_[2], model.kappa_[2], model.alpha_[2], model.beta_[2]]
    np.testing.assert_allclose(actual, [mu, kappa, alpha, beta])
    assert model.mu_[2] != pytest.approx(_nig_closed_form(np.array([2.0, 3.0]), 0.0, 1.0, 1.0, 1.0)[0])


def test_observation_models_reject_invalid_hyperparameters_and_observations() -> None:
    with pytest.raises(ValueError, match="sigma_obs_sq"):
        GaussianKnownVarianceModel(mu_0=0.0, sigma0_sq=1.0, sigma_obs_sq=0.0)
    with pytest.raises(ValueError, match="kappa_0"):
        NormalInverseGammaModel(kappa_0=0.0)
    with pytest.raises(ValueError, match="alpha_0"):
        GammaExponentialModel(alpha_0=0.0, beta_0=1.0)

    model = GammaExponentialModel(alpha_0=1.0, beta_0=1.0)
    model.reset(3)
    with pytest.raises(ValueError, match="> 0"):
        model.log_pred_probs(0.0)
    with pytest.raises(ValueError, match="> 0"):
        model.update(-1.0)


def test_observation_models_require_reset_before_use() -> None:
    model = NormalInverseGammaModel()
    with pytest.raises(RuntimeError, match="reset"):
        model.log_pred_probs(0.0)
    with pytest.raises(RuntimeError, match="reset"):
        model.update(0.0)
