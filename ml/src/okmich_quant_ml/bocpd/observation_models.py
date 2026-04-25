from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.special import gammaln


_LOG_2PI = float(np.log(2.0 * np.pi))


@njit(cache=True)
def _gaussian_log_pred_kernel(x: float, mu: np.ndarray, sigma_post: np.ndarray,
                              sigma_obs: float, out: np.ndarray) -> None:
    for i in range(len(mu)):
        pred_var = sigma_post[i] + sigma_obs
        diff = x - mu[i]
        out[i] = -0.5 * (_LOG_2PI + np.log(pred_var) + (diff * diff) / pred_var)


@njit(cache=True)
def _gaussian_update_kernel(x: float, mu: np.ndarray, sigma_post: np.ndarray, sigma_obs: float,
                            prior_mu: float, prior_var: float, n_updates: int,
                            next_mu: np.ndarray, next_var: np.ndarray) -> None:
    r_max = len(mu)
    next_mu[0] = prior_mu
    next_var[0] = prior_var
    for j in range(1, r_max):
        src = j - 1
        advanced_var = 1.0 / (1.0 / sigma_post[src] + 1.0 / sigma_obs)
        next_var[j] = advanced_var
        next_mu[j] = advanced_var * (mu[src] / sigma_post[src] + x / sigma_obs)
    if n_updates >= r_max - 1:
        src = r_max - 1
        advanced_var = 1.0 / (1.0 / sigma_post[src] + 1.0 / sigma_obs)
        next_var[src] = advanced_var
        next_mu[src] = advanced_var * (mu[src] / sigma_post[src] + x / sigma_obs)


@njit(cache=True)
def _nig_log_pred_kernel(x: float, mu: np.ndarray, kappa: np.ndarray,
                         alpha: np.ndarray, beta: np.ndarray, out: np.ndarray) -> None:
    for i in range(len(mu)):
        df = 2.0 * alpha[i]
        scale_sq = beta[i] * (kappa[i] + 1.0) / (alpha[i] * kappa[i])
        diff = x - mu[i]
        z_sq = (diff * diff) / (df * scale_sq)
        out[i] = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * (np.log(df * np.pi) + np.log(scale_sq))
            - ((df + 1.0) / 2.0) * np.log1p(z_sq)
        )


@njit(cache=True)
def _nig_update_kernel(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                       prior_mu: float, prior_kappa: float, prior_alpha: float, prior_beta: float, n_updates: int,
                       next_mu: np.ndarray, next_kappa: np.ndarray,
                       next_alpha: np.ndarray, next_beta: np.ndarray) -> None:
    r_max = len(mu)
    next_mu[0] = prior_mu
    next_kappa[0] = prior_kappa
    next_alpha[0] = prior_alpha
    next_beta[0] = prior_beta
    for j in range(1, r_max):
        src = j - 1
        advanced_kappa = kappa[src] + 1.0
        diff = x - mu[src]
        next_mu[j] = (kappa[src] * mu[src] + x) / advanced_kappa
        next_kappa[j] = advanced_kappa
        next_alpha[j] = alpha[src] + 0.5
        next_beta[j] = beta[src] + kappa[src] * diff * diff / (2.0 * advanced_kappa)
    if n_updates >= r_max - 1:
        src = r_max - 1
        advanced_kappa = kappa[src] + 1.0
        diff = x - mu[src]
        next_mu[src] = (kappa[src] * mu[src] + x) / advanced_kappa
        next_kappa[src] = advanced_kappa
        next_alpha[src] = alpha[src] + 0.5
        next_beta[src] = beta[src] + kappa[src] * diff * diff / (2.0 * advanced_kappa)


@njit(cache=True)
def _logaddexp_scalar(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@njit(cache=True)
def _student_t_log_pred_scalar(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    df = 2.0 * alpha
    scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
    diff = x - mu
    z_sq = (diff * diff) / (df * scale_sq)
    return (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * (math.log(df * math.pi) + math.log(scale_sq))
        - ((df + 1.0) / 2.0) * math.log1p(z_sq)
    )


@njit(cache=True)
def _student_t_log_pred_precomputed(x: float, mu: float, kappa: float, alpha: float,
                                    beta: float, prior_kappa: float, const_by_count: np.ndarray) -> float:
    count = int(kappa - prior_kappa + 0.5)
    scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
    df = 2.0 * alpha
    diff = x - mu
    z_sq = (diff * diff) / (df * scale_sq)
    return const_by_count[count] - 0.5 * math.log(scale_sq) - ((df + 1.0) / 2.0) * math.log1p(z_sq)


@njit(cache=True)
def _nig_bocpd_batch_kernel(xs: np.ndarray, log_hazard: float, log_growth: float,
                            prior_mu: float, prior_kappa: float, prior_alpha: float, prior_beta: float,
                            posterior: np.ndarray, mu: np.ndarray, kappa: np.ndarray,
                            alpha: np.ndarray, beta: np.ndarray, n_updates: int,
                            next_mu: np.ndarray, next_kappa: np.ndarray,
                            next_alpha: np.ndarray, next_beta: np.ndarray,
                            log_joint: np.ndarray, next_log: np.ndarray,
                            const_by_count: np.ndarray, out: np.ndarray) -> None:
    r_max = len(posterior)
    for t in range(len(xs)):
        x = xs[t]
        joint_max = -np.inf
        for r in range(r_max):
            if posterior[r] > 0.0:
                pred = _student_t_log_pred_precomputed(
                    x,
                    mu[r],
                    kappa[r],
                    alpha[r],
                    beta[r],
                    prior_kappa,
                    const_by_count,
                )
                log_joint[r] = math.log(posterior[r]) + pred
                if log_joint[r] > joint_max:
                    joint_max = log_joint[r]
            else:
                log_joint[r] = -np.inf

        joint_sum = 0.0
        for r in range(r_max):
            if log_joint[r] != -np.inf:
                joint_sum += math.exp(log_joint[r] - joint_max)
        joint_lse = joint_max + math.log(joint_sum)

        next_log[0] = joint_lse + log_hazard
        for r in range(1, r_max):
            next_log[r] = log_joint[r - 1] + log_growth
        next_log[r_max - 1] = _logaddexp_scalar(next_log[r_max - 1], log_joint[r_max - 1] + log_growth)

        norm_max = next_log[0]
        for r in range(1, r_max):
            if next_log[r] > norm_max:
                norm_max = next_log[r]
        norm_sum = 0.0
        for r in range(r_max):
            norm_sum += math.exp(next_log[r] - norm_max)
        log_norm = norm_max + math.log(norm_sum)

        # exp(next_log - log_norm) is self-normalising up to float precision because
        # log_norm == logsumexp(next_log); the previous explicit renorm pass only
        # removed ~r_max * 1e-16 drift, which does not accumulate across steps.
        for r in range(r_max):
            posterior[r] = math.exp(next_log[r] - log_norm)
            out[t, r] = posterior[r]

        _nig_update_kernel(
            x,
            mu,
            kappa,
            alpha,
            beta,
            prior_mu,
            prior_kappa,
            prior_alpha,
            prior_beta,
            n_updates,
            next_mu,
            next_kappa,
            next_alpha,
            next_beta,
        )
        n_updates += 1
        for r in range(r_max):
            mu[r] = next_mu[r]
            kappa[r] = next_kappa[r]
            alpha[r] = next_alpha[r]
            beta[r] = next_beta[r]


@njit(cache=True)
def _gamma_exp_log_pred_kernel(x: float, alpha: np.ndarray, beta: np.ndarray, out: np.ndarray) -> None:
    for i in range(len(alpha)):
        out[i] = math.log(alpha[i]) + alpha[i] * math.log(beta[i]) - (alpha[i] + 1.0) * math.log(beta[i] + x)


@njit(cache=True)
def _gamma_exp_update_kernel(x: float, alpha: np.ndarray, beta: np.ndarray,
                             prior_alpha: float, prior_beta: float, n_updates: int,
                             next_alpha: np.ndarray, next_beta: np.ndarray) -> None:
    r_max = len(alpha)
    next_alpha[0] = prior_alpha
    next_beta[0] = prior_beta
    for j in range(1, r_max):
        src = j - 1
        next_alpha[j] = alpha[src] + 1.0
        next_beta[j] = beta[src] + x
    if n_updates >= r_max - 1:
        src = r_max - 1
        next_alpha[src] = alpha[src] + 1.0
        next_beta[src] = beta[src] + x


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _validate_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _validate_r_max(r_max: int) -> int:
    r_max = int(r_max)
    if r_max < 2:
        raise ValueError(f"r_max must be >= 2, got {r_max}")
    return r_max


def _validate_scalar(x: float, context: str) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{context}: observation must be finite, got {x}")
    return value


class GaussianKnownVarianceModel:
    """Normal-Normal conjugate observation model with known likelihood variance.

    Attribute naming:
        - ``sigma_obs_sq``: scalar likelihood variance (configuration, known a priori).
        - ``sigma0_sq``: scalar prior variance of the mean.
        - ``sigma_sq_``: per-slot posterior variance array (trailing underscore = fitted state).
    """

    def __init__(self, mu_0: float, sigma0_sq: float, sigma_obs_sq: float) -> None:
        self.mu_0 = _validate_finite(mu_0, "mu_0")
        self.sigma0_sq = _validate_positive(sigma0_sq, "sigma0_sq")
        self.sigma_obs_sq = _validate_positive(sigma_obs_sq, "sigma_obs_sq")
        self.r_max: int | None = None
        self._n_updates = 0
        self.mu_: np.ndarray | None = None
        self.sigma_sq_: np.ndarray | None = None
        self._next_mu: np.ndarray | None = None
        self._next_sigma_sq: np.ndarray | None = None
        self._log_pred: np.ndarray | None = None

    def reset(self, r_max: int) -> None:
        self.r_max = _validate_r_max(r_max)
        self._n_updates = 0
        self.mu_ = np.full(self.r_max, self.mu_0, dtype=np.float64)
        self.sigma_sq_ = np.full(self.r_max, self.sigma0_sq, dtype=np.float64)
        self._next_mu = np.empty(self.r_max, dtype=np.float64)
        self._next_sigma_sq = np.empty(self.r_max, dtype=np.float64)
        self._log_pred = np.empty(self.r_max, dtype=np.float64)

    def log_pred_probs(self, x: float) -> NDArray:
        """Return log predictive probabilities for ``x`` at each run-length slot.

        The returned array is a view into an internal buffer; it is overwritten on
        the next call to ``log_pred_probs``. Copy the result if you need to keep it
        alongside subsequent calls.
        """
        value = _validate_scalar(x, "GaussianKnownVarianceModel.log_pred_probs")
        mu, sigma_post = self._require_state()
        assert self._log_pred is not None
        _gaussian_log_pred_kernel(value, mu, sigma_post, self.sigma_obs_sq, self._log_pred)
        return self._log_pred

    def update(self, x: float) -> None:
        value = _validate_scalar(x, "GaussianKnownVarianceModel.update")
        mu, sigma_post = self._require_state()
        assert self._next_mu is not None and self._next_sigma_sq is not None
        _gaussian_update_kernel(
            value,
            mu,
            sigma_post,
            self.sigma_obs_sq,
            self.mu_0,
            self.sigma0_sq,
            self._n_updates,
            self._next_mu,
            self._next_sigma_sq,
        )
        self.mu_, self._next_mu = self._next_mu, self.mu_
        self.sigma_sq_, self._next_sigma_sq = self._next_sigma_sq, self.sigma_sq_
        self._n_updates += 1

    def _require_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mu_ is None or self.sigma_sq_ is None:
            raise RuntimeError("Model has not been reset. Call reset(r_max) before use.")
        return self.mu_, self.sigma_sq_


class NormalInverseGammaModel:
    """Normal-Inverse-Gamma conjugate model with Student-T posterior predictive."""

    def __init__(self, mu_0: float = 0.0, kappa_0: float = 1.0, alpha_0: float = 1.0, beta_0: float = 1.0) -> None:
        self.mu_0 = _validate_finite(mu_0, "mu_0")
        self.kappa_0 = _validate_positive(kappa_0, "kappa_0")
        self.alpha_0 = _validate_positive(alpha_0, "alpha_0")
        self.beta_0 = _validate_positive(beta_0, "beta_0")
        self.r_max: int | None = None
        self._n_updates = 0
        self.mu_: np.ndarray | None = None
        self.kappa_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self._next_mu: np.ndarray | None = None
        self._next_kappa: np.ndarray | None = None
        self._next_alpha: np.ndarray | None = None
        self._next_beta: np.ndarray | None = None
        self._log_pred: np.ndarray | None = None
        self._log_joint: np.ndarray | None = None
        self._next_log: np.ndarray | None = None

    def reset(self, r_max: int) -> None:
        self.r_max = _validate_r_max(r_max)
        self._n_updates = 0
        self.mu_ = np.full(self.r_max, self.mu_0, dtype=np.float64)
        self.kappa_ = np.full(self.r_max, self.kappa_0, dtype=np.float64)
        self.alpha_ = np.full(self.r_max, self.alpha_0, dtype=np.float64)
        self.beta_ = np.full(self.r_max, self.beta_0, dtype=np.float64)
        self._next_mu = np.empty(self.r_max, dtype=np.float64)
        self._next_kappa = np.empty(self.r_max, dtype=np.float64)
        self._next_alpha = np.empty(self.r_max, dtype=np.float64)
        self._next_beta = np.empty(self.r_max, dtype=np.float64)
        self._log_pred = np.empty(self.r_max, dtype=np.float64)
        self._log_joint = np.empty(self.r_max, dtype=np.float64)
        self._next_log = np.empty(self.r_max, dtype=np.float64)

    def log_pred_probs(self, x: float) -> NDArray:
        """Return log predictive probabilities for ``x`` at each run-length slot.

        The returned array is a view into an internal buffer; it is overwritten on
        the next call to ``log_pred_probs``. Copy the result if you need to keep it
        alongside subsequent calls.
        """
        value = _validate_scalar(x, "NormalInverseGammaModel.log_pred_probs")
        mu, kappa, alpha, beta = self._require_state()
        assert self._log_pred is not None
        _nig_log_pred_kernel(value, mu, kappa, alpha, beta, self._log_pred)
        return self._log_pred

    def update(self, x: float) -> None:
        value = _validate_scalar(x, "NormalInverseGammaModel.update")
        mu, kappa, alpha, beta = self._require_state()
        assert self._next_mu is not None and self._next_kappa is not None
        assert self._next_alpha is not None and self._next_beta is not None
        _nig_update_kernel(
            value,
            mu,
            kappa,
            alpha,
            beta,
            self.mu_0,
            self.kappa_0,
            self.alpha_0,
            self.beta_0,
            self._n_updates,
            self._next_mu,
            self._next_kappa,
            self._next_alpha,
            self._next_beta,
        )
        self.mu_, self._next_mu = self._next_mu, self.mu_
        self.kappa_, self._next_kappa = self._next_kappa, self.kappa_
        self.alpha_, self._next_alpha = self._next_alpha, self.alpha_
        self.beta_, self._next_beta = self._next_beta, self.beta_
        self._n_updates += 1

    def batch_update_posterior(self, xs: NDArray, posterior: NDArray, log_hazard: float, log_growth: float) -> NDArray:
        """Run the full Adams-MacKay recursion over ``xs`` using the fused NIG kernel.

        Mutates ``posterior`` in place to hold the final run-length posterior, advances
        the observation model's sufficient-statistic bank by ``len(xs)`` updates, and
        returns a ``(len(xs), r_max)`` matrix of per-step posteriors.

        Callers (detector.batch) are expected to have already validated ``xs`` for
        shape, dtype, and finiteness — this method trusts its input.
        """
        values = np.asarray(xs, dtype=np.float64)
        mu, kappa, alpha, beta = self._require_state()
        if posterior.shape != mu.shape:
            raise ValueError(f"posterior shape {posterior.shape} does not match model capacity {mu.shape}")
        out = np.empty((len(values), len(posterior)), dtype=np.float64)
        if len(values) == 0:
            return out
        # Max count any slot can reach after this batch: n_updates already absorbed
        # plus one more per step in the batch. Over-provision by +1 only to let the
        # lookup use count indices without boundary branching.
        max_count = self._n_updates + len(values) + 1
        counts = np.arange(max_count + 1, dtype=np.float64)
        alpha_values = self.alpha_0 + 0.5 * counts
        df_values = 2.0 * alpha_values
        const_by_count = (
            gammaln((df_values + 1.0) / 2.0)
            - gammaln(df_values / 2.0)
            - 0.5 * np.log(df_values * np.pi)
        ).astype(np.float64)
        assert self._next_mu is not None and self._next_kappa is not None
        assert self._next_alpha is not None and self._next_beta is not None
        assert self._log_joint is not None and self._next_log is not None
        _nig_bocpd_batch_kernel(
            values,
            float(log_hazard),
            float(log_growth),
            self.mu_0,
            self.kappa_0,
            self.alpha_0,
            self.beta_0,
            posterior,
            mu,
            kappa,
            alpha,
            beta,
            self._n_updates,
            self._next_mu,
            self._next_kappa,
            self._next_alpha,
            self._next_beta,
            self._log_joint,
            self._next_log,
            const_by_count,
            out,
        )
        self._n_updates += len(values)
        # Guard the asymmetry with single-step update(): the njit kernel cannot raise,
        # so pathological inputs (all-zero incoming posterior, numeric underflow) would
        # silently produce NaN. Catch here so callers see a clear failure.
        if not np.isfinite(out.sum()):
            raise ValueError(
                "NormalInverseGammaModel.batch_update_posterior produced non-finite "
                "run-length posterior; incoming posterior may be degenerate."
            )
        return out

    def _require_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.mu_ is None or self.kappa_ is None or self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("Model has not been reset. Call reset(r_max) before use.")
        return self.mu_, self.kappa_, self.alpha_, self.beta_


class GammaExponentialModel:
    """Gamma-Exponential conjugate model with Lomax posterior predictive."""

    def __init__(self, alpha_0: float, beta_0: float) -> None:
        self.alpha_0 = _validate_positive(alpha_0, "alpha_0")
        self.beta_0 = _validate_positive(beta_0, "beta_0")
        self.r_max: int | None = None
        self._n_updates = 0
        self.alpha_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self._next_alpha: np.ndarray | None = None
        self._next_beta: np.ndarray | None = None
        self._log_pred: np.ndarray | None = None

    def reset(self, r_max: int) -> None:
        self.r_max = _validate_r_max(r_max)
        self._n_updates = 0
        self.alpha_ = np.full(self.r_max, self.alpha_0, dtype=np.float64)
        self.beta_ = np.full(self.r_max, self.beta_0, dtype=np.float64)
        self._next_alpha = np.empty(self.r_max, dtype=np.float64)
        self._next_beta = np.empty(self.r_max, dtype=np.float64)
        self._log_pred = np.empty(self.r_max, dtype=np.float64)

    def log_pred_probs(self, x: float) -> NDArray:
        """Return log predictive probabilities for ``x`` at each run-length slot.

        The returned array is a view into an internal buffer; it is overwritten on
        the next call to ``log_pred_probs``. Copy the result if you need to keep it
        alongside subsequent calls.
        """
        value = self._validate_positive_observation(x, "GammaExponentialModel.log_pred_probs")
        alpha, beta = self._require_state()
        assert self._log_pred is not None
        _gamma_exp_log_pred_kernel(value, alpha, beta, self._log_pred)
        return self._log_pred

    def update(self, x: float) -> None:
        value = self._validate_positive_observation(x, "GammaExponentialModel.update")
        alpha, beta = self._require_state()
        assert self._next_alpha is not None and self._next_beta is not None
        _gamma_exp_update_kernel(
            value,
            alpha,
            beta,
            self.alpha_0,
            self.beta_0,
            self._n_updates,
            self._next_alpha,
            self._next_beta,
        )
        self.alpha_, self._next_alpha = self._next_alpha, self.alpha_
        self.beta_, self._next_beta = self._next_beta, self.beta_
        self._n_updates += 1

    def _require_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("Model has not been reset. Call reset(r_max) before use.")
        return self.alpha_, self.beta_

    @staticmethod
    def _validate_positive_observation(x: float, context: str) -> float:
        value = _validate_scalar(x, context)
        if value <= 0.0:
            raise ValueError(f"{context}: observation must be > 0, got {x}")
        return value
