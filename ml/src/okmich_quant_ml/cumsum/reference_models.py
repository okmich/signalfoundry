from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Sided(StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    TWO = "two"


def _validate_sided(sided: str | Sided) -> Sided:
    try:
        return Sided(sided)
    except ValueError as exc:
        raise ValueError(f"sided must be one of 'upper', 'lower', 'two', got {sided!r}") from exc


def _n_directions(sided: Sided) -> int:
    return 2 if sided is Sided.TWO else 1


def _validate_finite_scalar(x: float, context: str) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{context}: must be finite, got {x}")
    return value


class GaussianReferenceModel:
    """Univariate Gaussian reference (known mu_0, sigma)."""

    def __init__(self, mu_0: float, sigma: float, sided: str | Sided = Sided.TWO) -> None:
        self.mu_0 = _validate_finite_scalar(mu_0, "mu_0")
        self.sigma = _validate_finite_scalar(sigma, "sigma")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sided = _validate_sided(sided)
        self._n_directions = _n_directions(self.sided)
        self._buf = np.empty(self._n_directions, dtype=np.float64)

    def score(self, x: float) -> NDArray:
        z = (float(x) - self.mu_0) / self.sigma
        if self.sided is Sided.UPPER:
            self._buf[0] = z
        elif self.sided is Sided.LOWER:
            self._buf[0] = -z
        else:
            self._buf[0] = z
            self._buf[1] = -z
        return self._buf

    def update(self, x: float) -> None:  # noqa: ARG002 — static reference
        return None

    def reset(self) -> None:
        return None

    @property
    def n_directions(self) -> int:
        return self._n_directions

    @property
    def requires_external_sampler(self) -> bool:
        return False

    def sample_in_control(self, n: int, rng: np.random.Generator) -> NDArray:
        return rng.normal(loc=self.mu_0, scale=self.sigma, size=int(n))


class EwmaReferenceModel:
    """Univariate Gaussian reference with EWMA-adaptive mu_hat and sigma_hat.

    The variance recursion is unclipped so the EWMA can recover when variance
    returns; the score uses ``sigma_effective = max(min_sigma, sqrt(sigma_hat^2))``
    to guard against division by near-zero σ̂ on long flat stretches.
    """

    def __init__(self, mu_0: float, sigma_0: float, alpha_mu: float, alpha_sigma: float,
                 min_sigma: float = 1e-6, sided: str | Sided = Sided.TWO) -> None:
        self.mu_0 = _validate_finite_scalar(mu_0, "mu_0")
        self.sigma_0 = _validate_finite_scalar(sigma_0, "sigma_0")
        if self.sigma_0 <= 0.0:
            raise ValueError(f"sigma_0 must be > 0, got {sigma_0}")
        self.alpha_mu = _validate_finite_scalar(alpha_mu, "alpha_mu")
        self.alpha_sigma = _validate_finite_scalar(alpha_sigma, "alpha_sigma")
        if not 0.0 < self.alpha_mu < 1.0:
            raise ValueError(f"alpha_mu must be in (0, 1), got {alpha_mu}")
        if not 0.0 < self.alpha_sigma < 1.0:
            raise ValueError(f"alpha_sigma must be in (0, 1), got {alpha_sigma}")
        self.min_sigma = _validate_finite_scalar(min_sigma, "min_sigma")
        if self.min_sigma <= 0.0:
            raise ValueError(f"min_sigma must be > 0, got {min_sigma}")
        self.sided = _validate_sided(sided)
        self._n_directions = _n_directions(self.sided)
        self._buf = np.empty(self._n_directions, dtype=np.float64)
        self.reset()

    def score(self, x: float) -> NDArray:
        sigma_eff = self._sigma_hat if self._sigma_hat > self.min_sigma else self.min_sigma
        z = (float(x) - self._mu_hat) / sigma_eff
        if self.sided is Sided.UPPER:
            self._buf[0] = z
        elif self.sided is Sided.LOWER:
            self._buf[0] = -z
        else:
            self._buf[0] = z
            self._buf[1] = -z
        return self._buf

    def update(self, x: float) -> None:
        value = float(x)
        residual = value - self._mu_hat
        # Variance update uses the *prior* mu_hat residual (read-before-update on mu),
        # then mu_hat advances. Recursion stays unclipped so it can recover.
        self._sigma_hat_sq = (1.0 - self.alpha_sigma) * self._sigma_hat_sq + self.alpha_sigma * residual * residual
        self._sigma_hat = float(np.sqrt(self._sigma_hat_sq))
        self._mu_hat = (1.0 - self.alpha_mu) * self._mu_hat + self.alpha_mu * value

    def reset(self) -> None:
        self._mu_hat = self.mu_0
        self._sigma_hat = self.sigma_0
        self._sigma_hat_sq = self.sigma_0 * self.sigma_0

    @property
    def mu_hat(self) -> float:
        return self._mu_hat

    @property
    def sigma_hat(self) -> float:
        return self._sigma_hat

    @property
    def n_directions(self) -> int:
        return self._n_directions

    @property
    def requires_external_sampler(self) -> bool:
        return True

    def sample_in_control(self, n: int, rng: np.random.Generator) -> NDArray:
        raise NotImplementedError(
            "EwmaReferenceModel has no canonical H0 sampler; pass `in_control_sampler` to target_arl_threshold."
        )


class SignCusumReferenceModel:
    """Univariate distribution-free sign CUSUM (median_0)."""

    def __init__(self, median_0: float, sided: str | Sided = Sided.TWO) -> None:
        self.median_0 = _validate_finite_scalar(median_0, "median_0")
        self.sided = _validate_sided(sided)
        self._n_directions = _n_directions(self.sided)
        self._buf = np.empty(self._n_directions, dtype=np.float64)

    def score(self, x: float) -> NDArray:
        s = float(np.sign(float(x) - self.median_0))
        upper = max(0.0, s)
        lower = max(0.0, -s)
        if self.sided is Sided.UPPER:
            self._buf[0] = upper
        elif self.sided is Sided.LOWER:
            self._buf[0] = lower
        else:
            self._buf[0] = upper
            self._buf[1] = lower
        return self._buf

    def update(self, x: float) -> None:  # noqa: ARG002 — static reference
        return None

    def reset(self) -> None:
        return None

    @property
    def n_directions(self) -> int:
        return self._n_directions

    @property
    def requires_external_sampler(self) -> bool:
        return False

    def sample_in_control(self, n: int, rng: np.random.Generator) -> NDArray:
        # Two-point sampler around median_0 — values placed *around* the configured
        # median, so the sampler works for any median_0. Sign stream is identical to
        # any continuous distribution with median == median_0.
        offsets = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=int(n))
        return self.median_0 + offsets
