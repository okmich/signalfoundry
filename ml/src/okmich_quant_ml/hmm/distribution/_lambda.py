import numpy as np
import torch
from pomegranate._utils import _check_parameter, _cast_as_parameter, _cast_as_tensor
from pomegranate.distributions._distribution import Distribution
from torch.special import gammaln


class Lambda(Distribution):
    """
    A Symmetric Lambda Distribution (exponential power distribution).

    The symmetric lambda distribution is a flexible distribution that can represent both light-tailed (lambda > 2)
    and heavy-tailed (lambda < 2) distributions. It includes the normal distribution (lambda=2) and Laplace distribution
    (lambda=1) as special cases.

    PDF: P(x; μ, σ, λ) = 1/(σ * λ * Γ(λ/2)) * exp(-|(x-μ)/σ|^(2/λ))

        There are two ways to initialize this object. The first is to pass in the tensor of probability parameters, at which
        point they can immediately be used. The second is to not pass in the rate parameters and then call either `fit` or
        `summary` + `from_summaries`, at which point the probability parameter will be learned from data.

        Lambda Value Interpretation:
        - λ ≈ 1.0-1.5: Very heavy tails (crashes, bubbles, extreme events)
    - λ ≈ 1.5-2.0: Heavy tails (typical financial returns)
    - λ ≈ 2.0: Normal distribution
    - λ > 2.0: Light tails (low volatility periods)

    Parameters
    ----------
    mu: float, torch.Tensor, optional
        The location parameter (mean). Default is 0.0.

    sigma: float, torch.Tensor, optional
        The scale parameter. Default is 1.0.

    lambda_: float, torch.Tensor, optional
        The shape parameter (λ ≥ 1). Default is 2.0.

    inertia: float, [0, 1], optional
        The inertia for updating parameters. Default is 0.0.

    frozen: bool, optional
        Whether the parameters are frozen. Default is False.

    check_data: bool, optional
        Whether to check input data. Default is True.

    min_sigma: float, optional
        The minimum value for the scale parameter. Default is 1e-6.

    max_lambda: float, optional
        The maximum value for the shape parameter. Default is 10.0.
    """

    def __init__(
        self,
        mu=None,
        sigma=None,
        lambda_=None,
        inertia=0.0,
        frozen=False,
        check_data=True,
        min_sigma=1e-6,
        max_lambda=10.0,
        dtype=torch.float64,
    ):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "Lambda"

        # Store dtype for initialization (can't set dtype directly as it's a property)
        self._dtype_init = dtype

        self.min_sigma = min_sigma
        self.max_lambda = max_lambda

        self._mu_init = mu
        self._sigma_init = sigma
        self._lambda_init = lambda_

        self._initialized = False
        self.d = None

    def _initialize(self, d):
        """Initialize the distribution for d dimensions."""
        if self._mu_init is not None:
            mu = np.array(self._mu_init)
            if mu.ndim == 0 or len(mu) == 1:
                mu = np.full(d, mu.item() if hasattr(mu, "item") else mu)
            elif len(mu) != d:
                raise ValueError(
                    f"Initial mu has {len(mu)} elements but data has {d} features"
                )
        else:
            mu = np.zeros(d)

        if self._sigma_init is not None:
            sigma = np.array(self._sigma_init)
            if sigma.ndim == 0 or len(sigma) == 1:
                sigma = np.full(d, sigma.item() if hasattr(sigma, "item") else sigma)
            elif len(sigma) != d:
                raise ValueError(
                    f"Initial sigma has {len(sigma)} elements but data has {d} features"
                )
        else:
            sigma = np.ones(d)

        if self._lambda_init is not None:
            lambda_ = np.array(self._lambda_init)
            if lambda_.ndim == 0 or len(lambda_) == 1:
                lambda_ = np.full(
                    d, lambda_.item() if hasattr(lambda_, "item") else lambda_
                )
            elif len(lambda_) != d:
                raise ValueError(
                    f"Initial lambda_ has {len(lambda_)} elements but data has {d} features"
                )
        else:
            lambda_ = np.full(d, 2.0)

        # Create parameters with explicit dtype
        self.mu = _check_parameter(_cast_as_parameter(torch.tensor(mu, dtype=self._dtype_init)), "mu", ndim=1)
        self.sigma = _check_parameter(
            _cast_as_parameter(torch.tensor(sigma, dtype=self._dtype_init)), "sigma", ndim=1, min_value=self.min_sigma
        )
        self.lambda_ = _check_parameter(
            _cast_as_parameter(torch.tensor(lambda_, dtype=self._dtype_init)),
            "lambda_",
            ndim=1,
            min_value=1.0,
            max_value=self.max_lambda,
        )

        self._initialized = True
        self.d = d
        self._reset_cache()

    def _reset_cache(self):
        """Reset the internal cache and precompute constants."""
        if not self._initialized:
            return

        # Ensure parameters are valid for each dimension
        with torch.no_grad():
            self.sigma.data = torch.clamp(self.sigma.data, min=self.min_sigma)
            self.lambda_.data = torch.clamp(
                self.lambda_.data, min=1.0, max=self.max_lambda
            )

        # Precompute constants for each dimension
        log_norm = (
            -torch.log(self.sigma) - torch.log(self.lambda_) - gammaln(self.lambda_ / 2)
        )
        exponent = 2 / self.lambda_
        inv_exponent = self.lambda_ / 2

        self.register_buffer("_log_norm", log_norm)
        self.register_buffer("_exponent", exponent)
        self.register_buffer("_inv_exponent", inv_exponent)

        # Initialize dimension-specific buffers
        for dim in range(self.d):
            if not hasattr(self, f"_X_buffer_dim_{dim}"):
                setattr(self, f"_X_buffer_dim_{dim}", [])
                setattr(self, f"_weights_buffer_dim_{dim}", [])
                setattr(
                    self,
                    f"_n_dim_{dim}",
                    torch.tensor(0, dtype=torch.int32, device=self.device),
                )

    def log_probability(self, X):
        """Calculate the log probability of each example."""
        X = _check_parameter(
            _cast_as_tensor(X, dtype=self.mu.dtype),
            "X",
            ndim=2,
            shape=(-1, self.d),
            check_parameter=self.check_data,
        )

        if self.d == 1:
            # Univariate case - squeeze and compute
            X = X.squeeze(1)
            z = torch.abs((X - self.mu) / self.sigma)
            z_power = z**self._exponent
            return self._log_norm - z_power
        else:
            # Multivariate case
            z = torch.abs((X - self.mu) / self.sigma)
            z_power = z**self._exponent
            log_probs = self._log_norm - z_power
            return torch.sum(log_probs, dim=1)

    def summarize(self, X, sample_weight=None):
        """Extract sufficient statistics from data."""
        if self.frozen:
            return

        X = _cast_as_tensor(X)

        if not self._initialized:
            d = X.shape[1] if X.ndim == 2 else 1
            self._initialize(d)

        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        X = _cast_as_tensor(X, dtype=self.mu.dtype)
        sample_weight = _cast_as_tensor(sample_weight, dtype=self.mu.dtype)

        if not hasattr(self, "_X_buffer"):
            self._X_buffer = [[] for _ in range(self.d)]
            self._weights_buffer = [[] for _ in range(self.d)]
            self._n = [0] * self.d

        for dim in range(self.d):
            X_dim = X[:, dim].cpu().numpy()

            # Handle weights
            if sample_weight.ndim == 1:
                weights_dim = sample_weight.cpu().numpy()
            else:
                weights_dim = (
                    sample_weight[:, dim].cpu().numpy()
                    if sample_weight.shape[1] > 1
                    else sample_weight.squeeze(1).cpu().numpy()
                )

            self._X_buffer[dim].append(X_dim)
            self._weights_buffer[dim].append(weights_dim)
            self._n[dim] += len(X_dim)

    def from_summaries(self):
        """Update parameters from accumulated statistics."""
        if self.frozen:
            return

        # Update each dimension separately
        for dim in range(self.d):
            if self._n[dim] == 0:
                continue

            X_dim = np.concatenate(self._X_buffer[dim])
            weights_dim = np.concatenate(self._weights_buffer[dim])

            if weights_dim.ndim == 0 or len(weights_dim) != len(X_dim):
                weights_dim = np.ones_like(X_dim)

            # Your robust parameter estimation logic here
            new_mu = float(np.average(X_dim, weights=weights_dim))
            new_sigma = float(max(np.std(X_dim), self.min_sigma))

            # Lambda estimation logic
            abs_dev = np.abs(X_dim - new_mu)
            q90 = np.percentile(abs_dev, 90)
            q50 = np.percentile(abs_dev, 50)

            if q50 > 1e-10:
                tail_ratio = q90 / q50
                if tail_ratio > 3.0:
                    new_lambda_ = 1.2
                elif tail_ratio > 2.0:
                    new_lambda_ = 1.5
                elif tail_ratio > 1.5:
                    new_lambda_ = 1.8
                else:
                    new_lambda_ = 2.0
            else:
                new_lambda_ = 2.0

            # Update parameters for this dimension
            self.mu.data[dim] = new_mu
            self.sigma.data[dim] = max(new_sigma, self.min_sigma)
            self.lambda_.data[dim] = np.clip(new_lambda_, 1.0, self.max_lambda)

            # Reset buffers for this dimension
            self._X_buffer[dim] = []
            self._weights_buffer[dim] = []
            self._n[dim] = 0

        self._reset_cache()

    def sample(self, n):
        """Sample from the distribution.

        Parameters
        ----------
        n: int
            The number of samples to generate.

        Returns
        -------
        X: torch.Tensor, shape=(n, 1)
            Randomly generated samples in 2D format.
        """
        n = int(n)

        # Use inverse transform sampling with numerical stability
        u = torch.rand(n, dtype=self.dtype, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) > 0.5, 1.0, -1.0)

        u = torch.clamp(u, 1e-10, 1 - 1e-10)
        samples = (
            self.mu + self.sigma * sign * (-torch.log(1 - u)) ** self._inv_exponent
        )
        return samples.unsqueeze(1)

    def to_json(self):
        """Return the distribution as a JSON-serializable object."""
        return {
            "class": "Lambda",
            "parameters": {
                "mu": self.mu.item(),
                "sigma": self.sigma.item(),
                "lambda_": self.lambda_.item(),
            },
            "inertia": self.inertia,
            "frozen": self.frozen,
            "min_sigma": self.min_sigma,
            "max_lambda": self.max_lambda,
            "dtype": str(self._dtype_init),
        }

    @classmethod
    def from_json(cls, d):
        """Create a distribution from a JSON object."""
        dtype_str = d.get("dtype", "torch.float32")
        dtype = torch.float64 if "float64" in dtype_str else torch.float32
        return cls(
            mu=d["parameters"]["mu"],
            sigma=d["parameters"]["sigma"],
            lambda_=d["parameters"]["lambda_"],
            inertia=d.get("inertia", 0.0),
            frozen=d.get("frozen", False),
            min_sigma=d.get("min_sigma", 1e-6),
            max_lambda=d.get("max_lambda", 10.0),
            dtype=dtype,
        )
