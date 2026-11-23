"""Symmetric exponential-power ("Lambda") emission distribution for pomegranate models (DistType.LAMDA).

Used by every pomegranate-backed model in this package: ``PomegranateHMM``, the mixture-emission HMM
(``pomegranate_mm``), the IOHMM and anything composed from them. They all refit emissions through pomegranate's
weighted ``summarize`` / ``from_summaries`` protocol, so the posterior weights handed to ``summarize`` must drive
every parameter. That is what this module guarantees.
"""
from __future__ import annotations

import numpy as np
import torch
from pomegranate._utils import _cast_as_parameter, _cast_as_tensor, _check_parameter, _update_parameter
from pomegranate.distributions._distribution import Distribution
from scipy.optimize import minimize_scalar
from scipy.special import gammaln as _np_gammaln
from torch.special import gammaln

_MIN_SHAPE_ESS = 10.0       # effective sample size below which the shape is held (location and scale still update)
_SHAPE_GRID_SIZE = 13       # coarse log-spaced grid that brackets the profile-likelihood maximum before Brent
_SHAPE_LOCAL_SPAN = 0.7     # warm-started shape search: Brent over log(lambda) +- this, full grid if it ends on an edge
_MAX_OUTER_ITER = 100       # coordinate-ascent sweeps (location -> shape -> scale) per M-step
_OUTER_TOL = 1e-10          # relative log-likelihood change that ends the sweeps
_XATOL = 1e-9               # location tolerance, in units of the weighted SD
_LOG2 = float(np.log(2.0))
LAMBDA_KWARGS = ("min_sigma", "min_lambda", "max_lambda", "inertia")   # model-level dist_kwargs forwarded to Lambda


def lambda_kwargs(dist_kwargs: dict) -> dict:
    """The subset of a model's ``dist_kwargs`` that configures ``Lambda`` (bounds, scale floor, inertia)."""
    return {k: dist_kwargs[k] for k in LAMBDA_KWARGS if k in dist_kwargs}


# Every sum of powers is taken in log space. With concentrated weights, a point far out in the tail can carry
# |r|^p ~ 1e400 against a weight of 1e-100, which overflows if the power is formed before the weight is applied.
def _lse(a: np.ndarray) -> float:
    """log(sum(exp(a))) without overflow; -inf when every term is -inf (or there are none)."""
    if a.size == 0:
        return -np.inf
    top = float(a.max())
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.exp(a - top).sum()))


def _log_abs(r: np.ndarray) -> np.ndarray:
    """log|r|, with -inf for r = 0 (that term then drops out of every power sum, for any p > 0)."""
    with np.errstate(divide="ignore"):
        return np.log(np.abs(r))


def _log_scale(log_s: float, log_W: float, p: float, log_lo: float, log_hi: float) -> float:
    """log of the scale MLE given the shape and location, constrained to [lo, hi].

    Unconstrained, sigma^p = p * S / W with S = sum(w |r|^p). The log-likelihood is unimodal in log(sigma), so clipping
    the unconstrained optimum gives the constrained one. S = 0 (all residuals zero) gives the lower bound.
    """
    return float(np.clip((np.log(p) + log_s - log_W) / p, log_lo, log_hi))


def _log_lik(log_s: float, W: float, p: float, log_sigma: float) -> float:
    """Weighted log-likelihood from S = sum(w |r|^p) (passed as log S), the total weight W, exponent p and scale."""
    return float(W * (np.log(p) - _LOG2 - log_sigma - _np_gammaln(1.0 / p)) - np.exp(log_s - p * log_sigma))


def _weighted_quantile(z_sorted: np.ndarray, cw: np.ndarray, q: float) -> float:
    return float(z_sorted[min(np.searchsorted(cw, q * cw[-1]), len(z_sorted) - 1)])


def _location(z: np.ndarray, z_sorted: np.ndarray, w: np.ndarray, lw: np.ndarray, m: float, p: float,
              bracket: tuple[float, float]) -> float:
    """Location update: minimise sum(w |z - c|^p) over c, never returning a point worse than the current ``m``. The
    minimiser does not depend on the scale, so the step is the same whether or not the scale is constrained.

    - p = 2: the weighted mean (exact).
    - p > 1: the objective is strictly convex, so bounded Brent over ``bracket`` (the data range) finds the global
      minimum to within its tolerance.
    - p <= 1 (Laplace and heavier): the objective is linear (p = 1) or concave (p < 1) between consecutive data points,
      so every local minimum is a data point. Brent over ``bracket`` (the data range at p = 1, the weighted inter-decile
      range when p < 1) gets close; the step then returns the best of the data points either side of Brent's estimate,
      exactly. A continuous estimate ~1e-14 off the point would lose likelihood at the density's cusp. When p < 1 the
      minimum is local (near the bulk), not necessarily global.

    Brent minimises log(sum(w |z - c|^p)), which has the same minimiser and cannot overflow.
    """
    if p == 2.0:
        return float(w @ z / w.sum())

    def f(c):
        return _lse(lw + p * _log_abs(z - c))

    res = minimize_scalar(f, bounds=bracket, method="bounded", options={"xatol": _XATOL})
    candidates = [m] + ([float(res.x)] if np.isfinite(res.fun) else [])
    if p <= 1.0:
        k = int(np.searchsorted(z_sorted, res.x))
        candidates += z_sorted[max(k - 2, 0):k + 2].tolist()
    return min(candidates, key=f)                       # ties keep the current point


def _profile_shape_ll(log_r: np.ndarray, lw: np.ndarray, W: float, lam: float, log_lo: float, log_hi: float) -> float:
    """Weighted log-likelihood at shape ``lam``, location fixed, with the scale at its MLE within [lo, hi] (the floor
    ``min_sigma``, or a frozen value when lo = hi). ``log_r`` holds log|residual| and ``lw`` log(weight)."""
    p = 2.0 / lam
    log_s = _lse(lw + p * log_r)
    return _log_lik(log_s, W, p, _log_scale(log_s, float(np.log(W)), p, log_lo, log_hi))


def _shape(log_r: np.ndarray, lw: np.ndarray, W: float, lam: float, bounds: tuple[float, float],
           log_scale_bounds: tuple[float, float], warm: bool) -> float:
    """Profile-likelihood MLE of the shape within ``bounds``, searched in log(lambda), never worse than the current
    ``lam``. Candidates with a non-finite likelihood are rejected.

    Warm start (later sweeps): bounded Brent within ``_SHAPE_LOCAL_SPAN`` of the current value. Cold start, or a warm
    search that ends on its bracket edge: a coarse grid over the bounds brackets the peak, then Brent.
    """
    def neg(t):
        return -_profile_shape_ll(log_r, lw, W, float(np.exp(t)), *log_scale_bounds)

    t_min, t_max, t_now = float(np.log(bounds[0])), float(np.log(bounds[1])), float(np.log(lam))
    f_now = neg(t_now)

    def better(f_new):
        return bool(np.isfinite(f_new) and f_new <= f_now)

    if warm:
        lo, hi = max(t_min, t_now - _SHAPE_LOCAL_SPAN), min(t_max, t_now + _SHAPE_LOCAL_SPAN)
        res = minimize_scalar(neg, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        on_edge = (res.x - lo < 1e-3 and lo > t_min) or (hi - res.x < 1e-3 and hi < t_max)
        if not on_edge:
            return float(np.exp(res.x)) if better(res.fun) else lam
    grid = np.linspace(t_min, t_max, _SHAPE_GRID_SIZE)
    vals = np.array([neg(t) for t in grid])
    if not np.isfinite(vals).any():
        return lam
    i = int(np.argmin(np.where(np.isfinite(vals), vals, np.inf)))
    res = minimize_scalar(neg, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded",
                          options={"xatol": 1e-8})
    t_best, f_best = (res.x, res.fun) if np.isfinite(res.fun) and res.fun <= vals[i] else (grid[i], vals[i])
    return float(np.exp(t_best)) if better(f_best) else lam


def _weighted_mle(x: np.ndarray, w: np.ndarray, mu0: float, sigma0: float, lam0: float, min_lambda: float,
                  max_lambda: float, min_sigma: float, fix_mu: bool = False, fix_sigma: bool = False,
                  fix_lambda: bool = False) -> tuple[float, float, float]:
    """Posterior-weighted maximum-likelihood (mu, sigma, lambda) of one dimension, subject to sigma >= ``min_sigma``,
    lambda in [``min_lambda``, ``max_lambda``], and any parameter flagged ``fix_*`` held at its current value.

    Coordinate ascent on the constrained weighted log-likelihood:
    - location: global for p >= 1, a local minimum on a data point for p < 1 (``_location``); skipped when fixed;
    - shape: profile likelihood with the scale profiled out inside its constraint (``_shape``); skipped when fixed or
      when the effective sample size (sum w)^2 / sum(w^2) is below ``_MIN_SHAPE_ESS``;
    - scale: closed form, sigma^p = p * sum(w |x - mu|^p) / sum(w) with p = 2 / lambda, clipped to the floor (or
      pinned when fixed).

    This is an improving update, not a guaranteed global maximiser: when p < 1 the location is a local minimum, a held
    shape stays where it is, and the sweeps stop at a tolerance. The current parameters are admissible and no step
    lowers the constrained likelihood. The returned parameters are checked again in the original units, and the
    current ones are kept if the update would lower the likelihood there, so the result is never worse than the start.
    A location that stays put or lands on a data point is returned exactly, not via the standardised units.

    The fit works on data standardised by the weighted mean and SD, with weights normalised to a maximum of 1 and every
    power sum in log space, so it is safe at any data or weight scale.
    """
    keep = w > 0
    x, lw = x[keep], np.log(w[keep])
    lw = lw - lw.max()
    w = np.exp(lw)
    W = float(w.sum())
    log_W = float(np.log(W))
    centre = float(w @ x / W)
    dev = x - centre
    a = float(np.abs(dev).max())
    spread = a * float(np.sqrt(w @ (dev / a) ** 2 / W)) if a > 0.0 else 0.0
    if not (np.isfinite(spread) and spread > 0.0):
        spread = 1.0                                        # every weighted point is identical: z = 0
    z = dev / spread
    log_spread = float(np.log(spread))
    log_scale_bounds = (float(np.log(min_sigma)) - log_spread, np.inf)       # z units; a ratio could underflow
    if fix_sigma:
        log_scale_bounds = (float(np.log(sigma0)) - log_spread,) * 2
    fit_shape = not fix_lambda and W ** 2 / float(w @ w) >= _MIN_SHAPE_ESS
    lam = float(np.clip(lam0, min_lambda, max_lambda))

    order = np.argsort(z)
    z_sorted, cw = z[order], np.cumsum(w[order])
    full_range = (float(z_sorted[0]), float(z_sorted[-1]))
    deciles = (_weighted_quantile(z_sorted, cw, 0.1), _weighted_quantile(z_sorted, cw, 0.9))
    if deciles[1] <= deciles[0]:
        deciles = full_range
    m_start = (mu0 - centre) / spread
    m = m_start
    if not fix_mu:
        starts = ([m] if np.isfinite(m) else []) + [_weighted_quantile(z_sorted, cw, 0.5)]
        m = min(starts, key=lambda s: _lse(lw + (2.0 / lam) * _log_abs(z - s)))

    ll_prev, log_sigma = -np.inf, log_scale_bounds[0]
    for sweep in range(_MAX_OUTER_ITER):
        p = 2.0 / lam
        if not fix_mu:
            m = _location(z, z_sorted, w, lw, m, p, full_range if p >= 1.0 else deciles)
        log_r = _log_abs(z - m)
        if fit_shape:
            lam = _shape(log_r, lw, W, lam, (min_lambda, max_lambda), log_scale_bounds, warm=sweep > 0)
        p = 2.0 / lam
        log_s = _lse(lw + p * log_r)
        log_sigma = _log_scale(log_s, log_W, p, *log_scale_bounds)
        ll = _log_lik(log_s, W, p, log_sigma)
        if not np.isfinite(ll) or ll - ll_prev <= _OUTER_TOL * abs(ll):
            break
        ll_prev = ll
    start = (mu0, max(sigma0, min_sigma), float(np.clip(lam0, min_lambda, max_lambda)))
    new = (start[0] if fix_mu or m == m_start else _to_original(m, z_sorted, x[order], centre, spread),
           start[1] if fix_sigma else max(float(np.exp(log_sigma + log_spread)), min_sigma),
           start[2] if fix_lambda else lam)
    return start if _log_lik_at(x, lw, W, *new) < _log_lik_at(x, lw, W, *start) else new


def _to_original(m: float, z_sorted: np.ndarray, x_sorted: np.ndarray, centre: float, spread: float) -> float:
    """Map a standardised location back to original units: the exact data value when it is a data point, so the
    round trip cannot move it off the point (which costs likelihood at the density's cusp when p <= 1)."""
    k = int(np.searchsorted(z_sorted, m))
    if k < len(z_sorted) and z_sorted[k] == m:
        return float(x_sorted[k])
    return centre + spread * m


def _log_lik_at(x: np.ndarray, lw: np.ndarray, W: float, mu: float, sigma: float, lam: float) -> float:
    """Weighted log-likelihood in original units; ``lw`` are log weights whose exponentials sum to ``W``."""
    p = 2.0 / lam
    return _log_lik(_lse(lw + p * _log_abs(x - mu)), W, p, float(np.log(sigma)))


class Lambda(Distribution):
    r"""Symmetric exponential-power distribution, called "Lambda" in this package (``DistType.LAMDA``).

    Density, per dimension:

    .. math::

        f(x;\ \mu, \sigma, \lambda) = \frac{1}{\sigma\,\lambda\,\Gamma(\lambda/2)}
            \exp\!\left(-\left|\frac{x-\mu}{\sigma}\right|^{2/\lambda}\right)

    This is the generalised normal distribution with exponent :math:`p = 2/\lambda`. ``sigma`` is a scale, not the
    standard deviation:

    - variance :math:`= \sigma^2\,\Gamma(3\lambda/2)\,/\,\Gamma(\lambda/2)`;
    - excess kurtosis :math:`= \Gamma(5\lambda/2)\,\Gamma(\lambda/2)\,/\,\Gamma(3\lambda/2)^2 - 3`.

    =========  =====  ===============  ============================================================
    lambda     p      excess kurtosis  shape
    =========  =====  ===============  ============================================================
    0.2        10     -1.12            near-uniform (the default lower bound)
    0.5        4      -0.81            flat-topped, light tails (bounded oscillators: RSI, TSI, ...)
    1          2       0               normal (variance sigma^2 / 2)
    1.5        1.33    1.22            moderately heavy tails
    2          1       3               Laplace
    3          0.67    9.3             heavy tails
    4          0.5     22              very heavy tails
    =========  =====  ===============  ============================================================

    With more than one dimension the dimensions are independent: the log-density is the sum of the per-dimension
    log-densities (a "diagonal" model). There is no cross-dimension correlation.

    **Estimation.** ``summarize`` buffers the data and its weights, since the family has no finite sufficient
    statistics. ``from_summaries`` then computes the posterior-weighted MLE of every parameter of every dimension by
    coordinate ascent:

    - location by a bounded scalar search: the global minimum when p > 1 (convex). When p <= 1 every local minimum is a
      data point, and the update returns one exactly; when p < 1 it is a local minimum near the bulk of the data;
    - shape by maximising the profile likelihood over ``[min_lambda, max_lambda]``, with the scale profiled out
      inside its floor ``min_sigma``;
    - scale in closed form, clipped to ``min_sigma``.

    The constraints (scale floor, shape bounds) are part of the maximisation, not applied afterwards. The shape is held
    when a state's effective sample size :math:`(\sum w)^2 / \sum w^2` is below 10; location and scale still update.

    **An improving update, not an exact maximiser.** The location search when p < 1 is local, a held shape does not
    move, and the sweeps stop at a tolerance, so ``from_summaries`` is not guaranteed to reach the global constrained
    MLE. What it does guarantee, with ``inertia=0``, is that it never lowers the weighted log-likelihood of the current
    parameters: every coordinate step improves or keeps it, and the result is checked in the original units, with the
    current parameters kept if the likelihood would fall. Inside an HMM or mixture this makes each M-step a
    generalised-EM (GEM) step, which is all the outer EM needs to keep its likelihood non-decreasing.

    Every sum of powers is computed in log space, so very unequal weights or extreme data scales cannot overflow.
    ``inertia`` blends old and new values as in every pomegranate distribution; the blend itself carries no
    monotonicity guarantee. A parameter tensor flagged ``.frozen`` (e.g. ``dist.mu.frozen = True``) keeps its value,
    and the others are fitted conditional on it. Sample weights may be a masked tensor; a masked-out weight counts as
    zero.

    Parameters
    ----------
    mu: float, list, numpy.ndarray or torch.Tensor, optional
        Location per dimension (a scalar broadcasts). Default 0.0.
    sigma: float, list, numpy.ndarray or torch.Tensor, optional
        Scale per dimension (>= ``min_sigma``; a scalar broadcasts). Default 1.0, or ``min_sigma`` when that is higher.
        An explicit value below ``min_sigma`` raises.
    lambda_: float, list, numpy.ndarray or torch.Tensor, optional
        Shape per dimension, in ``[min_lambda, max_lambda]``. Default 1.0 (normal), or the bound nearest to 1.0 when
        the bounds exclude it. An explicit value outside the bounds raises.
    inertia: float, [0, 1], optional
        Weight of the old parameters in each update. Default 0.0.
    frozen: bool, optional
        If True, the parameters are never updated. Default False.
    check_data: bool, optional
        Whether to validate inputs. Default True.
    min_sigma: float, optional
        Floor on the scale. Default 1e-6.
    max_lambda: float, optional
        Upper bound on the shape (heaviest tails allowed). Default 10.0.
    dtype: torch.dtype, optional
        Parameter dtype. Default torch.float64.
    min_lambda: float, optional
        Lower bound on the shape (lightest tails allowed). Default 0.2.

    Notes
    -----
    Fixed 2026-11-23. Before that, ``from_summaries`` ignored the posterior weights for the scale and the shape: every
    state got the pooled standard deviation as its scale, and a tail heuristic that was inverted and floored at 1
    (normal) as its shape. Models fitted before the fix are tied-scale models and should be refitted.

    Model-level use: ``PomegranateHMM(DistType.LAMDA, min_lambda=..., max_lambda=..., min_sigma=..., inertia=...)``
    (and the mixture HMM) forward these keyword arguments to every state's ``Lambda``.
    """

    min_lambda = 0.2    # class default, so pickles from before the fix (no instance attribute) can keep fitting

    def __init__(self, mu=None, sigma=None, lambda_=None, inertia=0.0, frozen=False, check_data=True, min_sigma=1e-6,
                 max_lambda=10.0, dtype=torch.float64, min_lambda=0.2):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data, dtype=dtype)
        if not 0.0 < min_lambda < max_lambda:
            raise ValueError(f"Need 0 < min_lambda < max_lambda, got min_lambda={min_lambda}, max_lambda={max_lambda}")
        if min_sigma <= 0.0:
            raise ValueError(f"min_sigma must be > 0, got {min_sigma}")
        self.name = "Lambda"
        self._dtype_init = dtype
        self.min_sigma = float(min_sigma)
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self._mu_init = mu
        self._sigma_init = sigma
        self._lambda_init = lambda_
        self._initialized = False
        self.d = None

    # ---- set-up --------------------------------------------------------------------------------------------------
    def _default_lambda(self) -> float:
        """Starting shape when none is given: 1.0 (normal), or the bound nearest to it when the bounds exclude 1."""
        return float(np.clip(1.0, self.min_lambda, self.max_lambda))

    def _default_sigma(self) -> float:
        """Starting scale when none is given: 1.0, or the floor ``min_sigma`` when that is higher."""
        return max(1.0, self.min_sigma)

    @staticmethod
    def _broadcast(value, d: int, default: float, name: str) -> np.ndarray:
        if value is None:
            return np.full(d, default, dtype=np.float64)
        arr = np.atleast_1d(np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float64))
        if arr.size == 1:
            return np.full(d, arr.item())
        if arr.size != d:
            raise ValueError(f"Initial {name} has {arr.size} elements but the data has {d} features")
        return arr

    def _initialize(self, d):
        """Create the parameters for ``d`` dimensions from the constructor values (scalars broadcast)."""
        mu = self._broadcast(self._mu_init, d, 0.0, "mu")
        sigma = self._broadcast(self._sigma_init, d, self._default_sigma(), "sigma")
        lam = self._broadcast(self._lambda_init, d, self._default_lambda(), "lambda_")
        self.mu = _check_parameter(_cast_as_parameter(torch.tensor(mu, dtype=self._dtype_init)), "mu", ndim=1)
        self.sigma = _check_parameter(_cast_as_parameter(torch.tensor(sigma, dtype=self._dtype_init)), "sigma", ndim=1,
                                      min_value=self.min_sigma)
        self.lambda_ = _check_parameter(_cast_as_parameter(torch.tensor(lam, dtype=self._dtype_init)), "lambda_",
                                        ndim=1, min_value=self.min_lambda, max_value=self.max_lambda)
        self._initialized = True
        self.d = d
        self._reset_cache()

    def _reset_cache(self):
        """Clamp the parameters to their bounds, precompute the density constants and empty the summary buffers."""
        if not self._initialized:
            return
        with torch.no_grad():
            self.sigma.data = torch.clamp(self.sigma.data, min=self.min_sigma)
            self.lambda_.data = torch.clamp(self.lambda_.data, min=self.min_lambda, max=self.max_lambda)
        self.register_buffer("_log_norm", -torch.log(self.sigma) - torch.log(self.lambda_) - gammaln(self.lambda_ / 2))
        self.register_buffer("_exponent", 2 / self.lambda_)
        self.register_buffer("_inv_exponent", self.lambda_ / 2)
        self._X_buffer = [[] for _ in range(self.d)]
        self._weights_buffer = [[] for _ in range(self.d)]

    def _ensure_initialized(self, d: int) -> None:
        if not self._initialized:
            self._initialize(d)

    def _init_from_given(self, d: int | None = None) -> None:
        """Initialise from constructor parameters for use before any fit. ``d`` defaults to the longest given
        parameter."""
        if self._initialized:
            return
        given = [v for v in (self._mu_init, self._sigma_init, self._lambda_init) if v is not None]
        if not given:
            raise ValueError("Lambda has no parameters yet: pass mu / sigma / lambda_ or fit it first.")
        if d is None:
            d = max(int(np.atleast_1d(np.asarray(v.detach().cpu() if torch.is_tensor(v) else v)).size) for v in given)
        self._initialize(d)

    # ---- density -------------------------------------------------------------------------------------------------
    def log_probability(self, X):
        """Log-density of each row (summed over dimensions)."""
        X = _cast_as_tensor(X)
        self._init_from_given(X.shape[1] if X.ndim == 2 else 1)
        X = _check_parameter(_cast_as_tensor(X, dtype=self.mu.dtype), "X", ndim=2, shape=(-1, self.d),
                             check_parameter=self.check_data)
        z = torch.abs((X - self.mu) / self.sigma)
        return torch.sum(self._log_norm - z ** self._exponent, dim=1)

    # ---- estimation ----------------------------------------------------------------------------------------------
    def summarize(self, X, sample_weight=None):
        """Buffer the data and its weights (per dimension) for ``from_summaries``. Rows that are masked out, non-finite,
        or carry a non-finite or non-positive weight are dropped. A frozen distribution is set up from its given
        parameters but buffers nothing."""
        X = _cast_as_tensor(X)
        self._ensure_initialized(X.shape[1] if X.ndim == 2 else 1)
        if self.frozen:
            return
        if isinstance(sample_weight, torch.masked.MaskedTensor):        # a masked-out weight counts as zero weight
            sample_weight = torch.where(sample_weight.get_mask(), sample_weight.get_data(), 0.0)
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        if isinstance(X, torch.masked.MaskedTensor):
            mask = X.get_mask().cpu().numpy()
            data = X.get_data().detach().cpu().numpy().astype(np.float64)
        else:
            data = X.detach().cpu().numpy().astype(np.float64)
            mask = np.ones_like(data, dtype=bool)
        sw = sample_weight.get_data() if isinstance(sample_weight, torch.masked.MaskedTensor) else sample_weight
        weights = sw.detach().cpu().numpy().astype(np.float64)
        if getattr(self, "_X_buffer", None) is None or len(self._X_buffer) != self.d:    # e.g. a pre-fix pickle
            self._X_buffer = [[] for _ in range(self.d)]
            self._weights_buffer = [[] for _ in range(self.d)]
        for dim in range(self.d):
            keep = mask[:, dim] & np.isfinite(data[:, dim]) & np.isfinite(weights[:, dim]) & (weights[:, dim] > 0)
            self._X_buffer[dim].append(data[keep, dim])
            self._weights_buffer[dim].append(weights[keep, dim])

    def from_summaries(self):
        """Posterior-weighted MLE of (mu, sigma, lambda_) per dimension from the buffered data; then empty the
        buffers. A dimension with no positive weight keeps its parameters. A parameter tensor flagged ``.frozen``
        (pomegranate's per-parameter freeze) keeps its value, and the others are fitted conditional on it."""
        if self.frozen or not self._initialized:
            return
        mu = self.mu.detach().cpu().numpy().copy()
        sigma = self.sigma.detach().cpu().numpy().copy()
        lam = self.lambda_.detach().cpu().numpy().copy()
        fix = {name: bool(getattr(getattr(self, name), "frozen", False)) for name in ("mu", "sigma", "lambda_")}
        for dim in range(self.d):
            if not self._X_buffer[dim]:
                continue
            x = np.concatenate(self._X_buffer[dim])
            w = np.concatenate(self._weights_buffer[dim])
            if x.size == 0 or w.sum() <= 0.0:
                continue
            mu[dim], sigma[dim], lam[dim] = _weighted_mle(x, w, float(mu[dim]), float(sigma[dim]), float(lam[dim]),
                                                          self.min_lambda, self.max_lambda, self.min_sigma,
                                                          fix_mu=fix["mu"], fix_sigma=fix["sigma"],
                                                          fix_lambda=fix["lambda_"])
        dt, dev = self.mu.dtype, self.mu.device
        _update_parameter(self.mu, torch.tensor(mu, dtype=dt, device=dev), self.inertia)
        _update_parameter(self.sigma, torch.tensor(sigma, dtype=dt, device=dev), self.inertia)
        _update_parameter(self.lambda_, torch.tensor(lam, dtype=dt, device=dev), self.inertia)
        self._reset_cache()

    # ---- sampling ------------------------------------------------------------------------------------------------
    def sample(self, n):
        """Draw ``n`` rows, shape (n, d). If |Z|^p ~ Gamma(1/p, 1) with p = 2 / lambda, then mu + sigma * sign * |Z|
        has this density."""
        n = int(n)
        self._init_from_given()
        conc = self._inv_exponent.detach()
        g = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample((n,))
        sign = torch.where(torch.rand(n, self.d, dtype=g.dtype, device=g.device) < 0.5, -1.0, 1.0).to(g.dtype)
        return (self.mu.detach() + self.sigma.detach() * sign * g ** conc).to(self.mu.dtype)

    # ---- serialisation -------------------------------------------------------------------------------------------
    def to_json(self):
        """JSON-serialisable description. Parameters are lists, one entry per dimension."""
        return {"class": "Lambda",
                "parameters": {"mu": self.mu.detach().cpu().tolist(), "sigma": self.sigma.detach().cpu().tolist(),
                               "lambda_": self.lambda_.detach().cpu().tolist()},
                "inertia": float(self.inertia), "frozen": bool(self.frozen), "min_sigma": self.min_sigma,
                "min_lambda": self.min_lambda, "max_lambda": self.max_lambda, "dtype": str(self._dtype_init)}

    @classmethod
    def from_json(cls, d):
        """Rebuild from ``to_json`` output. Also reads the older scalar-parameter format."""
        dtype = torch.float32 if "float32" in d.get("dtype", "torch.float64") else torch.float64
        p = d["parameters"]
        lam = p["lambda_"]
        dist = cls(mu=p["mu"], sigma=p["sigma"], lambda_=lam, inertia=d.get("inertia", 0.0),
                   frozen=d.get("frozen", False), min_sigma=d.get("min_sigma", 1e-6),
                   max_lambda=d.get("max_lambda", 10.0), dtype=dtype,
                   min_lambda=d.get("min_lambda", min(0.2, float(np.min(lam)))))
        dist._initialize(int(np.atleast_1d(p["mu"]).size))
        return dist
