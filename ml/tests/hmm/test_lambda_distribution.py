"""The symmetric exponential-power ("Lambda", DistType.LAMDA) emission distribution.

Verifies:
- The density: integrates to 1 over the shape range; equals the normal density at lambda = 1 (sigma = sqrt(2) sd) and
  the Laplace density at lambda = 2; sums over independent dimensions.
- Weighted estimation (the 2026-11-23 defect): the posterior weights drive the scale and the shape. Two groups with sds
  0.3 / 0.6 are recovered separately (the old M-step gave both the pooled 0.807). Integer weights equal repeated rows.
- The shape: recovered for normal, Laplace, light-tailed (lambda < 1) and heavy-tailed data; flat-topped data goes to
  the lower bound; it is held when the effective sample size is tiny.
- Invariance: an affine change of the data moves location and scale, not the shape.
- Sampling: the sample variance matches sigma^2 Gamma(3 lambda / 2) / Gamma(lambda / 2) at several shapes, and a refit
  recovers the parameters.
- inertia, frozen, degenerate input (constant data, zero weights), bounds, multivariate input, JSON round trips
  (including the pre-fix scalar format).
- Constraints and numerical range (review of 2026-11-24): the scale floor is part of the maximisation, so a fit never
  lowers the weighted likelihood of an admissible start (randomised over data shape, data scale, weights, floor and
  bounds), nor does a refit from the fit; very unequal weights and extreme data scales stay finite; the default shape
  and scale lie inside bounds that exclude 1; a parameter frozen with ``.frozen`` stays put and the others reach the
  conditional MLE (checked by brute force); masked-out sample weights count as zero; for p = 2 / lambda <= 1 the
  location lands exactly on a data point, and a refit started on it does not lose likelihood.
- Models: PomegranateHMM(DistType.LAMDA) and a pomegranate mixture recover planted states with distinct scales and
  shapes, and model-level dist_kwargs reach every state's Lambda.
"""

import numpy as np
import pytest
import torch
from pomegranate.gmm import GeneralMixtureModel
from scipy import integrate, stats
from scipy.optimize import minimize
from scipy.special import gammaln

from okmich_quant_ml.hmm import DistType, PomegranateHMM
from okmich_quant_ml.hmm.distribution import Lambda


def _sd(dist: Lambda) -> np.ndarray:
    lam = dist.lambda_.detach().numpy()
    return dist.sigma.detach().numpy() * np.sqrt(np.exp(gammaln(1.5 * lam) - gammaln(0.5 * lam)))


def _col(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


# ---- density -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.2, 0.5, 1.0, 2.0, 4.0])
def test_density_integrates_to_one(lam):
    dist = Lambda(mu=0.3, sigma=1.7, lambda_=lam)
    f = lambda x: float(torch.exp(dist.log_probability(torch.tensor([[x]], dtype=torch.float64)))[0])  # noqa: E731
    total = integrate.quad(f, -np.inf, 0.3)[0] + integrate.quad(f, 0.3, np.inf)[0]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_lambda_one_is_normal_and_two_is_laplace():
    x = np.linspace(-4, 4, 41)
    normal = Lambda(mu=0.5, sigma=np.sqrt(2.0) * 1.3, lambda_=1.0).log_probability(_col(x)).numpy()
    laplace = Lambda(mu=0.5, sigma=1.3, lambda_=2.0).log_probability(_col(x)).numpy()
    np.testing.assert_allclose(normal, stats.norm.logpdf(x, 0.5, 1.3), atol=1e-10)
    np.testing.assert_allclose(laplace, stats.laplace.logpdf(x, 0.5, 1.3), atol=1e-10)


def test_multivariate_density_is_sum_of_independent_dimensions():
    X = np.random.default_rng(0).normal(size=(50, 2))
    joint = Lambda(mu=[0.0, 1.0], sigma=[1.0, 2.0], lambda_=[0.5, 2.0]).log_probability(X).numpy()
    a = Lambda(mu=0.0, sigma=1.0, lambda_=0.5).log_probability(X[:, :1]).numpy()
    b = Lambda(mu=1.0, sigma=2.0, lambda_=2.0).log_probability(X[:, 1:]).numpy()
    np.testing.assert_allclose(joint, a + b, atol=1e-12)


def test_unfitted_without_parameters_raises():
    with pytest.raises(ValueError, match="no parameters"):
        Lambda().log_probability(_col([0.0, 1.0]))


# ---- weighted estimation -------------------------------------------------------------------------------------------
def test_weights_drive_the_scale_per_state():
    """The defect: unweighted np.std gave both groups the pooled spread (0.807)."""
    rng = np.random.default_rng(1)
    x = _col(np.r_[rng.normal(0, 0.3, 5000), rng.normal(0, 0.6, 5000)])
    first = np.r_[np.ones(5000), np.zeros(5000)]
    a = Lambda().fit(x, sample_weight=first)
    b = Lambda().fit(x, sample_weight=1.0 - first)
    assert _sd(a)[0] == pytest.approx(0.3, rel=0.03)
    assert _sd(b)[0] == pytest.approx(0.6, rel=0.03)
    assert a.lambda_.item() == pytest.approx(1.0, abs=0.1)
    assert b.lambda_.item() == pytest.approx(1.0, abs=0.1)


def test_weights_drive_the_location():
    rng = np.random.default_rng(2)
    x = _col(np.r_[rng.laplace(-2, 1, 4000), rng.laplace(3, 1, 4000)])
    w = np.r_[np.full(4000, 0.9), np.full(4000, 0.1)]
    hard = Lambda().fit(x[:4000])
    soft = Lambda().fit(x, sample_weight=np.r_[np.ones(4000), np.zeros(4000)])
    assert soft.mu.item() == pytest.approx(hard.mu.item(), abs=1e-9)
    assert Lambda().fit(x, sample_weight=w).mu.item() < 0.0


def test_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(3)
    x = rng.standard_t(5, 600)
    reps = rng.integers(1, 4, 600)
    weighted = Lambda().fit(_col(x), sample_weight=reps.astype(float))
    repeated = Lambda().fit(_col(np.repeat(x, reps)))
    for p in ("mu", "sigma", "lambda_"):
        assert getattr(weighted, p).item() == pytest.approx(getattr(repeated, p).item(), rel=1e-6, abs=1e-9)


def test_batched_summaries_equal_one_summary():
    x = _col(np.random.default_rng(4).laplace(0, 1, 3000))
    one = Lambda().fit(x)
    batched = Lambda()
    for chunk in np.array_split(x, 5):
        batched.summarize(chunk)
    batched.from_summaries()
    for p in ("mu", "sigma", "lambda_"):
        assert getattr(batched, p).item() == pytest.approx(getattr(one, p).item(), rel=1e-9, abs=1e-12)


# ---- shape -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name, draw, lo, hi", [
    ("normal", lambda r, n: r.normal(0, 1, n), 0.93, 1.07),
    ("laplace", lambda r, n: r.laplace(0, 1, n), 1.88, 2.12),
    ("student_t3", lambda r, n: r.standard_t(3, n), 1.8, 3.0),
    ("uniform", lambda r, n: r.uniform(-1, 1, n), 0.2, 0.21),
])
def test_shape_recovery(name, draw, lo, hi):
    dist = Lambda().fit(_col(draw(np.random.default_rng(5), 20_000)))
    assert lo <= dist.lambda_.item() <= hi, name


@pytest.mark.parametrize("lam", [0.4, 0.7, 1.5, 3.0])
def test_sample_then_refit_recovers_parameters(lam):
    torch.manual_seed(0)
    s = Lambda(mu=1.0, sigma=2.0, lambda_=lam).sample(100_000).numpy()
    var = 4.0 * np.exp(gammaln(1.5 * lam) - gammaln(0.5 * lam))
    assert s.shape == (100_000, 1)
    assert s.var() == pytest.approx(var, rel=0.05)
    fit = Lambda().fit(s)
    assert fit.lambda_.item() == pytest.approx(lam, rel=0.05)
    assert fit.sigma.item() == pytest.approx(2.0, rel=0.03)
    assert fit.mu.item() == pytest.approx(1.0, abs=0.03)


def test_affine_invariance():
    x = np.random.default_rng(6).standard_t(4, 5000)
    a = Lambda().fit(_col(x))
    b = Lambda().fit(_col(1000.0 * x + 5.0))
    assert b.lambda_.item() == pytest.approx(a.lambda_.item(), rel=1e-5)
    assert b.sigma.item() == pytest.approx(1000.0 * a.sigma.item(), rel=1e-5)
    assert b.mu.item() == pytest.approx(1000.0 * a.mu.item() + 5.0, rel=1e-5, abs=1e-3)


def test_shape_held_when_effective_sample_is_tiny():
    dist = Lambda(lambda_=1.7)
    dist.fit(_col([0.1, -0.4, 0.3, 1.2, -0.8]))
    assert dist.lambda_.item() == pytest.approx(1.7)
    assert dist.sigma.item() > 0.0


def test_shape_respects_bounds():
    x = _col(np.random.default_rng(7).standard_cauchy(5000))
    assert Lambda(max_lambda=3.0).fit(x).lambda_.item() == pytest.approx(3.0, abs=1e-4)
    u = _col(np.random.default_rng(8).uniform(-1, 1, 5000))
    assert Lambda(min_lambda=0.6).fit(u).lambda_.item() == pytest.approx(0.6, abs=1e-4)
    with pytest.raises(ValueError):
        Lambda(min_lambda=2.0, max_lambda=1.0)


def test_multivariate_dimensions_fit_independently():
    torch.manual_seed(1)
    s = Lambda(mu=[0.0, 5.0], sigma=[1.0, 0.5], lambda_=[0.5, 2.0]).sample(60_000).numpy()
    fit = Lambda().fit(s)
    np.testing.assert_allclose(fit.mu.numpy(), [0.0, 5.0], atol=0.02)
    np.testing.assert_allclose(fit.sigma.numpy(), [1.0, 0.5], rtol=0.03)
    np.testing.assert_allclose(fit.lambda_.numpy(), [0.5, 2.0], rtol=0.05)


# ---- update semantics --------------------------------------------------------------------------------------------
def test_inertia_blends_old_and_new():
    x = _col(np.random.default_rng(9).normal(4.0, 1.0, 5000))
    new = Lambda().fit(x)
    dist = Lambda(mu=0.0, sigma=1.0, lambda_=1.0, inertia=0.25)
    dist.fit(x)
    assert dist.mu.item() == pytest.approx(0.75 * new.mu.item(), rel=1e-9)
    assert dist.sigma.item() == pytest.approx(0.25 + 0.75 * new.sigma.item(), rel=1e-9)


def test_frozen_never_updates():
    dist = Lambda(mu=0.0, sigma=1.0, lambda_=1.0, frozen=True)
    dist.fit(_col(np.random.default_rng(10).normal(3.0, 2.0, 1000)))
    assert (dist.mu.item(), dist.sigma.item(), dist.lambda_.item()) == (0.0, 1.0, 1.0)


def test_constant_data_floors_the_scale():
    dist = Lambda(min_sigma=1e-4).fit(_col(np.full(200, 2.5)))
    assert dist.mu.item() == pytest.approx(2.5)
    assert dist.sigma.item() == pytest.approx(1e-4)
    assert np.isfinite(dist.log_probability(_col([2.5])).numpy()).all()


def test_zero_weights_keep_the_parameters():
    dist = Lambda(mu=1.0, sigma=2.0, lambda_=1.5)
    dist.fit(_col([0.0, 1.0, 2.0]), sample_weight=np.zeros(3))
    assert (dist.mu.item(), dist.sigma.item(), dist.lambda_.item()) == (1.0, 2.0, 1.5)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("weight_shape", [(3,), (3, 1)])
def test_masked_out_weight_counts_as_zero(weight_shape):
    mask = torch.tensor([True, True, False]).reshape(weight_shape)
    sw = torch.masked.MaskedTensor(torch.ones(weight_shape, dtype=torch.float64), mask)
    dist = Lambda()
    dist.summarize(_col([0.0, 0.0, 100.0]), sample_weight=sw)
    dist.from_summaries()
    assert dist.mu.item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_data_mask_and_weight_mask_combine():
    X = torch.masked.MaskedTensor(torch.tensor([[1.0], [2.0], [100.0]], dtype=torch.float64),
                                  torch.tensor([[False], [True], [True]]))
    sw = torch.masked.MaskedTensor(torch.ones(3, 1, dtype=torch.float64), torch.tensor([[True], [True], [False]]))
    dist = Lambda()
    dist.summarize(X, sample_weight=sw)
    dist.from_summaries()
    assert dist.mu.item() == pytest.approx(2.0)


# ---- constraints and numerical range -----------------------------------------------------------------------------
def _ll_grid(x: np.ndarray, w: np.ndarray, mu, sigma, lam) -> np.ndarray:
    """Weighted log-likelihood, broadcast over arrays of (mu, sigma, lambda)."""
    mu, sigma, lam = (np.asarray(v, dtype=np.float64)[..., None] for v in (mu, sigma, lam))
    logf = -np.log(sigma) - np.log(lam) - gammaln(lam / 2) - np.abs((x - mu) / sigma) ** (2 / lam)
    return logf @ w


def _brute_force_ll(x: np.ndarray, w: np.ndarray, mu=None, sigma=None, lam=None, min_sigma=1e-6,
                    bounds=(0.2, 10.0)) -> float:
    """Best weighted log-likelihood found by multi-start L-BFGS-B over the free parameters (sigma >= min_sigma, lambda
    within bounds), with any parameter given as a number held fixed."""
    free = [name for name, v in (("mu", mu), ("sigma", sigma), ("lam", lam)) if v is None]
    box = {"mu": (x.min(), x.max()), "sigma": (np.log(min_sigma), np.log(10 * np.ptp(x) + min_sigma)),
           "lam": (np.log(bounds[0]), np.log(bounds[1]))}

    def unpack(t):
        v = dict(zip(free, t))
        return (v.get("mu", mu), np.exp(v["sigma"]) if "sigma" in v else sigma, np.exp(v["lam"]) if "lam" in v else lam)

    best = -np.inf
    for start in np.random.default_rng(0).uniform(size=(12, len(free))):
        t0 = [lo + s * (hi - lo) for s, (lo, hi) in zip(start, (box[f] for f in free))]
        res = minimize(lambda t: -float(_ll_grid(x, w, *unpack(t))), t0, method="L-BFGS-B",
                       bounds=[box[f] for f in free])
        best = max(best, -res.fun)
    return best


def test_concentrated_weights_stay_finite():
    # p = 10 (lambda 0.2) with weight 1e-100 one unit away: in standardised units |r|^p ~ 1e400 overflowed before the
    # weight was applied, giving sigma = inf and -inf log-densities.
    dist = Lambda(lambda_=0.2)
    dist.fit(_col([0.0, 1.0]), sample_weight=np.array([1.0, 1e-100]))
    # exact MLE c^9 = 1e-100 (1 - c)^9, c = 7.74e-12; float64 resolves the flat objective only to ~1e-13
    assert dist.mu.item() == pytest.approx(10 ** (-100 / 9), abs=1e-12)
    assert dist.sigma.item() == pytest.approx(1e-6)                             # exact MLE 10^-9.9 is below the floor
    assert np.isfinite(dist.log_probability(_col([0.0, 1.0])).numpy()).all()


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_extreme_data_scales(scale):
    x = np.random.default_rng(11).laplace(0.0, 1.0, 3000)
    ref = Lambda().fit(_col(x))
    dist = Lambda(min_sigma=1e-300).fit(_col(scale * x))
    assert dist.lambda_.item() == pytest.approx(ref.lambda_.item(), rel=1e-5)
    assert dist.sigma.item() == pytest.approx(scale * ref.sigma.item(), rel=1e-5)


def test_scale_floor_is_part_of_the_fit():
    # 200 points at +-1e-7 with the floor at 1e-6: the unconstrained scale (1.26e-7) is below the floor. Profiling an
    # unconstrained scale and clipping afterwards picked lambda 0.2 and lowered the likelihood (2646.629 -> 2634.447).
    x = np.where(np.arange(200) % 2, 1e-7, -1e-7)
    w = np.ones_like(x)
    dist = Lambda(mu=0.0, sigma=1e-6, lambda_=1.0)
    before = float(_ll_grid(x, w, 0.0, 1e-6, 1.0))
    dist.fit(_col(x))
    after = float(_ll_grid(x, w, dist.mu.item(), dist.sigma.item(), dist.lambda_.item()))
    assert before == pytest.approx(2646.629, abs=1e-3)
    assert dist.sigma.item() == pytest.approx(1e-6)
    lam = np.exp(np.linspace(np.log(0.2), np.log(10.0), 4001))
    assert after >= _ll_grid(x, w, dist.mu.item(), 1e-6, lam).max() - 1e-6     # the constrained optimum over lambda
    assert after > before


@pytest.mark.parametrize("seed", range(8))
def test_m_step_never_lowers_the_likelihood(seed):
    rng = np.random.default_rng(200 + seed)
    n, scale = 400, 10.0 ** rng.uniform(-8, 3)
    draw = [rng.normal(size=n), rng.standard_t(1.5, n), rng.uniform(-1, 1, n), np.round(rng.normal(size=n), 1),
            np.where(rng.random(n) < 0.5, -1.0, 1.0)][seed % 5]
    x = scale * draw + rng.normal()
    w = np.exp(rng.normal(0.0, 3.0, n))                                         # very unequal posterior weights
    lo, hi = sorted(np.exp(rng.uniform(np.log(0.2), np.log(10.0), 2)))
    floor = scale * 10.0 ** rng.uniform(-3, 1)
    lam0 = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    mu0, sigma0 = float(rng.choice(x)), floor * 10.0 ** rng.uniform(0, 3)
    dist = Lambda(mu=mu0, sigma=sigma0, lambda_=lam0, min_lambda=lo, max_lambda=hi, min_sigma=floor)
    before = float(_ll_grid(x, w, mu0, sigma0, lam0))
    dist.fit(_col(x), sample_weight=w)
    after = float(_ll_grid(x, w, dist.mu.item(), dist.sigma.item(), dist.lambda_.item()))
    assert np.isfinite(after)
    assert after >= before
    dist.fit(_col(x), sample_weight=w)                                          # a refit from the fit: no loss either
    assert float(_ll_grid(x, w, dist.mu.item(), dist.sigma.item(), dist.lambda_.item())) >= after


@pytest.mark.parametrize("lam", [2.0, 4.0, 10.0])
def test_heavy_tailed_location_lands_exactly_on_a_data_point(lam):
    # For p = 2 / lambda <= 1 the location objective is concave (p < 1) or linear (p = 1) between consecutive data
    # points, so its minima are data points. A continuous search lands ~1e-14 off the point, and the cusp of the
    # density turns that into a visible likelihood loss (review 2026-11-24).
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.0, 1.0, 20), rng.normal(1000.0, 1.0, 5)]
    w = np.ones_like(x)
    dist = Lambda(lambda_=lam, min_lambda=lam - 1e-9, max_lambda=lam)
    dist.fit(_col(x))
    mu = dist.mu.item()
    assert mu in set(x.tolist())
    first = float(_ll_grid(x, w, mu, dist.sigma.item(), dist.lambda_.item()))
    for start in x[np.argsort(np.abs(x - mu))[:3]]:                             # start on the point and its neighbours
        again = Lambda(mu=start, sigma=dist.sigma.item(), lambda_=lam, min_lambda=lam - 1e-9, max_lambda=lam)
        before = float(_ll_grid(x, w, start, dist.sigma.item(), lam))
        again.fit(_col(x))
        after = float(_ll_grid(x, w, again.mu.item(), again.sigma.item(), again.lambda_.item()))
        assert after >= before
        assert after >= first - 1e-9 * abs(first)


@pytest.mark.parametrize("bounds, start", [((2.0, 4.0), 2.0), ((0.2, 0.8), 0.8)])
def test_default_shape_lies_within_bounds_that_exclude_normal(bounds, start):
    x = _col(np.random.default_rng(12).laplace(0.0, 1.0, 3000))
    dist = Lambda(min_lambda=bounds[0], max_lambda=bounds[1])
    dist._initialize(1)
    assert dist.lambda_.item() == start
    fitted = Lambda(min_lambda=bounds[0], max_lambda=bounds[1]).fit(x)
    assert bounds[0] <= fitted.lambda_.item() <= bounds[1]
    with pytest.raises(ValueError):
        Lambda(lambda_=1.0, min_lambda=bounds[0], max_lambda=bounds[1]).fit(x)


def test_default_scale_respects_a_floor_above_one():
    x = _col(np.random.default_rng(14).normal(0.0, 10.0, 2000))
    dist = Lambda(min_sigma=2.0)
    dist._initialize(1)
    assert dist.sigma.item() == 2.0
    assert Lambda(min_sigma=2.0).fit(x).sigma.item() >= 2.0
    with pytest.raises(ValueError):
        Lambda(sigma=1.0, min_sigma=2.0).fit(x)
    model = PomegranateHMM(DistType.LAMDA, n_states=2, min_sigma=2.0, random_state=0, max_iter=5)
    model.fit(x)                                                                # the forwarded model-level floor
    assert all(p["sigma"][0] >= 2.0 for p in model.parameters)


@pytest.mark.parametrize("frozen, value", [("mu", 0.0), ("sigma", 3.0), ("lambda_", 2.0)])
def test_frozen_parameter_fits_the_others_conditionally(frozen, value):
    x = np.random.default_rng(13).normal(10.0, 0.5, 2000)
    w = np.ones_like(x)
    dist = Lambda(mu=0.0 if frozen == "mu" else 9.0, sigma=3.0 if frozen == "sigma" else 1.0,
                  lambda_=2.0 if frozen == "lambda_" else 1.0)
    dist._initialize(1)
    getattr(dist, frozen).frozen = True
    dist.fit(_col(x))
    assert getattr(dist, frozen).item() == value
    fit_ll = float(_ll_grid(x, w, dist.mu.item(), dist.sigma.item(), dist.lambda_.item()))
    fixed = {"mu": None, "sigma": None, "lam": None} | {{"lambda_": "lam"}.get(frozen, frozen): value}
    assert fit_ll >= _brute_force_ll(x, w, **fixed) - 1e-6 * abs(fit_ll)


# ---- serialisation -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 3])
def test_json_round_trip(d):
    torch.manual_seed(2)
    src = Lambda(mu=np.arange(d, dtype=float), sigma=np.ones(d), lambda_=np.linspace(0.5, 2.5, d))
    src.fit(src.sample(5000).numpy())
    back = Lambda.from_json(src.to_json())
    for p in ("mu", "sigma", "lambda_"):
        np.testing.assert_allclose(getattr(back, p).numpy(), getattr(src, p).numpy())
    X = np.zeros((4, d))
    np.testing.assert_allclose(back.log_probability(X).numpy(), src.log_probability(X).numpy())


def test_json_reads_the_pre_fix_scalar_format():
    old = {"class": "Lambda", "parameters": {"mu": 0.1, "sigma": 1.2, "lambda_": 1.5}, "inertia": 0.0,
           "frozen": False, "min_sigma": 1e-6, "max_lambda": 10.0, "dtype": "torch.float64"}
    dist = Lambda.from_json(old)
    assert (dist.d, dist.lambda_.item(), dist.min_lambda) == (1, 1.5, 0.2)


# ---- models ------------------------------------------------------------------------------------------------------
def _planted(seed: int = 3, T: int = 3000):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < 0.97 else 1 - s[t - 1]
    x = np.where(s == 1, rng.laplace(0.5, 1.0, T), rng.normal(-0.5, 0.3, T))
    return _col(x), s


def test_hmm_recovers_states_with_distinct_scale_and_shape():
    x, s = _planted()
    model = PomegranateHMM(DistType.LAMDA, n_states=2, random_state=1, max_iter=50)
    model.fit(x)
    params = sorted(model.parameters, key=lambda p: float(p["means"][0]))
    normal_state, laplace_state = params
    assert normal_state["sigma"][0] == pytest.approx(0.3 * np.sqrt(2.0), rel=0.15)
    assert normal_state["lambda"][0] == pytest.approx(1.0, abs=0.2)
    assert laplace_state["sigma"][0] == pytest.approx(1.0, rel=0.15)
    assert laplace_state["lambda"][0] == pytest.approx(2.0, abs=0.3)
    labels = model.predict(x)
    assert max(np.mean(labels == s), np.mean(labels != s)) > 0.93


def test_mixture_recovers_components():
    x, _ = _planted()
    gmm = GeneralMixtureModel([Lambda(), Lambda()], max_iter=100, random_state=0).fit(torch.tensor(x))
    comps = sorted(gmm.distributions, key=lambda d: d.mu.item())
    assert comps[0].sigma.item() < 0.6 < comps[1].sigma.item()
    assert comps[1].lambda_.item() > comps[0].lambda_.item()


def test_model_dist_kwargs_reach_every_state():
    model = PomegranateHMM(DistType.LAMDA, n_states=3, min_lambda=0.5, max_lambda=4.0, min_sigma=1e-3)
    for dist in model._build_model().distributions:
        assert (dist.min_lambda, dist.max_lambda, dist.min_sigma) == (0.5, 4.0, 1e-3)
