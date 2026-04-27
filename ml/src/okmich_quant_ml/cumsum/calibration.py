from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .detector import CusumDetector, _broadcast_slack
from .protocols import ReferenceModel
from .reference_models import EwmaReferenceModel, GaussianReferenceModel, Sided, SignCusumReferenceModel

# Search bounds and tolerances for the bisection threshold search.
_BISECT_MAX_ITER = 30
_BISECT_TOLERANCE = 0.05
_H_LO_FLOOR = 1e-3
_H_HI_CEIL = 50.0
_BRACKET_MAX_DOUBLINGS = 10
_MIN_CALIBRATION_WINDOW = 200
_SUPPORTED_FACTORIES: tuple[type, ...] = (GaussianReferenceModel, EwmaReferenceModel, SignCusumReferenceModel)


class CalibrationMethod(StrEnum):
    AUTO = "auto"
    CLOSED_FORM = "closed_form"
    MONTE_CARLO = "monte_carlo"


def _siegmund_arl0(h: float, k: float) -> float:
    """Siegmund (1985) corrected-Wald ARL_0 approximation for one-sided Page CUSUM.

    For increments ~ N(-k, 1) under H0 with reflection at 0 and absorbing barrier h:

        ARL_0(h, k) ≈ (exp(2k(h + 1.166)) - 1 - 2k(h + 1.166)) / (2 k^2)

    The denominator is 2 μ^2 with μ = k (the conjugate-measure drift, not 2k).
    Continuity correction 1.166 ≈ -2 ζ(1/2) / sqrt(2π).

    Empirically verified: at k=0.5, h=4 the formula gives 338.1, Monte Carlo
    over 50 000 runs gives 335.6 (within 0.7%).
    """
    if k <= 0.0:
        raise ValueError(f"slack k must be > 0 for Siegmund formula, got {k}")
    u = 2.0 * k * (h + 1.166)
    return (math.exp(u) - 1.0 - u) / (2.0 * k * k)


def _siegmund_threshold(target_arl: float, k: float) -> float:
    """Invert Siegmund's ARL_0 formula via monotone bisection on h.

    The formula is strictly increasing in h for h > 0, so a bracket-and-bisect
    is unconditionally safe.
    """
    h_lo, h_hi = _H_LO_FLOOR, 1.0
    while _siegmund_arl0(h_hi, k) < target_arl:
        h_hi *= 2.0
        if h_hi > _H_HI_CEIL:
            raise ValueError(
                f"target_arl={target_arl} unreachable for k={k} within Siegmund bracket [<= {_H_HI_CEIL}]"
            )
    for _ in range(80):  # plenty for 1e-9 precision on a monotone scalar invert
        h_mid = 0.5 * (h_lo + h_hi)
        if _siegmund_arl0(h_mid, k) < target_arl:
            h_lo = h_mid
        else:
            h_hi = h_mid
        if h_hi - h_lo < 1e-9:
            break
    return 0.5 * (h_lo + h_hi)


def _is_one_sided_gaussian(reference_model: ReferenceModel) -> bool:
    return (
        isinstance(reference_model, GaussianReferenceModel)
        and reference_model.sided is not Sided.TWO
    )


def _empirical_arl(reference_model: ReferenceModel, slack: NDArray, threshold: float, *,
                   n_simulations: int, simulation_horizon: int,
                   in_control_sampler: Callable[[int, np.random.Generator], NDArray],
                   rng: np.random.Generator) -> tuple[float, float]:
    """Run Monte Carlo and return (mean_run_length, censored_fraction)."""
    detector = CusumDetector(reference_model, slack=slack, reset_to_zero=True)
    run_lengths = np.empty(n_simulations, dtype=np.float64)
    censored_count = 0
    for i in range(n_simulations):
        detector.reference_model = copy.deepcopy(reference_model)
        detector.reset()
        xs = in_control_sampler(simulation_horizon, rng)
        # Walk per-bar so we can stop at first crossing.
        run_length = simulation_horizon
        for t in range(simulation_horizon):
            statistic = detector.update(float(xs[t]))
            if np.any(statistic > threshold):
                run_length = t + 1
                break
        else:
            censored_count += 1
        run_lengths[i] = run_length
    return float(run_lengths.mean()), censored_count / n_simulations


def _resolve_sampler(reference_model: ReferenceModel,
                     in_control_sampler: Callable[[int, np.random.Generator], NDArray] | None
                     ) -> Callable[[int, np.random.Generator], NDArray]:
    if in_control_sampler is not None:
        return in_control_sampler
    if reference_model.requires_external_sampler:
        raise ValueError(
            f"{type(reference_model).__name__} requires `in_control_sampler` for Monte Carlo calibration."
        )
    return reference_model.sample_in_control


def target_arl_threshold(reference_model: ReferenceModel, slack: float | NDArray,
                         target_arl: float, *, method: CalibrationMethod = CalibrationMethod.AUTO,
                         n_simulations: int = 10_000, simulation_horizon: int | None = None,
                         in_control_sampler: Callable[[int, np.random.Generator], NDArray] | None = None,
                         seed: int | None = None) -> float:
    """Find threshold h producing the target ARL under H0. See spec §6."""
    target_arl = float(target_arl)
    if not target_arl > 0.0:
        raise ValueError(f"target_arl must be > 0, got {target_arl}")
    if n_simulations < 100:
        raise ValueError(f"n_simulations must be >= 100, got {n_simulations}")
    method = CalibrationMethod(method)

    slack_arr = _broadcast_slack(slack, int(reference_model.n_directions))

    # Closed-form fast path: only one-sided Gaussian.
    if method is CalibrationMethod.CLOSED_FORM:
        if not _is_one_sided_gaussian(reference_model):
            raise ValueError(
                "CalibrationMethod.CLOSED_FORM is only available for one-sided GaussianReferenceModel; "
                f"got {type(reference_model).__name__} sided={getattr(reference_model, 'sided', None)}"
            )
        return _siegmund_threshold(target_arl, float(slack_arr[0]))

    if method is CalibrationMethod.AUTO and _is_one_sided_gaussian(reference_model):
        return _siegmund_threshold(target_arl, float(slack_arr[0]))

    # Monte Carlo path.
    horizon = int(simulation_horizon) if simulation_horizon is not None else max(int(20 * target_arl), 5_000)
    if target_arl > horizon / 20.0:
        raise ValueError(
            f"target_arl={target_arl} exceeds simulation_horizon/20={horizon / 20.0}; "
            "increase simulation_horizon or lower target_arl."
        )
    sampler = _resolve_sampler(reference_model, in_control_sampler)
    # Common random numbers: each evaluate(h) reseeds from the same root so
    # ARL(h) is a deterministic monotonic function of h, removing MC noise from
    # the bisection. Without CRN, two evaluations at the same h could disagree
    # by ~SE/sqrt(n_sim) and the bisection might fail to converge.
    seed_root = int(np.random.SeedSequence(seed).generate_state(1)[0])

    def evaluate(h: float) -> float:
        rng_eval = np.random.default_rng(seed_root)
        mean_arl, censored = _empirical_arl(
            reference_model, slack_arr, h,
            n_simulations=n_simulations, simulation_horizon=horizon,
            in_control_sampler=sampler, rng=rng_eval,
        )
        if censored > 0.05:
            raise ValueError(
                f"Monte Carlo censoring fraction {censored:.3f} > 0.05 at h={h}; "
                "increase simulation_horizon or lower target_arl."
            )
        if censored > 0.0:
            warnings.warn(
                f"Monte Carlo censoring fraction {censored:.3f} at h={h} (censored runs treated as alarming at horizon).",
                RuntimeWarning,
                stacklevel=2,
            )
        return mean_arl

    # Bracket: empirical ARL is monotone non-decreasing in h.
    h_lo, h_hi = 0.5, 2.0
    arl_lo = evaluate(h_lo)
    arl_hi = evaluate(h_hi)
    doublings = 0
    while arl_hi < target_arl:
        h_hi *= 2.0
        doublings += 1
        if h_hi > _H_HI_CEIL or doublings > _BRACKET_MAX_DOUBLINGS:
            raise ValueError(
                f"target_arl={target_arl} unreachable: empirical ARL at h={h_hi} is {arl_hi}, "
                f"bracket capped at {_H_HI_CEIL}."
            )
        arl_hi = evaluate(h_hi)
    halvings = 0
    while arl_lo > target_arl:
        h_lo *= 0.5
        halvings += 1
        if h_lo < _H_LO_FLOOR or halvings > _BRACKET_MAX_DOUBLINGS:
            raise ValueError(
                f"target_arl={target_arl} unreachable: empirical ARL at h={h_lo} is {arl_lo}, "
                f"bracket floored at {_H_LO_FLOOR}."
            )
        arl_lo = evaluate(h_lo)

    # Bisect.
    last_arl = arl_lo
    for _ in range(_BISECT_MAX_ITER):
        h_mid = 0.5 * (h_lo + h_hi)
        arl_mid = evaluate(h_mid)
        last_arl = arl_mid
        if abs(arl_mid - target_arl) / target_arl <= _BISECT_TOLERANCE:
            return h_mid
        if arl_mid < target_arl:
            h_lo = h_mid
        else:
            h_hi = h_mid

    raise RuntimeError(
        f"Bisection failed to converge within {_BISECT_MAX_ITER} iterations: "
        f"final bracket [{h_lo}, {h_hi}], last empirical ARL={last_arl}, target={target_arl}. "
        "Increase n_simulations or loosen tolerance."
    )


def calibrate_from_window(xs: NDArray, reference_factory: Callable[..., ReferenceModel],
                          slack: float | NDArray, target_arl: float,
                          **kwargs) -> tuple[ReferenceModel, float]:
    """Fit reference + threshold from a calibration window. See spec §6."""
    if reference_factory not in _SUPPORTED_FACTORIES:
        raise TypeError(
            f"calibrate_from_window supports {[c.__name__ for c in _SUPPORTED_FACTORIES]}, "
            f"got {getattr(reference_factory, '__name__', reference_factory)!r}"
        )
    values = np.asarray(xs, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"xs must be one-dimensional, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("xs contains NaN or Inf values.")

    sided = kwargs.pop("sided", "two")
    in_control_sampler = kwargs.pop("in_control_sampler", None)
    seed = kwargs.pop("seed", None)
    target_kwargs = {
        k: kwargs.pop(k) for k in ("n_simulations", "simulation_horizon", "method")
        if k in kwargs
    }

    if reference_factory is GaussianReferenceModel:
        if len(values) < 2:
            raise ValueError(f"GaussianReferenceModel calibration requires len(xs) >= 2, got {len(values)}")
        std = float(np.std(values, ddof=1))
        if std <= 0.0:
            raise ValueError(f"sigma estimate is non-positive ({std}); calibration window is constant.")
        reference = GaussianReferenceModel(mu_0=float(np.mean(values)), sigma=std, sided=sided)

    elif reference_factory is SignCusumReferenceModel:
        reference = SignCusumReferenceModel(median_0=float(np.median(values)), sided=sided)

    elif reference_factory is EwmaReferenceModel:
        for required in ("alpha_mu", "alpha_sigma"):
            if required not in kwargs:
                raise ValueError(f"EwmaReferenceModel calibration requires `{required}` keyword argument.")
        if len(values) < 2:
            raise ValueError(f"EwmaReferenceModel calibration requires len(xs) >= 2, got {len(values)}")
        std = float(np.std(values, ddof=1))
        if std <= 0.0:
            raise ValueError(f"sigma_0 estimate is non-positive ({std}); calibration window is constant.")
        ewma_kwargs = {k: kwargs.pop(k) for k in ("alpha_mu", "alpha_sigma", "min_sigma") if k in kwargs}
        reference = EwmaReferenceModel(
            mu_0=float(np.mean(values)), sigma_0=std, sided=sided, **ewma_kwargs,
        )
        if in_control_sampler is None:
            if len(values) < _MIN_CALIBRATION_WINDOW:
                raise ValueError(
                    f"EwmaReferenceModel bootstrap sampler requires len(xs) >= {_MIN_CALIBRATION_WINDOW}, "
                    f"got {len(values)}; pass `in_control_sampler` explicitly to bypass."
                )
            xs_for_bootstrap = values.copy()

            def _bootstrap_sampler(n: int, rng: np.random.Generator) -> NDArray:
                return rng.choice(xs_for_bootstrap, size=int(n), replace=True)

            in_control_sampler = _bootstrap_sampler

    else:  # defensive — already gated by _SUPPORTED_FACTORIES check above
        raise TypeError(f"Unhandled reference_factory: {reference_factory!r}")

    if kwargs:
        raise TypeError(f"calibrate_from_window: unexpected kwargs {sorted(kwargs)}")

    threshold = target_arl_threshold(
        reference, slack=slack, target_arl=target_arl,
        in_control_sampler=in_control_sampler, seed=seed, **target_kwargs,
    )
    return reference, threshold
