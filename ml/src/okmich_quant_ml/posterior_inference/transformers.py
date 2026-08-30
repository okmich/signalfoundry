from __future__ import annotations

import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from .features import _require_integral, _validate_posterior_matrix


@njit(cache=True)
def _ema_recurrence(probs: np.ndarray, alpha: float) -> np.ndarray:
    n_rows, n_cols = probs.shape
    out = np.empty_like(probs)
    out[0] = probs[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n_rows):
        for k in range(n_cols):
            out[i, k] = alpha * probs[i, k] + one_minus_alpha * out[i - 1, k]
    return out


def _scalar_nll(probs: NDArray, y_idx: NDArray, eps: float) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    chosen = np.clip(p[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


class EmaPosteriorTransformer:
    """Exponential moving average smoothing on posterior rows.

    The recurrence is JIT-compiled via ``@njit(cache=True)``. The very first
    call after a fresh install incurs a one-time Numba compile cost (~1s);
    subsequent calls within the same install load the cached artifact and
    pay no compile overhead. Production deployments that warm the inference
    path at startup (e.g. a pre-market smoke run against historical data)
    absorb this transparently.
    """

    def __init__(self, alpha: float = 0.20, eps: float = 1e-12) -> None:
        self.alpha = float(alpha)
        self.eps = float(eps)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "EmaPosteriorTransformer", eps=self.eps, normalize=True)
        if p.shape[0] == 0:
            return p
        out = _ema_recurrence(p, self.alpha)
        np.clip(out, self.eps, None, out=out)
        out /= out.sum(axis=1, keepdims=True)
        return out


class RollingMeanPosteriorTransformer:
    """Trailing rolling-mean smoothing on posterior rows.

    At each row ``t`` the output is the mean of rows in the trailing window ``[max(0, t - window + 1), t]``.
    Causal at every row: never reads future observations. For ``t < window - 1`` the window is shorter, matching what
    a live stream would have available.
    """

    def __init__(self, window: int = 5, eps: float = 1e-12) -> None:
        self.window = _require_integral(window, "window")
        self.eps = float(eps)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "RollingMeanPosteriorTransformer", eps=self.eps, normalize=True)
        n_rows = p.shape[0]
        if n_rows == 0:
            return p

        window = self.window
        out = np.cumsum(p, axis=0)
        if n_rows > window:
            # The copy() guards against aliasing: out[window:] and out[:-window]
            # share index `window` in memory, so in-place subtraction without the
            # copy would read already-updated elements further along the slice.
            out[window:] -= out[:-window].copy()
            denom = np.full((n_rows, 1), float(window))
            denom[:window, 0] = np.arange(1, window + 1)
        else:
            denom = np.arange(1, n_rows + 1, dtype=out.dtype).reshape(-1, 1)

        out /= denom
        np.clip(out, self.eps, None, out=out)
        out /= out.sum(axis=1, keepdims=True)
        return out


class TemperatureScalingTransformer:
    """Post-hoc temperature scaling on posterior probabilities.

    Calibration is specific to the posterior distribution it is fit on. Filtered (``lag=0``) and matured (``lag > 0``)
    posteriors from the same HMM have different entropy profiles, so a temperature fit on one regime applied to the other will
    produce miscalibrated output. Always fit on the same lag regime used at inference.

    ``fit`` uses bounded Brent (``scipy.optimize.minimize_scalar``) over
    ``[search_min, search_max]``. NLL as a function of temperature is unimodal for
    well-posed calibration data, so Brent converges in ~10–20 evaluations.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-12,
                 search_min: float = 0.25, search_max: float = 20.0) -> None:
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.search_min = float(search_min)
        self.search_max = float(search_max)
        self.temperature_: float | None = None
        self.fit_success_: bool = False
        self.fit_message_: str = "not fitted"
        self._validate_configuration()

    def fit(self, probs: NDArray, y_idx: NDArray) -> TemperatureScalingTransformer:
        p = _validate_posterior_matrix(probs, "TemperatureScalingTransformer", eps=self.eps, normalize=True)
        y_raw = np.asarray(y_idx)
        if not np.issubdtype(y_raw.dtype, np.integer):
            raise ValueError(
                f"TemperatureScalingTransformer.fit: y_idx must be an integer array (got dtype {y_raw.dtype}); "
                f"silent float-to-int truncation would corrupt the calibration targets."
            )
        y = y_raw.astype(np.int64)
        if y.ndim != 1:
            raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
        if len(y) != len(p):
            raise ValueError(f"y_idx length must equal number of rows in probs. Got {len(y)} vs {len(p)}.")
        if len(y) == 0:
            raise ValueError("TemperatureScalingTransformer.fit: empty calibration set; calibration is undefined.")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError(f"y_idx must be in [0, K-1] where K={p.shape[1]}.")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "TemperatureScalingTransformer.fit: degenerate calibration set — y_idx contains only one class; "
                "calibration is undefined."
            )

        def nll_at_temperature(t: float) -> float:
            return _scalar_nll(self._apply_temperature(p, t), y, eps=self.eps)

        result = minimize_scalar(
            nll_at_temperature,
            bounds=(self.search_min, self.search_max),
            method="bounded",
        )
        self.fit_success_ = bool(result.success)
        self.fit_message_ = str(getattr(result, "message", "")) or ("converged" if result.success else "failed")
        if not result.success:
            raise RuntimeError(
                f"TemperatureScalingTransformer.fit: optimizer did not converge ({self.fit_message_}). "
                f"Inspect the calibration data for degeneracies (e.g., single-class y_idx, near-constant probs)."
            )
        self.temperature_ = float(result.x)
        return self

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "TemperatureScalingTransformer", eps=self.eps, normalize=True)
        t = self.temperature_ if self.temperature_ is not None else self.temperature
        return self._apply_temperature(p, t)

    def _apply_temperature(self, probs: NDArray, temperature: float) -> NDArray:
        if not (np.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {temperature}")
        logits = np.log(np.clip(probs, self.eps, 1.0)) / temperature
        logits -= logsumexp(logits, axis=1, keepdims=True)
        out = np.exp(logits)
        np.clip(out, self.eps, None, out=out)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def _validate_configuration(self) -> None:
        if not (np.isfinite(self.temperature) and self.temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {self.temperature}")
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and > 0, got {self.eps}")
        search_min_ok = np.isfinite(self.search_min) and self.search_min > 0.0
        search_max_ok = np.isfinite(self.search_max) and self.search_max > 0.0
        if not search_min_ok or not search_max_ok:
            raise ValueError("search_min and search_max must be finite and > 0.")
        if self.search_min >= self.search_max:
            raise ValueError(f"search_min must be < search_max, got {self.search_min} >= {self.search_max}")


class PlattScalingTransformer:
    """Multiclass Platt calibration via vector scaling.

    Learns per-class scale ``a`` and bias ``b`` on log-posteriors, then re-softmaxes:

        l_ik = a_k * log(p_ik) + b_k
        p_cal_i = softmax(l_i)

    The identity parameters ``a = 1, b = 0`` reproduce the input posteriors exactly (up to floating-point renormalisation),
    so an un-fit transformer is a no-op. ``fit(probs, y_idx)`` minimises mean cross-entropy via L-BFGS-B using analytic gradients.

    Calibration is specific to the posterior distribution it is fit on. Filtered (``lag=0``) and matured (``lag > 0``)
    posteriors from the same HMM have different entropy profiles, so parameters fit on one regime applied to the other will produce
    miscalibrated output. Always fit on the same lag regime used at inference.
    """

    def __init__(self, eps: float = 1e-12, max_iter: int = 200, tol: float = 1e-8) -> None:
        self.eps = float(eps)
        self.max_iter = _require_integral(max_iter, "max_iter")
        self.tol = float(tol)
        self.a_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.fit_success_: bool = False
        self.fit_message_: str = "not fitted"
        self._validate_configuration()

    def fit(self, probs: NDArray, y_idx: NDArray) -> PlattScalingTransformer:
        p = _validate_posterior_matrix(probs, "PlattScalingTransformer", eps=self.eps, normalize=True)
        y_raw = np.asarray(y_idx)
        if not np.issubdtype(y_raw.dtype, np.integer):
            raise ValueError(
                f"PlattScalingTransformer.fit: y_idx must be an integer array (got dtype {y_raw.dtype}); "
                f"silent float-to-int truncation would corrupt the calibration targets."
            )
        y = y_raw.astype(np.int64)
        if y.ndim != 1:
            raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
        if len(y) != len(p):
            raise ValueError(f"y_idx length must equal number of rows in probs. Got {len(y)} vs {len(p)}.")
        if len(y) == 0:
            raise ValueError("PlattScalingTransformer.fit: empty calibration set; calibration is undefined.")
        n_classes = p.shape[1]
        if np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError(f"y_idx must be in [0, K-1] where K={n_classes}.")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "PlattScalingTransformer.fit: degenerate calibration set — y_idx contains only one class; "
                "calibration is undefined."
            )

        log_p = np.log(p)
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(len(y)), y] = 1.0
        sample_rows = np.arange(len(y))
        n_samples = float(len(y))

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a = theta[:n_classes]
            b = theta[n_classes:]
            logits = log_p * a + b
            log_z = logsumexp(logits, axis=1, keepdims=True)
            log_p_cal = logits - log_z
            loss = float(-np.mean(log_p_cal[sample_rows, y]))
            p_cal = np.exp(log_p_cal)
            diff = (p_cal - y_onehot) / n_samples
            grad_a = np.sum(diff * log_p, axis=0)
            grad_b = np.sum(diff, axis=0)
            return loss, np.concatenate([grad_a, grad_b])

        theta0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        result = minimize(loss_and_grad, x0=theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol})
        self.fit_success_ = bool(result.success)
        self.fit_message_ = str(getattr(result, "message", "")) or ("converged" if result.success else "failed")
        if not result.success:
            raise RuntimeError(
                f"PlattScalingTransformer.fit: optimizer did not converge ({self.fit_message_}). "
                f"Inspect the calibration data for degeneracies (e.g., single-class y_idx, separable probs)."
            )
        theta = result.x
        self.a_ = np.asarray(theta[:n_classes], dtype=float)
        self.b_ = np.asarray(theta[n_classes:], dtype=float)
        return self

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "PlattScalingTransformer", eps=self.eps, normalize=True)
        n_classes = p.shape[1]
        if self.a_ is None or self.b_ is None:
            warnings.warn(
                "PlattScalingTransformer.transform called before fit; falling back to identity. "
                "Call fit(probs, y_idx) first to learn calibration parameters.",
                UserWarning, stacklevel=2,
            )
        a = self.a_ if self.a_ is not None else np.ones(n_classes)
        b = self.b_ if self.b_ is not None else np.zeros(n_classes)
        if len(a) != n_classes or len(b) != n_classes:
            raise ValueError(f"Fitted parameters have K={len(a)}, but input has K={n_classes}.")
        logits = np.log(p) * a + b
        logits -= logsumexp(logits, axis=1, keepdims=True)
        out = np.exp(logits)
        np.clip(out, self.eps, None, out=out)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def _validate_configuration(self) -> None:
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and > 0, got {self.eps}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not (np.isfinite(self.tol) and self.tol > 0.0):
            raise ValueError(f"tol must be finite and > 0, got {self.tol}")


class MaturationAlignTransformer:
    """Align a forward-looking posterior matrix to the timestamp it is actually available at.

    ``predict_proba_fixed_lag(X, lag)`` returns row ``t`` = ``P(z_t | o_1..o_{t+lag})``. In live trading, that posterior
    for time ``t`` cannot be observed until bar ``t + lag`` arrives, so a decision made *at* time ``t`` must consume the
    posterior for time ``t - lag`` (the row that was computable with data available up to ``t``).

    This transformer performs that shift: output row ``t`` = input row ``t - lag`` for ``t >= lag``, and the uniform
    prior ``[1/K, ..., 1/K]`` for ``t < lag`` (warmup). Uniform prior is chosen over NaN because every gate inferer in
    this package validates and rejects NaN at entry — emitting NaN would break pipeline composition with the standard
    gates. The trade-off: aggregate metrics computed naively across the full output (e.g. ``mean(top_prob(out))``) will
    include ``1/K`` placeholder mass for the first ``lag`` rows; callers reporting aggregates should slice off the
    warmup region or use NaN-aware aggregation upstream.

    Alignment is a pure rearrangement on the post-warmup tail: the transformer does not clip or renormalize the input,
    so degenerate posteriors (e.g. rows with exact zeros) survive unchanged.
    """

    def __init__(self, lag: int) -> None:
        self.lag = _require_integral(lag, "lag")
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {self.lag}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "MaturationAlignTransformer")
        T, K = p.shape
        out = np.empty((T, K), dtype=p.dtype)
        effective_lag = min(self.lag, T)
        if effective_lag > 0:
            out[:effective_lag] = 1.0 / K
        if effective_lag < T:
            out[effective_lag:] = p[: T - effective_lag]
        return out


@njit(cache=True)
def _kalman_posterior_smooth(probs: np.ndarray, process_noise: float, measurement_noise: float,
                             adaptation_rate: float, error_window: int, eps: float) -> np.ndarray:
    """Bayesian recursive smoother on the simplex.

    Per-bar dynamics:

    1. **Diffuse** the prior belief toward uniform by ``process_noise``:
       ``b ← (1 - process_noise) * b + process_noise * uniform``.
    2. **Update** with the observation via a Bayesian step in log space:
       ``log b ← log b + measurement_weight * log p_t``, then softmax.
       ``measurement_weight = 1 / (measurement_noise + eps)``.

    The update is computed in log space + softmax (rather than ``p_t ^ measurement_weight``
    in linear space) so that small ``measurement_noise`` — equivalently large ``measurement_weight``
    — does not underflow the likelihood to zero. The mathematically equivalent linear-space form is
    safe for moderate weights but collapses to uniform when ``measurement_weight ≳ 30`` for typical
    posteriors; the log-space form has no such regime.

    Adaptation (when ``adaptation_rate > 0``): a moving prediction error — the fraction of recent bars
    where ``argmax(diffused_belief) != argmax(p_t)`` — drives ``measurement_noise`` via an EMA toward
    the observed error. Higher error ⇒ higher measurement noise ⇒ less weight on each observation.
    """
    T, K = probs.shape
    out = np.empty_like(probs)
    belief = np.full(K, 1.0 / K, dtype=np.float64)
    log_belief = np.empty(K, dtype=np.float64)
    recent_errors = np.zeros(error_window, dtype=np.float64)
    error_idx = 0

    for t in range(T):
        # Diffuse: mix toward uniform.
        for k in range(K):
            belief[k] = (1.0 - process_noise) * belief[k] + process_noise * (1.0 / K)
        total = 0.0
        for k in range(K):
            total += belief[k]
        if total > 0.0:
            for k in range(K):
                belief[k] /= total

        # Prediction error (binary) using diffused belief vs current observation argmax.
        pred_arg = 0
        pred_max = belief[0]
        obs_arg = 0
        obs_max = probs[t, 0]
        for k in range(1, K):
            if belief[k] > pred_max:
                pred_max = belief[k]
                pred_arg = k
            if probs[t, k] > obs_max:
                obs_max = probs[t, k]
                obs_arg = k
        pred_error = 0.0 if pred_arg == obs_arg else 1.0

        # Update: Bayesian step in log space + softmax.
        # log b_new = log b_diffused + measurement_weight * log p_t  (up to a constant).
        # Computing in log space avoids underflow when measurement_weight is large
        # (small measurement_noise tempers the likelihood aggressively).
        measurement_weight = 1.0 / (measurement_noise + eps)
        for k in range(K):
            b = belief[k] if belief[k] >= eps else eps
            p = probs[t, k] if probs[t, k] >= eps else eps
            log_belief[k] = np.log(b) + measurement_weight * np.log(p)

        # Numerically stable softmax: subtract max before exp.
        log_max = log_belief[0]
        for k in range(1, K):
            if log_belief[k] > log_max:
                log_max = log_belief[k]
        total = 0.0
        for k in range(K):
            belief[k] = np.exp(log_belief[k] - log_max)
            total += belief[k]
        if total > 0.0:
            for k in range(K):
                belief[k] /= total
        else:
            for k in range(K):
                belief[k] = 1.0 / K

        # Floor + renormalise so downstream log-based math is safe.
        total = 0.0
        for k in range(K):
            if belief[k] < eps:
                belief[k] = eps
            total += belief[k]
        for k in range(K):
            out[t, k] = belief[k] / total
            belief[k] = out[t, k]

        # Record error and adapt measurement noise.
        recent_errors[error_idx % error_window] = pred_error
        error_idx += 1
        if adaptation_rate > 0.0 and error_idx > error_window:
            mean_error = 0.0
            for j in range(error_window):
                mean_error += recent_errors[j]
            mean_error /= error_window
            target = mean_error
            if target < 0.01:
                target = 0.01
            elif target > 1.0:
                target = 1.0
            measurement_noise = (1.0 - adaptation_rate) * measurement_noise + adaptation_rate * target
    return out


class KalmanPosteriorTransformer:
    """Bayesian recursive smoother on the simplex — the posterior-space analogue of Kalman filtering.

    The numba kernel ``_kalman_posterior_smooth`` runs per-bar diffuse + Bayesian update; outputs are
    re-clipped and renormalised so every row remains a valid simplex. This is the **transformer**
    redesign of the regime_filters ``AdaptiveKalmanStyleSmoother`` — input and output are both
    ``(T, K)`` posteriors, so it composes with argmax, gates, calibrators, and drift monitoring just
    like ``EmaPosteriorTransformer``.

    Hyperparameters:

    * ``process_noise ∈ [0, 1]`` — how aggressively belief diffuses back toward uniform each bar.
      0 = no diffusion (belief never forgets), 1 = full reset to uniform per bar (smoothing disabled).
    * ``measurement_noise ∈ (0, ∞)`` — Bayesian-update "trust" knob. The likelihood is tempered to
      ``posterior ^ (1 / measurement_noise)``. Small values (≪ 1) sharpen the update (trust the bar);
      large values (> 1) flatten the update (distrust the bar).
    * ``adaptation_rate ∈ [0, 1)`` — when > 0, ``measurement_noise`` adapts via EMA toward the recent
      prediction error (fraction of bars where the diffused belief's argmax disagreed with the
      observation's argmax). ``0`` disables adaptation.

    Compared to ``EmaPosteriorTransformer`` (additive smoothing in probability space), this performs
    *Bayesian* smoothing: the update is multiplicative on log-probabilities, so a single decisive
    observation can shift belief faster than an EMA at the same nominal "smoothing rate."

    The "process noise = mix-toward-uniform" is a Kalman-style heuristic rather than strict
    linear-Gaussian dynamics — the simplex is neither linear nor Gaussian. A theoretically cleaner
    alternative is to run Kalman on the additive log-ratio transform and softmax back; that's a
    separate operator. This one matches the regime_filters implementation it replaces.
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 0.2,
                 adaptation_rate: float = 0.0, error_window: int = 10, eps: float = 1e-12) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.adaptation_rate = float(adaptation_rate)
        self.error_window = _require_integral(error_window, "error_window")
        self.eps = float(eps)
        if not 0.0 <= self.process_noise <= 1.0:
            raise ValueError(f"process_noise must be in [0, 1], got {self.process_noise}")
        if not (np.isfinite(self.measurement_noise) and self.measurement_noise > 0.0):
            raise ValueError(f"measurement_noise must be finite and > 0, got {self.measurement_noise}")
        if not 0.0 <= self.adaptation_rate < 1.0:
            raise ValueError(f"adaptation_rate must be in [0, 1), got {self.adaptation_rate}")
        if self.error_window < 1:
            raise ValueError(f"error_window must be >= 1, got {self.error_window}")
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "KalmanPosteriorTransformer", eps=self.eps, normalize=True)
        if p.shape[0] == 0:
            return p
        return _kalman_posterior_smooth(p, self.process_noise, self.measurement_noise,
                                        self.adaptation_rate, self.error_window, self.eps)
