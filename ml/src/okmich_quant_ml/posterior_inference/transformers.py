from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


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


def _as_probability_matrix(probs: NDArray, eps: float = 1e-12, normalize: bool = True) -> NDArray:
    """Validate a posterior matrix and optionally normalize rows to the simplex.

    With ``normalize=True`` (default), the returned array is clipped to ``eps`` and
    row-renormalised so downstream log-space computations are safe. With
    ``normalize=False``, only the shape / NaN-Inf validation runs and the input
    values pass through unchanged — use this for rearrangement transformers that
    should not silently alter probabilities.
    """
    array = np.asarray(probs, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected posterior matrix with shape (T, K), got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError(f"Expected at least 2 classes (K >= 2), got K={array.shape[1]}")
    if array.size > 0 and not np.isfinite(array.sum()):
        raise ValueError("Posterior matrix contains NaN or Inf values.")
    if not normalize:
        return array

    clipped = np.clip(array, eps, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    if row_sums.size > 0 and row_sums.min() <= 0.0:
        raise ValueError("Posterior rows must have strictly positive sums.")
    clipped /= row_sums
    return clipped


def _scalar_nll(probs: NDArray, y_idx: NDArray, eps: float) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    chosen = np.clip(p[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


class EmaPosteriorTransformer:
    """Exponential moving average smoothing on posterior rows."""

    def __init__(self, alpha: float = 0.20, eps: float = 1e-12) -> None:
        self.alpha = float(alpha)
        self.eps = float(eps)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
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
        self.window = int(window)
        self.eps = float(eps)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
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

    Calibration is specific to the posterior distribution it is fit on. Filtered
    (``lag=0``) and matured (``lag > 0``) posteriors from the same HMM have different
    entropy profiles, so a temperature fit on one regime applied to the other will
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
        self._validate_configuration()

    def fit(self, probs: NDArray, y_idx: NDArray) -> TemperatureScalingTransformer:
        p = _as_probability_matrix(probs, eps=self.eps)
        y = np.asarray(y_idx, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
        if len(y) != len(p):
            raise ValueError(f"y_idx length must equal number of rows in probs. Got {len(y)} vs {len(p)}.")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError(f"y_idx must be in [0, K-1] where K={p.shape[1]}.")

        def nll_at_temperature(t: float) -> float:
            return _scalar_nll(self._apply_temperature(p, t), y, eps=self.eps)

        result = minimize_scalar(
            nll_at_temperature,
            bounds=(self.search_min, self.search_max),
            method="bounded",
        )
        # Guard against Brent returning an invalid solution for pathological input.
        if result.success:
            self.temperature = float(result.x)
        return self

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
        return self._apply_temperature(p, self.temperature)

    def _apply_temperature(self, probs: NDArray, temperature: float) -> NDArray:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        logits = np.log(np.clip(probs, self.eps, 1.0)) / temperature
        logits -= logsumexp(logits, axis=1, keepdims=True)
        out = np.exp(logits)
        np.clip(out, self.eps, None, out=out)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def _validate_configuration(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.search_min <= 0.0 or self.search_max <= 0.0:
            raise ValueError("search_min and search_max must be > 0.")
        if self.search_min >= self.search_max:
            raise ValueError(f"search_min must be < search_max, got {self.search_min} >= {self.search_max}")


class PlattScalingTransformer:
    """Multiclass Platt calibration via vector scaling.

    Learns per-class scale ``a`` and bias ``b`` on log-posteriors, then re-softmaxes:

        l_ik = a_k * log(p_ik) + b_k
        p_cal_i = softmax(l_i)

    The identity parameters ``a = 1, b = 0`` reproduce the input posteriors exactly (up to floating-point renormalisation),
    so an un-fit transformer is a no-op. ``fit(probs, y_idx)`` minimises mean cross-entropy via L-BFGS-B using analytic gradients.

    Calibration is specific to the posterior distribution it is fit on. Filtered
    (``lag=0``) and matured (``lag > 0``) posteriors from the same HMM have different
    entropy profiles, so parameters fit on one regime applied to the other will produce
    miscalibrated output. Always fit on the same lag regime used at inference.
    """

    def __init__(self, eps: float = 1e-12, max_iter: int = 200, tol: float = 1e-8) -> None:
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.a_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self._validate_configuration()

    def fit(self, probs: NDArray, y_idx: NDArray) -> PlattScalingTransformer:
        p = _as_probability_matrix(probs, eps=self.eps)
        y = np.asarray(y_idx, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
        if len(y) != len(p):
            raise ValueError(f"y_idx length must equal number of rows in probs. Got {len(y)} vs {len(p)}.")
        n_classes = p.shape[1]
        if np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError(f"y_idx must be in [0, K-1] where K={n_classes}.")

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
        theta = result.x
        self.a_ = np.asarray(theta[:n_classes], dtype=float)
        self.b_ = np.asarray(theta[n_classes:], dtype=float)
        return self

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
        n_classes = p.shape[1]
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
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {self.tol}")


class MaturationAlignTransformer:
    """Align a forward-looking posterior matrix to the timestamp it is actually available at.

    ``predict_proba_fixed_lag(X, lag)`` returns row ``t`` = ``P(z_t | o_1..o_{t+lag})``. In live
    trading, that posterior for time ``t`` cannot be observed until bar ``t + lag`` arrives,
    so a decision made *at* time ``t`` must consume the posterior for time ``t - lag`` (the
    row that was computable with data available up to ``t``).

    This transformer performs that shift: output row ``t`` = input row ``t - lag`` for
    ``t >= lag``, and a uniform prior ``[1/K, ..., 1/K]`` for ``t < lag``. The result has
    the same shape as the input and is safe to read by time index in a backtest without
    leaking future information.

    Alignment is a pure rearrangement: the transformer does not clip or renormalize the
    input, so degenerate posteriors (e.g. rows with exact zeros) survive unchanged.
    """

    def __init__(self, lag: int) -> None:
        self.lag = int(lag)
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {self.lag}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, normalize=False)
        T, K = p.shape
        out = np.empty((T, K), dtype=p.dtype)
        effective_lag = min(self.lag, T)
        if effective_lag > 0:
            out[:effective_lag] = 1.0 / K
        if effective_lag < T:
            out[effective_lag:] = p[: T - effective_lag]
        return out
