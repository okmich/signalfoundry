"""
Emission log densities and EM updates, with a compiled fast path for float64 ``Normal`` states.

Pomegranate evaluates and summarises each state separately through torch, with argument checks on every call. When
every state is an initialised float64 pomegranate ``Normal`` with the same covariance type (full, diag or sphere),
the kernels below compute the same quantities in one parallel pass over the data, from each distribution's own cached
parameters (``_inv_cov``, ``_log_sigma_sqrt_2pi`` ...). The sufficient statistics are added to the distributions'
own buffers and their own ``from_summaries`` runs, so ``min_cov``, inertia and frozen states behave exactly as in
pomegranate. Anything else (other families, float32 states) goes through pomegranate unchanged.

Sums over bars use fixed-size row chunks added in a fixed order, so results do not depend on the thread count.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from numba import njit, prange
from pomegranate.distributions import Normal

_LOG_2_PI = math.log(2 * math.pi)
_ROW_CHUNK = 4096


@njit(cache=True, nogil=True, parallel=True)
def _diag_log_density(X: np.ndarray, means: np.ndarray, log_norm: np.ndarray, inv_two_var: np.ndarray) -> np.ndarray:
    """``sum_m log_norm[k, m] - (X[t, m] - means[k, m])^2 * inv_two_var[k, m]``, shape ``(T, K)``."""
    T, d = X.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for t in prange(T):
        for k in range(K):
            acc = 0.0
            for m in range(d):
                diff = X[t, m] - means[k, m]
                acc += log_norm[k, m] - diff * diff * inv_two_var[k, m]
            out[t, k] = acc
    return out


@njit(cache=True, nogil=True, parallel=True)
def _full_log_density(X: np.ndarray, inv_cov: np.ndarray, inv_cov_dot_mu: np.ndarray,
                      log_det: np.ndarray) -> np.ndarray:
    """Pomegranate's full-covariance form: ``log_det[k] - 0.5 (d log 2 pi + |X inv_cov[k] - inv_cov_dot_mu[k]|^2)``."""
    T, d = X.shape
    K = inv_cov.shape[0]
    out = np.empty((T, K))
    for t in prange(T):
        for k in range(K):
            q = 0.0
            for j in range(d):
                z = 0.0
                for m in range(d):
                    z += X[t, m] * inv_cov[k, m, j]
                z -= inv_cov_dot_mu[k, j]
                q += z * z
            out[t, k] = log_det[k] - 0.5 * (d * _LOG_2_PI + q)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _weighted_moments(X: np.ndarray, gamma: np.ndarray, full: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per state: ``sum_t w``, ``sum_t x w`` and ``sum_t (x w) x^T`` (full, (K, d, d)) or ``sum_t x^2 w`` (diag,
    (K, d, 1)), with ``w = gamma[:, k]``: pomegranate's ``summarize`` statistics.
    """
    T, d = X.shape
    K = gamma.shape[1]
    d2 = d if full else 1
    n_chunks = (T + _ROW_CHUNK - 1) // _ROW_CHUNK
    cw = np.zeros((n_chunks, K))
    cxw = np.zeros((n_chunks, K, d))
    cxxw = np.zeros((n_chunks, K, d, d2))
    for chunk in prange(n_chunks):
        for t in range(chunk * _ROW_CHUNK, min(T, (chunk + 1) * _ROW_CHUNK)):
            for k in range(K):
                w = gamma[t, k]
                cw[chunk, k] += w
                for m in range(d):
                    xw = X[t, m] * w
                    cxw[chunk, k, m] += xw
                    if full:
                        for m2 in range(d):
                            cxxw[chunk, k, m, m2] += xw * X[t, m2]
                    else:
                        cxxw[chunk, k, m, 0] += X[t, m] * X[t, m] * w
    w_sum = np.zeros(K)
    xw_sum = np.zeros((K, d))
    xxw_sum = np.zeros((K, d, d2))
    for chunk in range(n_chunks):  # fixed order
        for k in range(K):
            w_sum[k] += cw[chunk, k]
            for m in range(d):
                xw_sum[k, m] += cxw[chunk, k, m]
                for m2 in range(d2):
                    xxw_sum[k, m, m2] += cxxw[chunk, k, m, m2]
    return w_sum, xw_sum, xxw_sum


def _normal_kind(dists: list) -> Optional[str]:
    """The shared covariance type when every state can take the fast path, else None."""
    kinds = set()
    for dist in dists:
        if type(dist) is not Normal or not dist._initialized or dist.means.dtype != torch.float64:
            return None
        cached = "_inv_cov" if dist.covariance_type == "full" else "_inv_two_sigma"
        if not hasattr(dist, cached):
            return None  # degenerate covariances: let pomegranate raise its own error
        kinds.add(dist.covariance_type)
    return kinds.pop() if len(kinds) == 1 else None


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def log_emissions(dists: list, X: np.ndarray, X_t: Optional[torch.Tensor] = None) -> np.ndarray:
    """Per-state emission log densities, shape ``(T, K)``. ``X`` is float64 (T, d); ``X_t`` its tensor, if at hand."""
    kind = _normal_kind(dists)
    if kind == "full":
        return _full_log_density(X, np.stack([_numpy(d._inv_cov) for d in dists]),
                                 np.stack([_numpy(d._inv_cov_dot_mu) for d in dists]),
                                 np.array([float(d._log_det) for d in dists]))
    if kind in ("diag", "sphere"):
        width = (X.shape[1],)  # sphere: one variance broadcast over the features
        return _diag_log_density(X, np.stack([np.broadcast_to(_numpy(d.means), width) for d in dists]),
                                 np.stack([np.broadcast_to(_numpy(d._log_sigma_sqrt_2pi), width) for d in dists]),
                                 np.stack([np.broadcast_to(_numpy(d._inv_two_sigma), width) for d in dists]))
    X_t = torch.as_tensor(X, dtype=torch.float64) if X_t is None else X_t
    log_B = np.empty((X_t.shape[0], len(dists)), dtype=np.float64)
    for j, dist in enumerate(dists):
        log_B[:, j] = _numpy(dist.log_probability(X_t))
    return log_B


def update_emissions(dists: list, X: np.ndarray, X_t: torch.Tensor, gamma: np.ndarray) -> None:
    """EM update of every state from the posteriors ``gamma`` (T, K): ``summarize`` then ``from_summaries``."""
    kind = _normal_kind(dists)
    if kind is None:
        for k, dist in enumerate(dists):
            dist.summarize(X_t, sample_weight=torch.as_tensor(gamma[:, k:k + 1], dtype=torch.float64))
            dist.from_summaries()
        return
    w_sum, xw_sum, xxw_sum = _weighted_moments(X, gamma, kind == "full")
    d = X.shape[1]
    for k, dist in enumerate(dists):
        if not bool(dist.frozen):  # pomegranate's summarize is a no-op on a frozen state
            dist._w_sum += torch.from_numpy(np.full(d, w_sum[k]))
            dist._xw_sum += torch.from_numpy(xw_sum[k].copy())
            dist._xxw_sum += torch.from_numpy(xxw_sum[k].copy() if kind == "full" else xxw_sum[k, :, 0].copy())
        dist.from_summaries()
