"""Emission blocks of a hierarchical HMM.

A block reads a set of columns of X and gives every leaf a log density for them. ``tie_level`` shares the block's
parameters between leaves: leaves with the same ancestor at that depth form one *group* and use one distribution
(``None`` = one distribution per leaf; ``0`` = one for all leaves). Tying lets a level own its features: e.g. slow
features tied at depth 1 inform the top-level state, fast features untied inform the leaves under it.

Blocks multiply: a leaf's emission density is the product over blocks. A row with a NaN in any of a block's columns
is unobserved for that block: it contributes log density 0 and is skipped in the M-step.

``param_names`` lists the learnt parameters; the model copies exactly these for warm starts and snapshots.
"""
from __future__ import annotations

import abc
import math
from enum import StrEnum
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

from .topology import Topology

_LOG_2PI = math.log(2.0 * math.pi)
_KMEANS_SAMPLE = 5000


class CovarianceType(StrEnum):
    """Covariance structure of a multivariate emission."""

    DIAG = "diag"
    FULL = "full"


class LeafEmission(abc.ABC):
    """One emission block over ``columns``, with parameters per leaf group."""

    columns: tuple[int, ...]
    tie_level: Optional[int]
    param_names: tuple[str, ...] = ()

    def __init__(self, columns: Union[int, Sequence[int]], tie_level: Optional[int]):
        cols = (int(columns),) if np.isscalar(columns) else tuple(int(c) for c in columns)
        if not cols or any(c < 0 for c in cols) or len(set(cols)) != len(cols):
            raise ValueError(f"columns must be distinct non-negative indices, got {columns}")
        if tie_level is not None and int(tie_level) < 0:
            raise ValueError(f"tie_level must be None or >= 0, got {tie_level}")
        self.columns = cols
        self.tie_level = None if tie_level is None else int(tie_level)
        self.groups: Optional[np.ndarray] = None
        self.n_groups = 0

    # -- binding and parameter bookkeeping ------------------------------------------------------------------------
    def bind(self, topology: Topology) -> None:
        """Resolve leaf groups against the model's state tree."""
        self.groups = topology.group_of_leaves(self.tie_level)
        self.n_groups = int(self.groups.max()) + 1
        self._topology = topology

    def get_params(self) -> dict[str, np.ndarray]:
        return {name: np.array(getattr(self, name), copy=True) for name in self.param_names}

    def set_params(self, params: Mapping[str, np.ndarray]) -> None:
        for name in self.param_names:
            setattr(self, name, np.array(params[name], dtype=np.float64, copy=True))

    def load_params(self, other: "LeafEmission") -> None:
        self.set_params(other.get_params())

    def signature(self) -> tuple:
        """Identity of the block (kind, columns, tying)."""
        return type(self).__name__, self.columns, self.tie_level

    def config_signature(self) -> tuple:
        """Everything that fixes the parameter shapes and meaning; warm starts require it to match exactly."""
        return self.signature()

    def log_prior(self) -> float:
        """Log prior implied by the block's own pseudo-counts (0 for maximum-likelihood blocks). EM adds it to the
        objective it maximises, so convergence checks and restart ranking see what the M-step optimises."""
        return 0.0

    # -- the contract -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        """Starting parameters for fresh start ``restart``."""

    @abc.abstractmethod
    def log_density(self, X: np.ndarray) -> np.ndarray:
        """(T, n_groups) log density of the block's columns under every group."""

    @abc.abstractmethod
    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """Update from (T, n_groups) group posteriors."""

    @abc.abstractmethod
    def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(T, len(columns)) draws, one per entry of the group sequence ``groups``."""

    @abc.abstractmethod
    def n_free_params(self) -> int: ...

    # -- helpers ----------------------------------------------------------------------------------------------------
    def _block(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = X[:, list(self.columns)]
        return x, np.isfinite(x).all(axis=1)

    def _check_bound(self) -> None:
        if self.groups is None:
            raise RuntimeError(f"{type(self).__name__} is not bound to a model")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(columns={list(self.columns)}, tie_level={self.tie_level})"


# ======================================================================================================================
# Categorical
# ======================================================================================================================
class CategoricalEmission(LeafEmission):
    """x | group g ~ Categorical(probs_g) over symbols 0..n_symbols-1 in one column.

    Parameters
    ----------
    column : the column holding integer symbols (as floats; NaN = unobserved).
    n_symbols : alphabet size.
    support : symbols each leaf may emit (structural zeros). Either {leaf name: iterable of symbols} (unlisted leaves
        may emit every symbol) or an (N_leaves, n_symbols) bool array. Leaves tied into one group must share support.
    tie_level : see the module docstring.
    pseudo_count : added to every allowed symbol's expected count in the M-step, so a symbol unseen in training keeps
        a small probability out of sample. 0 = maximum likelihood.
    """

    param_names = ("probs",)

    def __init__(self, column: int, n_symbols: int,
                 support: Union[Mapping[str, Iterable[int]], np.ndarray, None] = None, tie_level: Optional[int] = None,
                 pseudo_count: float = 1e-3):
        super().__init__(column, tie_level)
        if int(n_symbols) < 1:
            raise ValueError(f"n_symbols must be >= 1, got {n_symbols}")
        if pseudo_count < 0:
            raise ValueError(f"pseudo_count must be >= 0, got {pseudo_count}")
        self.n_symbols = int(n_symbols)
        self.support_spec = support
        self.pseudo_count = float(pseudo_count)
        self.probs: Optional[np.ndarray] = None
        self.support: Optional[np.ndarray] = None

    def bind(self, topology: Topology) -> None:
        super().bind(topology)
        leaf_support = self._leaf_support(topology)
        support = np.zeros((self.n_groups, self.n_symbols), dtype=bool)
        for g in range(self.n_groups):
            members = leaf_support[self.groups == g]
            if not (members == members[0]).all():
                raise ValueError(f"Leaves tied into one group at tie_level={self.tie_level} have different support")
            support[g] = members[0]
        if not support.any(axis=1).all():
            raise ValueError("Every leaf must be allowed at least one symbol")
        self.support = support

    def _leaf_support(self, topology: Topology) -> np.ndarray:
        n = topology.n_leaves
        spec = self.support_spec
        if spec is None:
            return np.ones((n, self.n_symbols), dtype=bool)
        if isinstance(spec, Mapping):
            out = np.ones((n, self.n_symbols), dtype=bool)
            names = topology.leaf_names
            for name, symbols in spec.items():
                if name not in names:
                    raise ValueError(f"support names leaf {name!r}, which is not a leaf of the tree")
                row = np.zeros(self.n_symbols, dtype=bool)
                symbols = np.asarray(list(symbols), dtype=np.int64)
                if symbols.size and (symbols.min() < 0 or symbols.max() >= self.n_symbols):
                    raise ValueError(f"support of {name!r} lists symbols outside 0..{self.n_symbols - 1}")
                row[symbols] = True
                out[names.index(name)] = row
            return out
        arr = np.asarray(spec, dtype=bool)
        if arr.shape != (n, self.n_symbols):
            raise ValueError(f"support array needs shape ({n}, {self.n_symbols}), got {arr.shape}")
        return arr.copy()

    def _symbols(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, ok = self._block(X)
        x = x[:, 0]
        vals = x[ok]
        if vals.size and (np.any(vals != np.floor(vals)) or vals.min() < 0 or vals.max() >= self.n_symbols):
            raise ValueError(f"Column {self.columns[0]} must hold integer symbols in 0..{self.n_symbols - 1} or NaN")
        sym = np.zeros(len(x), dtype=np.int64)
        sym[ok] = vals.astype(np.int64)
        return sym, ok

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        self._check_bound()
        sym, ok = self._symbols(X)
        freq = np.bincount(sym[ok], minlength=self.n_symbols).astype(np.float64) + 1.0
        probs = np.zeros((self.n_groups, self.n_symbols))
        for g in range(self.n_groups):
            allowed = self.support[g]
            conc = 10.0 * freq[allowed] / freq[allowed].sum() + 0.1
            probs[g, allowed] = rng.dirichlet(conc)
        self.probs = probs

    def log_density(self, X: np.ndarray) -> np.ndarray:
        sym, ok = self._symbols(X)
        with np.errstate(divide="ignore"):
            logp = np.log(self.probs)                       # (G, S); structural zeros -> -inf
        out = logp[:, sym].T.copy()
        out[~ok] = 0.0
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        sym, ok = self._symbols(X)
        counts = np.zeros((self.n_groups, self.n_symbols))
        np.add.at(counts.T, sym[ok], gamma[ok])
        counts = np.where(self.support, counts + self.pseudo_count, 0.0)
        total = counts.sum(axis=1, keepdims=True)
        self.probs = np.where(total > 0, counts / np.where(total > 0, total, 1.0), self.probs)

    def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cum = np.cumsum(self.probs, axis=1)
        u = rng.random(len(groups))
        draws = np.array([min(int(np.searchsorted(cum[g], u[t], side="right")), self.n_symbols - 1)
                          for t, g in enumerate(groups)], dtype=np.float64)
        return draws[:, None]

    def n_free_params(self) -> int:
        return int((self.support.sum(axis=1) - 1).sum())

    def config_signature(self) -> tuple:
        return self.signature() + (self.n_symbols, self.support.tobytes())

    def log_prior(self) -> float:
        """Dirichlet(1 + pseudo_count) on each group's allowed symbols: the M-step above is its MAP update."""
        if self.pseudo_count <= 0 or self.probs is None:
            return 0.0
        allowed = self.probs[self.support]
        return float(self.pseudo_count * np.log(np.maximum(allowed, 1e-300)).sum())

    def __repr__(self) -> str:
        return (f"CategoricalEmission(column={self.columns[0]}, n_symbols={self.n_symbols}, "
                f"tie_level={self.tie_level})")


# ======================================================================================================================
# Gaussian-family helpers
# ======================================================================================================================
def _kmeanspp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeds from (a subsample of) the rows of x."""
    if len(x) > _KMEANS_SAMPLE:
        x = x[rng.choice(len(x), _KMEANS_SAMPLE, replace=False)]
    centres = [x[rng.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        if d2.sum() <= 0:
            centres.append(x[rng.integers(len(x))])
        else:
            centres.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(axis=1))
    return np.array(centres)


def _mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, full: bool) -> tuple[np.ndarray, float]:
    """Squared Mahalanobis distance of each row and log |cov|."""
    diff = x - mean
    if not full:
        return (diff ** 2 / cov).sum(axis=1), float(np.log(cov).sum())
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, diff.T)
    return (z ** 2).sum(axis=0), 2.0 * float(np.log(np.diag(chol)).sum())


class _LocationScaleBase(LeafEmission):
    """Shared state of the Gaussian and Student-t blocks: means (G, d) and covs (G, d) or (G, d, d)."""

    def __init__(self, columns: Union[int, Sequence[int]], covariance_type: CovarianceType, tie_level: Optional[int],
                 var_floor: float):
        super().__init__(columns, tie_level)
        if var_floor <= 0:
            raise ValueError(f"var_floor must be > 0, got {var_floor}")
        self.covariance_type = CovarianceType(covariance_type)
        self.var_floor = float(var_floor)
        self.means: Optional[np.ndarray] = None
        self.covs: Optional[np.ndarray] = None
        self._floor: Optional[np.ndarray] = None

    @property
    def full(self) -> bool:
        return self.covariance_type is CovarianceType.FULL

    @property
    def dim(self) -> int:
        return len(self.columns)

    def _data_floor(self, x: np.ndarray) -> np.ndarray:
        var = np.var(x, axis=0) if len(x) > 1 else np.ones(x.shape[1])
        return self.var_floor * np.maximum(var, 1e-300)

    def _init_location_scale(self, X: np.ndarray, rng: np.random.Generator) -> None:
        self._check_bound()
        x, ok = self._block(X)
        x = x[ok]
        if len(x) < 2:
            raise ValueError(f"Columns {list(self.columns)} have fewer than 2 observed rows")
        self._floor = self._data_floor(x)
        self.means = _kmeanspp(x, self.n_groups, rng)
        if self.full:
            cov = np.atleast_2d(np.cov(x, rowvar=False)) + np.diag(self._floor)
            self.covs = np.repeat(cov[None], self.n_groups, axis=0)
        else:
            self.covs = np.repeat(np.maximum(np.var(x, axis=0), self._floor)[None], self.n_groups, axis=0)

    def _weighted_cov(self, x: np.ndarray, w: np.ndarray, mean: np.ndarray, norm: float) -> np.ndarray:
        diff = x - mean
        if self.full:
            cov = (diff * w[:, None]).T @ diff / norm
            return cov + np.diag(self._floor)
        return np.maximum((w[:, None] * diff ** 2).sum(axis=0) / norm, self._floor)

    def _cov_params(self) -> int:
        d = self.dim
        return d * (d + 1) // 2 if self.full else d

    def config_signature(self) -> tuple:
        extra = ()
        if hasattr(self, "learn_dof"):
            # a fixed dof is configuration, not a learnt parameter: a warm start must not override it
            extra = (self.learn_dof, self.dof_bounds) + (() if self.learn_dof else (self.dof_init,))
        return self.signature() + (self.covariance_type.value,) + extra

    def set_params(self, params: Mapping[str, np.ndarray]) -> None:
        super().set_params(params)
        if self._floor is None:
            self._floor = np.full(self.dim, 1e-12)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(columns={list(self.columns)}, covariance_type={self.covariance_type.value}, "
                f"tie_level={self.tie_level})")


# ======================================================================================================================
# Gaussian
# ======================================================================================================================
class GaussianEmission(_LocationScaleBase):
    """x | group g ~ N(means_g, covs_g) over a block of columns (diagonal or full covariance).

    ``var_floor`` is relative: the variance floor of each column is var_floor times the column's sample variance
    (a full covariance gets it added to its diagonal).
    """

    param_names = ("means", "covs")

    def __init__(self, columns: Union[int, Sequence[int]], covariance_type: CovarianceType = CovarianceType.DIAG,
                 tie_level: Optional[int] = None, var_floor: float = 1e-6):
        super().__init__(columns, covariance_type, tie_level, var_floor)

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        self._init_location_scale(X, rng)

    def log_density(self, X: np.ndarray) -> np.ndarray:
        x, ok = self._block(X)
        out = np.zeros((len(x), self.n_groups))
        xo = x[ok]
        for g in range(self.n_groups):
            d2, log_det = _mahalanobis(xo, self.means[g], self.covs[g], self.full)
            out[ok, g] = -0.5 * (self.dim * _LOG_2PI + log_det + d2)
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        x, ok = self._block(X)
        x, gamma = x[ok], gamma[ok]
        for g in range(self.n_groups):
            w = gamma[:, g]
            total = w.sum()
            if total <= 1e-12:
                continue
            mean = w @ x / total
            self.means[g] = mean
            self.covs[g] = self._weighted_cov(x, w, mean, total)

    def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((len(groups), self.dim))
        for g in range(self.n_groups):
            idx = np.flatnonzero(groups == g)
            if idx.size == 0:
                continue
            cov = self.covs[g] if self.full else np.diag(self.covs[g])
            out[idx] = rng.multivariate_normal(self.means[g], cov, size=idx.size)
        return out

    def n_free_params(self) -> int:
        return self.n_groups * (self.dim + self._cov_params())


# ======================================================================================================================
# Student-t
# ======================================================================================================================
class StudentTEmission(_LocationScaleBase):
    """x | group g ~ multivariate t(means_g, covs_g, dofs_g): ``covs`` is the scale matrix, ``dofs`` the degrees of
    freedom.

    The M-step is one ECM pass (McLachlan & Peel 2000, ch. 7): with weights u = (nu + d) / (nu + delta^2) at the
    current parameters, the location and scale have closed forms and nu solves a 1-D equation. Each pass raises the
    posterior-weighted likelihood, so the HMM's EM stays monotone. ``learn_dof=False`` holds nu at ``dof_init``.
    """

    param_names = ("means", "covs", "dofs")

    def __init__(self, columns: Union[int, Sequence[int]], covariance_type: CovarianceType = CovarianceType.DIAG,
                 tie_level: Optional[int] = None, dof_init: float = 5.0, learn_dof: bool = True,
                 dof_bounds: tuple[float, float] = (2.1, 200.0), var_floor: float = 1e-6):
        super().__init__(columns, covariance_type, tie_level, var_floor)
        lo, hi = float(dof_bounds[0]), float(dof_bounds[1])
        if not 0 < lo < hi:
            raise ValueError(f"dof_bounds must satisfy 0 < low < high, got {dof_bounds}")
        if not lo <= dof_init <= hi:
            raise ValueError(f"dof_init {dof_init} must lie within dof_bounds {dof_bounds}")
        self.dof_init, self.learn_dof, self.dof_bounds = float(dof_init), bool(learn_dof), (lo, hi)
        self.dofs: Optional[np.ndarray] = None

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        self._init_location_scale(X, rng)
        self.dofs = np.full(self.n_groups, self.dof_init)

    def _log_t(self, d2: np.ndarray, log_det: float, nu: float) -> np.ndarray:
        d = self.dim
        return (gammaln(0.5 * (nu + d)) - gammaln(0.5 * nu) - 0.5 * d * math.log(nu * math.pi) - 0.5 * log_det
                - 0.5 * (nu + d) * np.log1p(d2 / nu))

    def log_density(self, X: np.ndarray) -> np.ndarray:
        x, ok = self._block(X)
        out = np.zeros((len(x), self.n_groups))
        xo = x[ok]
        for g in range(self.n_groups):
            d2, log_det = _mahalanobis(xo, self.means[g], self.covs[g], self.full)
            out[ok, g] = self._log_t(d2, log_det, float(self.dofs[g]))
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        x, ok = self._block(X)
        x, gamma = x[ok], gamma[ok]
        d = self.dim
        for g in range(self.n_groups):
            w = gamma[:, g]
            total = w.sum()
            if total <= 1e-12:
                continue
            nu = float(self.dofs[g])
            d2, _ = _mahalanobis(x, self.means[g], self.covs[g], self.full)
            u = (nu + d) / (nu + d2)
            wu = w * u
            mean = wu @ x / wu.sum()
            self.means[g] = mean
            self.covs[g] = self._weighted_cov(x, wu, mean, total)
            if self.learn_dof:
                const = 1.0 + float((w * (np.log(u) - u)).sum() / total) + digamma(0.5 * (nu + d)) \
                    - math.log(0.5 * (nu + d))
                self.dofs[g] = self._solve_dof(const)

    def _solve_dof(self, const: float) -> float:
        lo, hi = self.dof_bounds

        def f(nu: float) -> float:
            return -digamma(0.5 * nu) + math.log(0.5 * nu) + const

        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:                  # no root inside the bounds: f is decreasing, so take the nearer bound
            return lo if f_lo < 0 else hi
        return float(brentq(f, lo, hi, xtol=1e-8))

    def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((len(groups), self.dim))
        for g in range(self.n_groups):
            idx = np.flatnonzero(groups == g)
            if idx.size == 0:
                continue
            cov = self.covs[g] if self.full else np.diag(self.covs[g])
            z = rng.multivariate_normal(np.zeros(self.dim), cov, size=idx.size)
            scale = np.sqrt(self.dofs[g] / rng.chisquare(self.dofs[g], size=idx.size))
            out[idx] = self.means[g] + z * scale[:, None]
        return out

    def n_free_params(self) -> int:
        return self.n_groups * (self.dim + self._cov_params() + (1 if self.learn_dof else 0))
