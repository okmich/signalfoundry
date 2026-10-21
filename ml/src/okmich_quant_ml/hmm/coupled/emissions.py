"""Emission blocks of a coupled HMM. Each block reads one column of X and depends on one or two chains.

A block is bound to the model's joint-state table (J, C) before use. ``log_density`` returns (T, J); a NaN observation
contributes 0 (the feature is unobserved on that bar) and is skipped in the M-step. ``param_names`` lists the learnt
parameters: the model copies exactly these for warm starts and snapshots, never the configuration.
"""
from __future__ import annotations

import abc
import math

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


class Emission(abc.ABC):
    """One observed column whose density depends on the states of ``chains``."""

    column: int
    chains: tuple[int, ...]
    param_names: tuple[str, ...] = ()

    def bind(self, states: np.ndarray, n_states: tuple[int, ...]) -> None:
        """Attach the model's joint-state table (J, C) and states per chain."""
        if any(c < 0 or c >= len(n_states) for c in self.chains):
            raise ValueError(f"{type(self).__name__} on column {self.column} refers to chains {self.chains}, but the "
                             f"model has {len(n_states)} chains")
        self._states = states
        self._n_states = n_states

    def load_params(self, other: "Emission") -> None:
        """Copy the learnt parameters (not the configuration) from a block of the same kind."""
        for name in self.param_names:
            setattr(self, name, np.array(getattr(other, name), dtype=np.float64, copy=True))

    def get_params(self) -> dict[str, np.ndarray]:
        return {name: np.array(getattr(self, name), copy=True) for name in self.param_names}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for name, value in params.items():
            setattr(self, name, np.array(value, dtype=np.float64, copy=True))

    @abc.abstractmethod
    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        """Starting parameters for fresh start ``restart`` (0 = the deterministic one)."""

    @abc.abstractmethod
    def log_density(self, X: np.ndarray) -> np.ndarray:
        """(T, J) log density of the column under every joint state."""

    @abc.abstractmethod
    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """Update the parameters from the (T, J) joint posteriors."""

    @abc.abstractmethod
    def permute(self, chain: int, perm: np.ndarray) -> None:
        """Relabel ``chain``'s states: new state i is old state perm[i]."""

    @abc.abstractmethod
    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One draw per row of ``states`` (T, C)."""

    @abc.abstractmethod
    def n_free_params(self) -> int: ...

    def _column(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = X[:, self.column]
        return x, np.isfinite(x)

    def _marginal(self, gamma: np.ndarray, chain: int) -> np.ndarray:
        """(T, K_chain) posterior marginal of one chain from the joint posterior."""
        out = np.zeros((gamma.shape[0], self._n_states[chain]))
        for k in range(self._n_states[chain]):
            out[:, k] = gamma[:, self._states[:, chain] == k].sum(axis=1)
        return out


class GaussianEmission(Emission):
    """x | s_chain ~ N(mean_s, var_s), or one variance for every state when ``tied``."""

    param_names = ("mean", "var")

    def __init__(self, column: int, chain: int, tied: bool = False, var_floor: float = 1e-6):
        self.column, self.chain, self.tied, self.var_floor = int(column), int(chain), bool(tied), float(var_floor)
        self.chains = (self.chain,)
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        x, ok = self._column(X)
        k = self._n_states[self.chain]
        sd = float(np.std(x[ok]))
        jitter = rng.normal(0.0, 0.05 * sd, k) * (restart > 0)    # drawn on every start: one rng stream per seed
        self.mean = np.sort(np.quantile(x[ok], (np.arange(k) + 0.5) / k) + jitter)
        self.var = np.full(k, max(float(np.var(x[ok])) / (1.0 if self.tied else k), self.var_floor))

    def log_density(self, X: np.ndarray) -> np.ndarray:
        x, ok = self._column(X)
        s = self._states[:, self.chain]
        mean, var = self.mean[s], self.var[s]
        out = -0.5 * (_LOG_2PI + np.log(var) + (x[:, None] - mean) ** 2 / var)
        out[~ok] = 0.0
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        x, ok = self._column(X)
        g = self._marginal(gamma[ok], self.chain)
        x = x[ok]
        weight = np.maximum(g.sum(axis=0), 1e-12)
        self.mean = g.T @ x / weight
        sq = (g * (x[:, None] - self.mean) ** 2).sum(axis=0)
        floor = self.var_floor * max(float(np.var(x)), 1e-300)
        if self.tied:
            self.var = np.full(len(weight), max(sq.sum() / weight.sum(), floor))
        else:
            self.var = np.maximum(sq / weight, floor)

    def permute(self, chain: int, perm: np.ndarray) -> None:
        if chain == self.chain:
            self.mean, self.var = self.mean[perm], self.var[perm]

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = states[:, self.chain]
        return self.mean[s] + np.sqrt(self.var[s]) * rng.standard_normal(len(s))

    def n_free_params(self) -> int:
        k = self._n_states[self.chain]
        return k + (1 if self.tied else k)

    def __repr__(self) -> str:
        return f"GaussianEmission(column={self.column}, chain={self.chain}, tied={self.tied})"


class LocationScaleEmission(Emission):
    """x | a, b ~ N(mu_a * scale_b, scale_b^2): chain ``location`` sets the drift in units of chain ``scale``'s scale.

    The scale chain carries the variance, so the location chain can only model drift. The M-step is one coordinate
    pass (mu given scale, then scale given mu), each in closed form, so EM stays monotone. With ``fixed_mu`` the
    drifts are held and only the scales are learnt: a drift-regime filter whose states keep their meaning. (On 1-bar
    returns free drifts are not identified: the location chain collapses to one state, or, when it may drive the
    scale chain's transitions, becomes a second volatility chain.)
    """

    param_names = ("mu", "scale")

    def __init__(self, column: int, location: int, scale: int, init_spread: tuple[float, ...] = (0.1, 0.05, 0.2, 0.3),
                 fixed_mu: np.ndarray | None = None):
        if int(location) == int(scale):
            raise ValueError("location and scale must be different chains")
        self.column, self.location, self.scale_chain = int(column), int(location), int(scale)
        self.chains = (self.location, self.scale_chain)
        self.init_spread = tuple(init_spread)
        self.fixed_mu = None if fixed_mu is None else np.asarray(fixed_mu, dtype=np.float64).copy()
        self.mu: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def bind(self, states: np.ndarray, n_states: tuple[int, ...]) -> None:
        super().bind(states, n_states)
        if self.fixed_mu is not None and self.fixed_mu.shape != (n_states[self.location],):
            raise ValueError(f"fixed_mu needs {n_states[self.location]} values, one per location state")

    def load_params(self, other: "Emission") -> None:
        super().load_params(other)
        if self.fixed_mu is not None:
            self.mu = self.fixed_mu.copy()

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        x, ok = self._column(X)
        k_a, k_b = self._n_states[self.location], self._n_states[self.scale_chain]
        if self.fixed_mu is not None:
            self.mu = self.fixed_mu.copy()
        else:
            spread = self.init_spread[restart % len(self.init_spread)]
            self.mu = np.linspace(-spread, spread, k_a) if k_a > 1 else np.zeros(1)
        self.scale = np.sqrt(np.quantile(x[ok] ** 2, (np.arange(k_b) + 0.5) / k_b)) * 1.25
        self.scale = np.maximum(self.scale, 1e-6 * max(float(np.std(x[ok])), 1e-300))

    def log_density(self, X: np.ndarray) -> np.ndarray:
        x, ok = self._column(X)
        a, b = self._states[:, self.location], self._states[:, self.scale_chain]
        sc = self.scale[b]
        z = x[:, None] / sc - self.mu[a]
        out = -0.5 * (_LOG_2PI + z ** 2) - np.log(sc)
        out[~ok] = 0.0
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        x, ok = self._column(X)
        x, gamma = x[ok], gamma[ok]
        a, b = self._states[:, self.location], self._states[:, self.scale_chain]
        k_a, k_b = self._n_states[self.location], self._n_states[self.scale_chain]
        g = np.zeros((len(x), k_a, k_b))
        for j in range(gamma.shape[1]):
            g[:, a[j], b[j]] += gamma[:, j]
        if self.fixed_mu is None:             # mu given scale
            num = np.einsum("tab,t,b->a", g, x, 1.0 / self.scale)
            self.mu = num / np.maximum(g.sum(axis=(0, 2)), 1e-12)
        # scale given mu: the positive root of n s^2 + B s - A = 0
        n = g.sum(axis=(0, 1))
        big_a = np.einsum("tab,t->b", g, x ** 2)
        big_b = np.einsum("tab,t,a->b", g, x, self.mu)
        root = (-big_b + np.sqrt(big_b ** 2 + 4.0 * n * big_a)) / np.maximum(2.0 * n, 1e-12)
        self.scale = np.where(n > 1e-9, np.maximum(root, 1e-6 * max(float(np.std(x)), 1e-300)), self.scale)

    def permute(self, chain: int, perm: np.ndarray) -> None:
        if chain == self.location:
            self.mu = self.mu[perm]
            if self.fixed_mu is not None:
                self.fixed_mu = self.fixed_mu[perm]
        if chain == self.scale_chain:
            self.scale = self.scale[perm]

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a, b = states[:, self.location], states[:, self.scale_chain]
        return self.scale[b] * (self.mu[a] + rng.standard_normal(len(a)))

    def n_free_params(self) -> int:
        return (self._n_states[self.location] if self.fixed_mu is None else 0) + self._n_states[self.scale_chain]

    def __repr__(self) -> str:
        fixed = ", fixed_mu" if self.fixed_mu is not None else ""
        return f"LocationScaleEmission(column={self.column}, location={self.location}, scale={self.scale_chain}{fixed})"


class ClampedEmission(Emission):
    """Clamps ``chain`` to an observed state: column ``column`` holds the state index (NaN = unobserved).

    With a clamped chain as a parent of another, the model is the input-output HMM's one-way coupling with a one-hot
    input: per-input-value transition tables, fitted by closed-form EM.
    """

    def __init__(self, column: int, chain: int):
        self.column, self.chain = int(column), int(chain)
        self.chains = (self.chain,)

    def init(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        x, ok = self._column(X)
        k = self._n_states[self.chain]
        if not np.all((x[ok] == np.round(x[ok])) & (x[ok] >= 0) & (x[ok] < k)):
            raise ValueError(f"column {self.column} must hold state indices 0..{k - 1} of chain {self.chain} (or NaN)")

    def log_density(self, X: np.ndarray) -> np.ndarray:
        x, ok = self._column(X)
        s = self._states[:, self.chain]
        out = np.where(x[:, None] == s[None, :], 0.0, -np.inf)
        out[~ok] = 0.0
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        pass

    def permute(self, chain: int, perm: np.ndarray) -> None:
        if chain == self.chain:
            raise ValueError("A clamped chain's states are fixed by the data; relabel the input column instead.")

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return states[:, self.chain].astype(np.float64)

    def n_free_params(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"ClampedEmission(column={self.column}, chain={self.chain})"
