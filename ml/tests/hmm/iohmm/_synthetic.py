"""Ground truth of the synthetic data the IOHMM tests share.

- A clock-driven 2-state IOHMM and a homogeneous 2-state HMM (one feature).
- A 3-state IOHMM with 3 features and 2 persistent inputs known at bar close: one drives the transitions, one is
  irrelevant.
- Two-axis worlds: a path-structure axis and a volatility axis that interact in one of three ways (``Interaction``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

from okmich_quant_ml.hmm import DistType, PomegranateIOHMM

FIT_KW = dict(random_state=7, max_iter=300, covariance_type="diag")


def simulate_gaussian_iohmm(U: np.ndarray, b: np.ndarray, W: np.ndarray, means: np.ndarray, stds: np.ndarray,
                            pi: Optional[np.ndarray] = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw states and Gaussian outputs from an IOHMM: ``P(z_t = j | z_{t-1} = i) = softmax_j(b[i, j] + W[i, j] . U[t])``.

    ``U`` (T, d) are effective inputs in raw units (row ``t`` drives the transition into ``t``); ``b`` (K, K) and
    ``W`` (K, K, d) use the model's convention (``b[i, i] = W[i, i] = 0``). ``means`` and ``stds`` are (K,) for one
    feature or (K, D) for D independent features. Returns ``X`` (T, D) and the states (T,).
    """
    U = np.asarray(U, dtype=np.float64).reshape(len(U), -1)
    means, stds = np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)
    if means.ndim == 1:
        means, stds = means[:, None], stds[:, None]
    T, K = len(U), len(means)
    rng = np.random.default_rng(seed)
    pi = np.full(K, 1.0 / K) if pi is None else np.asarray(pi, dtype=np.float64)
    draws = rng.random(T)
    states = np.empty(T, dtype=np.int64)
    states[0] = min(int(np.searchsorted(np.cumsum(pi), draws[0])), K - 1)
    for t in range(1, T):
        p = softmax(b[states[t - 1]] + W[states[t - 1]] @ U[t])
        states[t] = min(int(np.searchsorted(np.cumsum(p), draws[t])), K - 1)
    X = rng.normal(means[states], stds[states])
    return X, states


# ----------------------------------------------------------------------------------------------------------------------
# 2 states, 1 feature, a clock input
# ----------------------------------------------------------------------------------------------------------------------
CLOCK_PERIOD = 96  # bars per "day": a 15m clock
TRUE_B = np.array([[0.0, -4.0], [-3.0, 0.0]])
TRUE_W = np.zeros((2, 2, 1))
TRUE_W[0, 1, 0], TRUE_W[1, 0, 0] = 2.0, 1.5  # the clock raises the hazard of BOTH moves near its peak
TRUE_STDS = np.array([0.6, 1.5])  # a calm and a turbulent state, both zero-mean


def clock_input(n_bars: int) -> np.ndarray:
    return np.sin(2 * np.pi * np.arange(n_bars) / CLOCK_PERIOD).reshape(-1, 1)


def clock_sequence(n_bars: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = clock_input(n_bars)
    X, states = simulate_gaussian_iohmm(U, TRUE_B, TRUE_W, means=[0.0, 0.0], stds=TRUE_STDS, seed=seed)
    return X, U, states


def homogeneous_sequence(n_bars: int, seed: int) -> np.ndarray:
    b = np.array([[0.0, -3.5], [-3.0, 0.0]])
    X, _ = simulate_gaussian_iohmm(np.zeros((n_bars, 1)), b, np.zeros((2, 2, 1)), means=[-0.5, 0.8], stds=[0.6, 1.4],
                                   seed=seed)
    return X


# ----------------------------------------------------------------------------------------------------------------------
# 3 states, 3 features, 2 inputs known at bar close
# ----------------------------------------------------------------------------------------------------------------------
K3_MEANS = np.array([[-1.0, 0.5, 0.0], [1.0, 0.0, -0.5], [0.0, -1.0, 1.0]])
K3_STD = 0.8
K3_B = np.full((3, 3), np.log(0.02 / 0.96))
np.fill_diagonal(K3_B, 0.0)
K3_W = np.zeros((3, 3, 2))
K3_W[:, :, 0] = 0.8  # input 0 raises the hazard of every move ...
K3_W[2, 0, 0] = -0.6  # ... except 2 -> 0, which it lowers
K3_W[np.arange(3), np.arange(3), :] = 0.0  # input 1 is irrelevant
K3_SEQUENCES, K3_SEQ_LEN = 10, 3_000


def persistent_inputs(n_bars: int, n_inputs: int, seed: int, persistence: float = 0.95) -> np.ndarray:
    """Unit-variance AR(1) columns."""
    rng = np.random.default_rng(seed)
    u = np.empty((n_bars, n_inputs))
    u[0] = rng.normal(size=n_inputs)
    for t in range(1, n_bars):
        u[t] = persistence * u[t - 1] + np.sqrt(1 - persistence ** 2) * rng.normal(size=n_inputs)
    return u


def three_state_sequences(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """``X`` (T, 3), RAW inputs ``U`` (T, 2) known at bar close, states, lengths. Row t-1's inputs drive t-1 -> t."""
    n_bars = K3_SEQUENCES * K3_SEQ_LEN
    U = persistent_inputs(n_bars, 2, seed)
    effective = np.vstack([U[:1], U[:-1]])  # AT_BAR_CLOSE: the transition into t uses the input of t - 1
    X, states = simulate_gaussian_iohmm(effective, K3_B, K3_W, K3_MEANS, np.full_like(K3_MEANS, K3_STD), seed=seed + 1)
    return X, U, states, [K3_SEQ_LEN] * K3_SEQUENCES


def match_states(fitted_means: np.ndarray, true_means: np.ndarray) -> np.ndarray:
    """``perm[k]`` = the fitted state that plays true state k (minimum total squared distance between means)."""
    rows, cols = linear_sum_assignment(((fitted_means[:, None, :] - true_means[None]) ** 2).sum(axis=-1))
    return rows[np.argsort(cols)]


# ----------------------------------------------------------------------------------------------------------------------
# Two-axis worlds
# ----------------------------------------------------------------------------------------------------------------------
class Interaction(StrEnum):
    """How the volatility axis relates to the path-structure axis in :func:`simulate_two_axis`."""

    INDEPENDENT = "independent"  # two separate chains: volatility says nothing about path regimes
    TIMING = "timing"            # high volatility on the previous bar raises the hazard of leaving every path regime
    LEVEL = "level"              # no separate volatility chain: each path regime has its own return volatility


@dataclass(frozen=True)
class TwoAxisWorld:
    """
    Parameters of :func:`simulate_two_axis`.

    The path axis has ``K`` regimes seen only through weak features (``path_means`` with unit-variance noise, so one
    bar says little). The volatility axis is seen through returns ``r ~ N(0, sigma)``.

    - INDEPENDENT and TIMING: a 2-state volatility chain (calm / high) sets ``sigma`` from ``vol_sigma``.
    - TIMING: when the previous bar is high-volatility, every "leave" logit of the path chain rises by
      ``timing_log_odds`` (the hazard rises in every direction).
    - LEVEL: ``sigma`` comes from ``level_sigma[path regime]``; the volatility "regime" is the high tier.
    """

    path_stay: float = 0.97
    path_means: np.ndarray = field(default_factory=lambda: np.array([[-0.5, 0.5, 0.0], [0.5, 0.0, -0.5],
                                                                     [0.0, -0.5, 0.5]]))
    vol_stay: tuple[float, float] = (0.99, 0.97)
    vol_sigma: tuple[float, float] = (0.6, 1.5)
    timing_log_odds: float = 2.0
    level_sigma: tuple[float, ...] = (0.6, 0.6, 1.5)

    @property
    def n_path_states(self) -> int:
        return len(self.path_means)

    def path_logits(self) -> np.ndarray:
        """Baseline transition logits ``b`` of the path chain: leave probability split evenly, stay as reference."""
        K = self.n_path_states
        b = np.full((K, K), np.log((1.0 - self.path_stay) / (K - 1) / self.path_stay))
        np.fill_diagonal(b, 0.0)
        return b


@dataclass(frozen=True)
class TwoAxisSample:
    """``X`` (T, D) path features; ``r`` (T,) returns; true ``path`` and ``vol`` regimes (T,)."""

    X: np.ndarray
    r: np.ndarray
    path: np.ndarray
    vol: np.ndarray


def simulate_two_axis(interaction: Interaction, n_bars: int, world: TwoAxisWorld = TwoAxisWorld(),
                      seed: int = 0) -> TwoAxisSample:
    """Draw one continuous sequence of the two-axis world. See :class:`TwoAxisWorld` for the three interactions."""
    interaction = Interaction(interaction)
    rng = np.random.default_rng(seed)
    K = world.n_path_states
    b = world.path_logits()
    path = np.empty(n_bars, dtype=np.int64)
    vol = np.zeros(n_bars, dtype=np.int64)
    path[0] = rng.integers(K)
    if interaction is not Interaction.LEVEL:
        stay = np.asarray(world.vol_stay)
        vol[0] = int(rng.random() < (1 - stay[0]) / (2 - stay[0] - stay[1]))  # stationary share of the high state
        for t in range(1, n_bars):
            vol[t] = vol[t - 1] if rng.random() < stay[vol[t - 1]] else 1 - vol[t - 1]
    for t in range(1, n_bars):
        logits = b[path[t - 1]].copy()
        if interaction is Interaction.TIMING and vol[t - 1] == 1:
            logits[np.arange(K) != path[t - 1]] += world.timing_log_odds
        path[t] = rng.choice(K, p=softmax(logits))
    if interaction is Interaction.LEVEL:
        sigma = np.asarray(world.level_sigma, dtype=np.float64)[path]
        vol = (sigma >= max(world.level_sigma)).astype(np.int64)
    else:
        sigma = np.asarray(world.vol_sigma, dtype=np.float64)[vol]
    X = world.path_means[path] + rng.normal(size=(n_bars, world.path_means.shape[1]))
    r = rng.normal(size=n_bars) * sigma
    return TwoAxisSample(X=X, r=r, path=path, vol=vol)


WORLD_BARS, WORLD_DAY = 40_000, 96
WORLD_FIT_KW = dict(random_state=7, max_iter=300, covariance_type="diag", n_restarts=3)


def two_axis_data(interaction: Interaction, seed: int = 1) -> tuple[TwoAxisSample, np.ndarray, np.ndarray, np.ndarray]:
    """
    One two-axis world with the input a user would build: ``vp``, the filtered P(high volatility) of a 2-state HMM on
    the returns, fitted on the first half (the training rows). It is known at the close of its bar. Returns the
    sample, ``vp`` (T,), the training mask and the day ids (the sequences).
    """
    sample = simulate_two_axis(interaction, WORLD_BARS, seed=seed)
    train = np.arange(WORLD_BARS) < WORLD_BARS // 2
    vol = PomegranateIOHMM(DistType.NORMAL, 2, **WORLD_FIT_KW).fit(sample.r[train].reshape(-1, 1))
    high = int(np.argmax([float(np.ravel(p["covs"])[0]) for p in vol.parameters]))
    vp = vol.predict_proba(sample.r.reshape(-1, 1))[:, high]
    return sample, vp, train, np.arange(WORLD_BARS) // WORLD_DAY
