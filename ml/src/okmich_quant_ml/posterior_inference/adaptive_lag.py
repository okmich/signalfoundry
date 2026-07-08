"""Adaptive-lag inference for HMM posterior trajectories.

Per-bar lag selector. Instead of committing the regime label at a fixed lag K for every bar, the
inferer commits at the earliest lag where a stability criterion fires; bars that never satisfy
the criterion fall back to ``k_max`` (oracle-equivalent under the fixed-lag smoother).

Default config — ``theta_kl=0.005``, ``criterion=KL_PLUS_ENTROPY``, ``m_consec=2`` — was validated
across four asset classes (BTCUSDT, XAUUSD, US500, GBPJPY) on per-axis HMMs: criterion beats
fixed-lag baseline at the same mean commit lag by roughly 4 percentage points of miss rate, with
the KL+Δentropy gate cutting miss roughly 6–7× vs KL-only at similar latency. See
``IDEA_adaptive_lag_maturation.md`` and the lab notebook
``adaptive_lag_conditional_miss_rate_001.ipynb`` for evidence; the spec carries a stale-numbers
caveat regarding the specific operating points (lab numbers pre-date the ``stable[0] = False``
semantics fix and will shift toward higher mean lag under the corrected criterion).

Caveats:

- **Not a calibration fix.** Assumes upstream HMM produces honest filtering posteriors. On mixed-axis
  or trap-feature HMMs (e.g. trend+ATR), the criterion fires on confidently-wrong transitions. Use
  only with screener-validated per-axis HMMs.
- **Disagreement policy is implicit LOCK.** Early commit is final. Revisit-at-K and gated-revisit
  policies are deferred (see ``IDEA_adaptive_lag_maturation.md`` caution #2).
- **Backtest accounting requires per-bar maturation alignment.** ``infer(trajectories)`` returns the
  richer per-bar detail (``commit_lag``, ``fired``, raw ``as_of_labels()``, diagnostic ``metadata``).
  Use the ``transform(trajectories)`` convenience instead when only the causally aligned ``(T, K)``
  posterior matrix is needed (``AdaptiveLagResult.as_of_posterior()`` under the hood), safe to feed
  into downstream gate inferers / calibration / EMA. ``MaturationAlignTransformer``'s constant-K
  shift is a different (simpler, non-adaptive) contract — do not compose the two over the same
  trajectory; pick one aligner per pipeline.

Usage
-----
Primary entry point — ``infer()`` returns the full per-bar detail (commit lag, fired flag, raw
labels)::

    from okmich_quant_ml.posterior_inference import AdaptiveLagInferer, compute_trajectories

    trajectories = compute_trajectories(hmm, X, k_max=7, window_size=48)
    result = AdaptiveLagInferer(theta_kl=0.005).infer(trajectories)
    # result.commit_lag (T,)        — per-bar lag at which the label was committed
    # result.regime_label (T,)      — argmax posterior at the committed lag
    # result.fired (T,)             — True iff criterion fired below k_max

For downstream consumers that only need the causally aligned ``(T, K)`` posterior matrix (composes
with the rest of ``posterior_inference``, e.g. inside a ``PosteriorPipeline``)::

    aligned = AdaptiveLagInferer(theta_kl=0.005).transform(trajectories)
    # aligned: (T, K) causally aligned posterior matrix, ready for EMA / calibration / gate inferers.
    # Equivalent to infer(trajectories).as_of_posterior().

For a diagnostic audit (matches the lab notebook's pass/fail metrics)::

    audit = lag_commitment_audit(trajectories, result)
    print(f"mean_lag={audit['mean_lag']:.2f}  uncond_miss={audit['uncond_miss']:.4f}")
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr, xlogy

from .features import _require_integral


class StabilityCriterion(StrEnum):
    """Per-bar stability rule selecting the commit lag.

    ``KL_PLUS_ENTROPY`` is the lab-validated default — it adds a ``entropy_l <= entropy_{l-1}`` gate
    on top of KL stability to reject bars where future evidence is starting to un-sharpen the
    posterior (transition-bar signal). On four lab symbols the entropy gate cut miss rate 6–7×
    vs ``KL_ONLY`` at similar mean lag.
    """
    KL_ONLY = "kl_only"
    KL_PLUS_ENTROPY = "kl_plus_entropy"
    MAP_PERSISTENCE = "map_persistence"


@dataclass(frozen=True)
class AdaptiveLagResult:
    """Per-bar output of ``AdaptiveLagInferer.infer()``.

    **Causal-leakage warning.** ``regime_label[t]`` is the label *for* bar t but it is only
    *available* once enough future evidence has accumulated — specifically at bar
    ``t + commit_lag[t]`` (see ``available_at``). Using ``regime_label`` or ``committed_posterior``
    directly at decision time leaks future information. For causal alignment use
    ``as_of_labels()`` (labels) or ``as_of_posterior()`` (full ``(T, K)`` posterior matrix) —
    equivalently, call ``AdaptiveLagInferer.transform()`` which returns ``as_of_posterior()``
    directly.

    Attributes
    ----------
    commit_lag : NDArray
        Shape ``(T,)``, dtype ``int64``. Per-bar lag at which the regime label was committed.
        Bars where the criterion never fires hold ``k_max``.
    regime_label : NDArray
        Shape ``(T,)``, dtype ``int64``. ``argmax(trajectories[commit_lag[t], t])`` — the label
        *for* bar t. Not causally available at bar t; see warning above.
    fired : NDArray
        Shape ``(T,)``, dtype ``bool``. ``True`` iff the criterion fired at ``commit_lag < k_max``.
    committed_posterior : NDArray
        Shape ``(T, K)``, dtype ``float``. The posterior at the committed lag per bar:
        ``trajectories[commit_lag[t], t, :]``. Stability only checks that the posterior has
        stopped updating — a uniform-but-stable posterior commits with an arbitrary argmax. Pass
        this through ``EntropyGateInferer`` or ``MarginGateInferer`` for downstream confidence
        gating if labels drive strategy state.
    metadata : dict
        Diagnostic counters: ``criterion``, ``theta_kl``, ``m_consec``, ``k_max``, ``n_bars``,
        ``n_fired``, ``fired_rate``, ``mean_lag``, ``median_lag``.
    """
    commit_lag: NDArray
    regime_label: NDArray
    fired: NDArray
    committed_posterior: NDArray
    metadata: dict

    @property
    def available_at(self) -> NDArray:
        """Per-bar arrival time: ``available_at[t] = t + commit_lag[t]``.

        The bar index at which ``regime_label[t]`` becomes causally known. Use this to align
        labels for backtest / live trading consumers.
        """
        return np.arange(len(self.commit_lag), dtype=np.int64) + self.commit_lag.astype(np.int64)

    def as_of_labels(self, abstain_label: int = -1) -> NDArray:
        """Causally aligned label series.

        Returns shape ``(T,)``, dtype ``int64``. ``output[t]`` is ``regime_label[s]`` for the
        latest target bar ``s`` whose label has become available by time t — i.e. the largest
        ``s`` with ``available_at[s] <= t``. Bars before any label has arrived receive
        ``abstain_label``.

        This is the view backtest / live consumers should use. Indexing ``regime_label`` directly
        at decision time t is a causal-leakage trap; this helper resolves it deterministically.
        """
        T = len(self.commit_lag)
        sorted_avail, max_s_at_arrival = self._as_of_index()
        idx = np.searchsorted(sorted_avail, np.arange(T), side="right") - 1
        out = np.full(T, abstain_label, dtype=np.int64)
        valid = idx >= 0
        out[valid] = self.regime_label[max_s_at_arrival[idx[valid]]]
        return out

    def as_of_posterior(self) -> NDArray:
        """Causally aligned posterior series.

        Returns shape ``(T, K)``, dtype ``float``. ``output[t]`` is ``committed_posterior[s]`` for
        the latest target bar ``s`` whose label has become available by time t — same selection
        rule as ``as_of_labels`` but emitting the full posterior row instead of the argmax label.
        Rows before any label has arrived receive the uniform prior ``[1/K, ..., 1/K]``.

        This is the view downstream ``(T, K)`` posterior consumers should use — gate inferers,
        calibration transformers, EMA, anything indexed by decision time. Without it, joining
        ``committed_posterior`` directly to a returns series leaks 1..k_max bars of future
        evidence per row, concentrated on the regime-transition bars where edge lives. See
        ``MaturationAlignTransformer`` for the constant-lag analogue. ``AdaptiveLagInferer.transform()``
        returns exactly this array.

        Warmup uses uniform ``1/K`` (not NaN) for parity with ``MaturationAlignTransformer`` and
        to keep composition with the standard gate inferers — every gate in this package
        validates and rejects NaN at entry. Aggregate metrics over the full output include
        ``1/K`` placeholder mass for the warmup region; slice off the first
        ``result.available_at.min()`` rows (the index of the first matured bar — not
        ``commit_lag[0]``, which is only correct under constant lag) or use NaN-aware
        aggregation upstream if that contaminates a diagnostic.
        """
        T, K = self.committed_posterior.shape
        sorted_avail, max_s_at_arrival = self._as_of_index()
        idx = np.searchsorted(sorted_avail, np.arange(T), side="right") - 1
        out = np.full((T, K), 1.0 / K, dtype=float)
        valid = idx >= 0
        out[valid] = self.committed_posterior[max_s_at_arrival[idx[valid]]]
        return out

    def _as_of_index(self) -> tuple[NDArray, NDArray]:
        """Shared index for ``as_of_labels`` and ``as_of_posterior``.

        Returns ``(sorted_avail, max_s_at_arrival)``: arrival times sorted ascending, and the
        running max of the source bar ``s`` whose label has arrived by each sorted position.
        ``np.maximum.accumulate(order)`` resolves the "most recent target bar" question when
        late-arriving low-s labels coexist with already-arrived higher-s ones.
        """
        avail = self.available_at
        order = np.argsort(avail, kind="stable")
        sorted_avail = avail[order]
        max_s_at_arrival = np.maximum.accumulate(order.astype(np.int64))
        return sorted_avail, max_s_at_arrival


def _validate_trajectories(trajectories: NDArray) -> NDArray:
    """Validate a ``(n_lags, T, K)`` trajectory array; return as ``float`` ndarray.

    Rejects shapes other than 3D, ``n_lags < 2``, ``K < 2``, ``T == 0``, NaN/Inf, negative mass
    below ``-1e-9`` (treated as upstream corruption), rows with non-positive sums, and rows whose
    sums deviate from 1.0 by more than ``1e-3``. Tiny floating-point negatives within tolerance
    are clipped to 0 so downstream log-based math (KL, entropy) is safe.
    """
    P = np.asarray(trajectories, dtype=float)
    if P.ndim != 3:
        raise ValueError(f"trajectories must have shape (n_lags, T, K), got ndim={P.ndim}")
    n_lags, T, K = P.shape
    if n_lags < 2:
        raise ValueError(
            f"trajectories must have n_lags >= 2 (lag 0 plus at least one extra), got n_lags={n_lags}"
        )
    if K < 2:
        raise ValueError(f"trajectories must have K >= 2 states, got K={K}")
    if T == 0:
        raise ValueError("trajectories must have T >= 1 bars, got T=0")
    if not np.isfinite(P.sum()):
        raise ValueError("trajectories contain NaN or Inf values.")

    negativity_tol = 1e-9
    if P.min() < -negativity_tol:
        raise ValueError(
            f"trajectories contain negative values below -{negativity_tol} (min={P.min():.3e}); "
            f"this indicates upstream model corruption, not floating-point drift."
        )

    row_sums = P.sum(axis=2)
    if row_sums.min() <= 0.0:
        raise ValueError(
            f"trajectories contain rows with non-positive sum (min={row_sums.min():.3e}); "
            f"posterior rows must sum to a positive value."
        )
    row_sum_tol = 1e-3
    max_dev = float(np.abs(row_sums - 1.0).max())
    if max_dev > row_sum_tol:
        raise ValueError(
            f"trajectories contain rows whose sums deviate from 1.0 by more than {row_sum_tol} "
            f"(max deviation={max_dev:.3e}); expected valid probability distributions per (lag, bar)."
        )

    if P.min() < 0.0:
        return np.maximum(P, 0.0)
    return P


def _first_lag_where_stable(stable_grid: NDArray, m: int, k_max: int) -> NDArray:
    """Per-bar smallest ``l`` in ``[m-1, k_max]`` s.t. all of ``stable_grid[l-m+1:l+1, t]`` are True.

    Falls back to ``k_max`` per bar if never satisfied. Mirrors the lab notebook's
    ``_first_lag_where_stable`` exactly.
    """
    n_lags, T = stable_grid.shape
    rolling_ok = np.zeros_like(stable_grid)
    for l in range(m - 1, n_lags):
        rolling_ok[l] = stable_grid[l - m + 1: l + 1].all(axis=0)
    window = rolling_ok[m - 1:]
    has_true = window.any(axis=0)
    first_idx = window.argmax(axis=0) + (m - 1)
    return np.where(has_true, first_idx, k_max).astype(np.int64)


class AdaptiveLagInferer:
    """Per-bar lag selector for fixed-lag HMM posterior trajectories.

    Implements the ``PosteriorInferer`` protocol: ``infer(trajectories)`` returns the rich
    per-bar ``AdaptiveLagResult`` — ``commit_lag``, ``fired``, raw ``as_of_labels()``, diagnostic
    ``metadata`` via ``get_metadata()``. Its input is a ``(n_lags, T, K)`` trajectory tensor
    rather than a plain ``(T, K)`` posterior matrix, built via ``compute_trajectories``.

    ``transform(trajectories)`` is a convenience wrapper — equivalent to
    ``infer(trajectories).as_of_posterior()`` — returning just the causally aligned ``(T, K)``
    posterior matrix for callers that only need to compose with the rest of the package
    (``PosteriorPipeline``, gate inferers, calibration transformers, EMA). Because its input shape
    differs from the ``(T, K)`` contract every other transformer in this module expects, it must
    be the first stage in a pipeline, immediately after ``compute_trajectories``, for the same
    reason ``MaturationAlignTransformer`` must lead: everything downstream assumes a single,
    already execution-time-aligned ``(T, K)`` series.

    Parameters
    ----------
    theta_kl : float, default 0.005
        KL threshold for ``KL_ONLY`` / ``KL_PLUS_ENTROPY`` criteria. Lab-validated value across
        four asset classes. Ignored when ``criterion=MAP_PERSISTENCE``.
    criterion : StabilityCriterion | str, default ``KL_PLUS_ENTROPY``
        Per-bar stability rule.
    m_consec : int, default 2
        Minimum number of consecutive lags satisfying the stability rule before commit.
    """

    def __init__(self, theta_kl: float = 0.005,
                 criterion: StabilityCriterion | str = StabilityCriterion.KL_PLUS_ENTROPY,
                 m_consec: int = 2) -> None:
        self.theta_kl = float(theta_kl)
        self.criterion = StabilityCriterion(criterion)
        self.m_consec = _require_integral(m_consec, "m_consec")
        theta_kl_ok = np.isfinite(self.theta_kl) and self.theta_kl > 0.0
        if self.criterion != StabilityCriterion.MAP_PERSISTENCE and not theta_kl_ok:
            raise ValueError(
                f"theta_kl must be finite and > 0 for criterion={self.criterion.value}, got {self.theta_kl}"
            )
        if self.m_consec < 1:
            raise ValueError(f"m_consec must be >= 1, got {self.m_consec}")
        self._last_metadata: dict = {}

    def transform(self, trajectories: NDArray) -> NDArray:
        """Return the causally aligned ``(T, K)`` posterior matrix for ``trajectories``.

        Equivalent to ``infer(trajectories).as_of_posterior()``. This is the ``PosteriorTransformer``
        entry point — pass it directly to ``PosteriorPipeline(transformers=[AdaptiveLagInferer(), ...])``.
        """
        return self.infer(trajectories).as_of_posterior()

    def infer(self, trajectories: NDArray) -> AdaptiveLagResult:
        """Pick per-bar commit lag and emit the regime label there.

        Parameters
        ----------
        trajectories : NDArray
            Shape ``(n_lags, T, K)``. ``trajectories[l, t, :]`` is the posterior at bar ``t`` under
            lag ``l``. By convention ``trajectories[0]`` is the filtering posterior. Build via
            ``compute_trajectories(hmm, X, k_max)``.
        """
        P = _validate_trajectories(trajectories)
        n_lags, T, _ = P.shape
        k_max = n_lags - 1
        if self.m_consec > k_max:
            raise ValueError(
                f"m_consec={self.m_consec} exceeds k_max={k_max} (= n_lags - 1) — under the "
                f"stable[0]=False convention only {k_max} real stability observations exist. "
                f"Pass trajectories with at least m_consec + 1 lags."
            )

        stable_grid = self._build_stable_grid(P)
        commit_lag = _first_lag_where_stable(stable_grid, m=self.m_consec, k_max=k_max)

        rows = np.arange(T)
        committed_posterior = P[commit_lag, rows]
        regime_label = np.argmax(committed_posterior, axis=1).astype(np.int64)
        fired = commit_lag < k_max

        metadata = {
            "criterion": self.criterion.value,
            "theta_kl": self.theta_kl if self.criterion != StabilityCriterion.MAP_PERSISTENCE else None,
            "m_consec": self.m_consec,
            "k_max": int(k_max),
            "n_bars": int(T),
            "n_fired": int(fired.sum()),
            "fired_rate": float(fired.mean()),
            "mean_lag": float(commit_lag.mean()),
            "median_lag": float(np.median(commit_lag)),
        }
        self._last_metadata = metadata
        return AdaptiveLagResult(
            commit_lag=commit_lag, regime_label=regime_label, fired=fired,
            committed_posterior=committed_posterior, metadata=metadata,
        )

    def get_metadata(self) -> dict:
        return dict(self._last_metadata)

    @classmethod
    def from_hmm(cls, hmm: Any, X: NDArray, k_max: int, *, theta_kl: float = 0.005,
                 criterion: StabilityCriterion | str = StabilityCriterion.KL_PLUS_ENTROPY,
                 m_consec: int = 2, window_size: int | None = 48) -> AdaptiveLagResult:
        """Convenience: build trajectories from an HMM and return the inference result.

        Equivalent to ``cls(...).infer(compute_trajectories(hmm, X, k_max, window_size))``.
        """
        trajectories = compute_trajectories(hmm, X, k_max, window_size=window_size)
        return cls(theta_kl=theta_kl, criterion=criterion, m_consec=m_consec).infer(trajectories)

    def _build_stable_grid(self, P: NDArray) -> NDArray:
        if self.criterion == StabilityCriterion.MAP_PERSISTENCE:
            maps = np.argmax(P, axis=2)
            stable = np.zeros_like(maps, dtype=bool)
            stable[1:] = maps[1:] == maps[:-1]
            return stable

        # KL-based criteria: kl_per_bar[l, t] = KL(P_l(t) || P_{l-1}(t)).
        # Convention: stable[0] is forced False because lag 0 has no prior lag to compare against.
        # This makes ``m_consec`` semantics symmetric with MAP_PERSISTENCE and prevents m_consec=1
        # from trivially committing at lag 0.
        # We use ``scipy.special.rel_entr`` so support changes (P[l-1, j] == 0 and P[l, j] > 0)
        # produce an infinite KL — correctly rejected by the < theta_kl check — rather than a
        # finite-but-large value that an eps-clamped log would emit.
        n_lags, T, _ = P.shape
        kl_per_bar = np.zeros((n_lags, T), dtype=float)
        for l in range(1, n_lags):
            kl_per_bar[l] = rel_entr(P[l], P[l - 1]).sum(axis=1)
        below = kl_per_bar < self.theta_kl
        below[0] = False

        if self.criterion == StabilityCriterion.KL_ONLY:
            return below

        # KL_PLUS_ENTROPY: also require entropy non-increasing along the lag axis.
        # xlogy(0, 0) = 0 makes -sum(xlogy(p, p)) the correct entropy without an eps clamp.
        entropy_per_bar = -xlogy(P, P).sum(axis=2)
        non_inc = np.zeros_like(below)
        non_inc[1:] = entropy_per_bar[1:] <= entropy_per_bar[:-1]
        # non_inc[0] left False — same lag-0-undefined convention.
        return below & non_inc


def compute_trajectories(hmm: Any, X: NDArray, k_max: int, window_size: int | None = 48) -> NDArray:
    """Compute lag trajectories for ``AdaptiveLagInferer.infer()`` / ``.transform()``.

    Wraps ``hmm.predict_proba_fixed_lag_sweep(X, lags=[0..k_max], window_size=window_size)`` and
    stacks the per-lag posteriors into a single ``(n_lags, T, K)`` array. ``n_lags = k_max + 1``.

    **window_size default (48) is deliberate** and matches the production batch script
    ``generate_all_001.py``'s ``fixed_lag_window_size`` default, which is the inference mode the
    lab validation was run against. This is the **bounded-history** (finite-window) smoother, not
    full-history fixed-lag inference. Pass ``window_size=None`` to opt into full-history mode, but
    note that operating points reported in ``IDEA_adaptive_lag_maturation.md`` will not directly
    apply — re-validate before changing this default in production.
    """
    k_max = _require_integral(k_max, "k_max")
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if window_size is not None and k_max >= window_size:
        raise ValueError(
            f"k_max={k_max} must be strictly less than window_size={window_size} in "
            f"bounded-history mode; otherwise the lag-K smoother's [t+L-W+1, t+L] window does "
            f"not include the target bar t. Pass window_size=None for full-history mode, or "
            f"increase window_size."
        )
    lags = list(range(0, k_max + 1))
    lag_posteriors = hmm.predict_proba_fixed_lag_sweep(X, lags=lags, window_size=window_size)
    return np.stack([np.asarray(lag_posteriors[l], dtype=float) for l in lags], axis=0)


def lag_commitment_audit(trajectories: NDArray, result: AdaptiveLagResult,
                         dist_bins: list[tuple[int, int, str]] | None = None) -> dict:
    """Diagnostic audit replicating the lab notebook's pass/fail metrics.

    Reports:
    - ``uncond_miss``: per-bar mismatch rate ``regime_label != argmax(trajectories[k_max])``.
    - ``miss_<bin>``: conditional miss rate per distance-to-nearest-change-point bin.
    - ``change_point_rate``, ``mean_lag``, ``median_lag``, ``fired_rate``.

    Parameters
    ----------
    trajectories : NDArray
        Same ``(n_lags, T, K)`` array passed to ``inferer.infer()``.
    result : AdaptiveLagResult
        Output of ``inferer.infer(trajectories)``.
    dist_bins : list[tuple[int, int, str]] | None
        Distance bins ``(lo, hi, label)``; inclusive on both ends. Defaults to lab convention
        ``[(0,0,'at_change'), (1,2,'near'), (3,k_max,'medium'), (k_max+1,∞,'far')]``.
    """
    P = _validate_trajectories(trajectories)
    n_lags, T, _ = P.shape
    k_max = n_lags - 1

    if result.commit_lag.shape != (T,):
        raise ValueError(
            f"result.commit_lag shape {result.commit_lag.shape} does not match trajectories T={T}"
        )
    if result.regime_label.shape != (T,):
        raise ValueError(
            f"result.regime_label shape {result.regime_label.shape} does not match trajectories T={T}"
        )
    if result.fired.shape != (T,):
        raise ValueError(
            f"result.fired shape {result.fired.shape} does not match trajectories T={T}"
        )
    K = P.shape[2]
    if result.committed_posterior.shape != (T, K):
        raise ValueError(
            f"result.committed_posterior shape {result.committed_posterior.shape} does not match "
            f"trajectories (T, K)=({T}, {K}); likely a result built from different trajectories."
        )
    if T > 0:
        lag_min = int(result.commit_lag.min())
        lag_max = int(result.commit_lag.max())
        if lag_min < 0 or lag_max > k_max:
            raise ValueError(
                f"result.commit_lag values out of range [0, {k_max}] (min={lag_min}, max={lag_max}); "
                f"likely a result built from different trajectories."
            )
        expected_label = np.argmax(result.committed_posterior, axis=1).astype(np.int64)
        if not np.array_equal(result.regime_label.astype(np.int64), expected_label):
            raise ValueError(
                "result.regime_label disagrees with argmax(result.committed_posterior); the result "
                "is internally inconsistent (likely manually constructed or stale)."
            )
        expected_fired = result.commit_lag < k_max
        if not np.array_equal(result.fired.astype(bool), expected_fired):
            raise ValueError(
                "result.fired disagrees with (result.commit_lag < k_max); the result is internally "
                "inconsistent (likely manually constructed or stale)."
            )

    oracle_map = np.argmax(P[k_max], axis=1).astype(np.int64)
    miss = result.regime_label != oracle_map

    is_change = np.zeros(T, dtype=bool)
    is_change[1:] = oracle_map[1:] != oracle_map[:-1]
    sent = 10**9
    fwd = np.full(T, sent, dtype=np.int64)
    nxt = sent
    for t in range(T - 1, -1, -1):
        if is_change[t]:
            nxt = 0
        elif nxt < sent:
            nxt += 1
        fwd[t] = nxt
    bwd = np.full(T, sent, dtype=np.int64)
    prev = sent
    for t in range(T):
        if is_change[t]:
            prev = 0
        elif prev < sent:
            prev += 1
        bwd[t] = prev
    dist = np.minimum(fwd, bwd)

    if dist_bins is None:
        # ``far`` starts above the union of ``at_change`` and ``near`` (max distance 2) to avoid
        # overlap when ``k_max < 3`` (e.g. k_max=1 would otherwise put distance 2 in both
        # ``near=[1,2]`` and ``far=[k_max+1, sent]=[2, sent]``). ``medium`` collapses to empty
        # when k_max < 3.
        far_lo = max(k_max + 1, 3)
        dist_bins = [(0, 0, "at_change"), (1, 2, "near"), (3, k_max, "medium"), (far_lo, sent, "far")]

    audit: dict = {
        "n_bars": int(T),
        "k_max": int(k_max),
        "n_change_points": int(is_change.sum()),
        "change_point_rate": float(is_change.mean()),
        "mean_lag": float(result.commit_lag.mean()),
        "median_lag": float(np.median(result.commit_lag)),
        "fired_rate": float(result.fired.mean()),
        "uncond_miss": float(miss.mean()),
    }
    for lo, hi, label in dist_bins:
        mask = (dist >= lo) & (dist <= hi)
        audit[f"miss_{label}"] = float(miss[mask].mean()) if mask.any() else float("nan")
        audit[f"n_{label}"] = int(mask.sum())
    return audit
