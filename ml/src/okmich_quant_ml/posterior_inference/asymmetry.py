"""Posterior asymmetry — economic profiling of HMM filtering posteriors by state.

This module turns a causal posterior matrix ``(T, K)`` plus forward-measured outcome series into a per-(axis, state)
profile whose cross-state contrast *is* the asymmetry signal. It is the offline ``audit`` core of the asymmetry workstream
(``forward_outcome_by_state``); the live exploitation of any asymmetry it surfaces is built separately and only after a
finding survives walk-forward validation.

**Causality contract.** Every row of ``P`` must already be a causal/decision-time posterior (pure filtering, or an
``as_of``-aligned matured posterior). The outcome series are forward-measured: ``values[t]`` is the outcome realised over
``(t, t + horizon]`` known only in hindsight — they are audit targets, never live inputs. This function must not be called
inside a live signal path.

**Overlap correction.** Horizon-``h`` forward outcomes at consecutive bars share ``h - 1`` bars, so a naive ``mean/std``
t-stat is invalid. Significance of the per-state contrast uses a Bartlett-kernel HAC (Newey–West) long-run variance with
bandwidth ``h - 1`` (``bartlett_hac_variance``); ``n_eff = coverage / h`` is reported as a rough independent-sample count.

**Axis-agnostic.** The caller supplies a dict of named ``ForwardOutcome`` series — one per market axis (trend, momentum,
volatility, path-structure, ...). This module does not know what an axis "means"; it profiles state-conditional
distributions of whatever forward series it is handed. Axes are a research taxonomy, not per-axis code.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .features import _validate_posterior_matrix


class ForwardOutcome(NamedTuple):
    """A causal, forward-measured outcome series and the horizon (in bars) it spans.

    ``values`` is length-``T`` and aligned row-for-row to the posterior matrix it is profiled against: ``values[t]`` is the
    outcome realised over ``(t, t + horizon]`` as measured from bar ``t``. The trailing ``horizon`` rows (no future
    available) and any leading warm-up rows of a rolling estimator are ``NaN`` and are excluded per state.

    ``horizon`` (>= 1) is the overlap length: consecutive rows share ``horizon - 1`` bars, so it sets the Bartlett-HAC
    bandwidth (``horizon - 1``) used for overlap-corrected significance.
    """

    values: NDArray
    horizon: int


def bartlett_hac_variance(contributions: NDArray, bandwidth: int) -> float:
    """Newey–West (Bartlett-kernel) long-run variance of ``sum_t contributions[t]``.

    ``contributions`` is a contiguous-in-time 1-D series whose sum is the statistic of interest (each element already
    carries its normalising weights, so this returns the variance of the *sum* directly, not of a mean). Bartlett weights
    ``1 - l / (bandwidth + 1)`` taper the autocovariances out to ``bandwidth`` lags, correcting for the serial correlation
    that overlapping horizon-``h`` windows induce. ``bandwidth = 0`` recovers the white-noise variance ``sum_t x_t**2``.

    Returns a non-negative float (the Newey–West estimator is clamped at 0). Returns ``nan`` for an empty series.
    """
    x = np.asarray(contributions, dtype=float)
    n = x.size
    if n == 0:
        return float("nan")
    bandwidth = max(int(bandwidth), 0)
    omega = float(x @ x)  # gamma_0
    max_lag = min(bandwidth, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma_lag = float(x[lag:] @ x[:-lag])
        omega += 2.0 * weight * gamma_lag
    return max(omega, 0.0)


def _weighted_quantiles(values: NDArray, weights: NDArray, quantiles: NDArray) -> NDArray:
    """Weighted quantiles of 1-D ``values`` under non-negative ``weights``; ``quantiles`` in ``[0, 1]``.

    Uses the weighted inverse-CDF (lower convention): the quantile is the smallest value whose cumulative weight reaches
    ``q * total``. This is robust to zero / near-zero-weight observations — which a linear-interpolation convention would
    let anchor the result through a value the state owns no mass at (e.g. ``values=[0, 100], weights=[0, 1]`` must return
    ``100``, not an interpolated ``50``). Returns the actual values the state had mass at; ``nan`` when total weight is
    non-positive.
    """
    if values.size == 0:
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values, kind="stable")
    v = values[order]
    cum = np.cumsum(weights[order])
    total = cum[-1]
    if total <= 0:
        return np.full(quantiles.shape, np.nan)
    idx = np.clip(np.searchsorted(cum, np.asarray(quantiles) * total, side="left"), 0, v.size - 1)
    return v[idx]


_QUANTILE_LEVELS = np.array([0.25, 0.5, 0.75])
_COLUMNS = ["axis", "state", "state_label", "horizon", "n_valid", "n_map", "coverage", "n_eff", "w_mean", "q25", "q50",
            "q75", "pooled_mean", "delta_vs_pooled", "se_hac", "t_hac", "low_coverage"]


def forward_outcome_by_state(P: NDArray, outcomes: dict[str, ForwardOutcome], *, state_names: list[str] | None = None,
                             min_coverage: float = 100.0, row_sum_tol: float = 1e-6) -> pd.DataFrame:
    """Profile forward outcomes by HMM state; the cross-state contrast per axis is the asymmetry.

    For each axis and state ``k``, computes the posterior-mass-weighted central tendency of the forward outcome (weights
    ``w_t = P[t, k]`` over rows where the outcome is finite) and its **contrast against the unconditional pooled mean**
    (``delta_vs_pooled``), with an overlap-corrected HAC t-stat. Soft posterior weighting (not ``argmax`` conditioning) is
    used deliberately — it is the posteriors-first reading and avoids discarding mass.

    Because ``sum_k P[t, k] = 1``, the pooled (all-state) posterior-weighted mean equals the simple mean of the outcome
    over the valid rows; ``delta_vs_pooled[k]`` is therefore exactly ``w_mean[k] - pooled_mean``.

    Parameters
    ----------
    P
        Causal posterior matrix ``(T, K)`` — pure filtering or ``as_of``-aligned. **Rows must sum to 1** within
        ``row_sum_tol``; the function raises otherwise (the contrast math depends on a proper posterior).
    outcomes
        Mapping ``axis_name -> ForwardOutcome``. Each ``values`` is length ``T``; its ``horizon`` sets the HAC bandwidth.
    state_names
        Optional length-``K`` labels (e.g. ``["bear", "neutral", "bull"]``) surfaced as ``state_label``.
    min_coverage
        Soft-count floor (``sum_t w_t``). Cells below it are flagged ``low_coverage=True`` and their ``se_hac`` / ``t_hac``
        are set to ``nan`` (a contrast on too little mass is not trustworthy, regardless of point estimate).
    row_sum_tol
        Maximum allowed ``|sum_k P[t, k] - 1|``. Rejects logits / likelihoods / scaled scores that would silently bias
        ``coverage``, ``w_mean``, and ``delta_vs_pooled`` and manufacture false asymmetry.

    Returns
    -------
    pandas.DataFrame
        Tidy long frame, one row per ``(axis, state)``, with columns: ``axis, state, state_label, horizon, n_valid,
        n_map, coverage, n_eff, w_mean, q25, q50, q75, pooled_mean, delta_vs_pooled, se_hac, t_hac, low_coverage``.
    """
    p = _validate_posterior_matrix(P, "forward_outcome_by_state")
    T, K = p.shape
    if T > 0:
        max_row_dev = float(np.abs(p.sum(axis=1) - 1.0).max())
        if max_row_dev > row_sum_tol:
            raise ValueError(
                f"forward_outcome_by_state: posterior rows must sum to 1 within row_sum_tol={row_sum_tol} "
                f"(max deviation {max_row_dev:.3e}). The pooled-mean / delta-vs-pooled / coverage math assumes a proper "
                f"posterior — pass normalized probabilities, not logits, likelihoods, or scaled scores."
            )
    if not outcomes:
        raise ValueError("forward_outcome_by_state: outcomes must contain at least one axis series.")
    if state_names is not None and len(state_names) != K:
        raise ValueError(f"forward_outcome_by_state: state_names has {len(state_names)} entries, expected K={K}.")

    map_state = np.argmax(p, axis=1)
    rows: list[dict] = []

    for axis_name, fo in outcomes.items():
        o = np.asarray(fo.values, dtype=float)
        if o.shape != (T,):
            raise ValueError(f"forward_outcome_by_state: outcome '{axis_name}' must have shape ({T},), got {o.shape}.")
        horizon = int(fo.horizon)
        if horizon < 1:
            raise ValueError(f"forward_outcome_by_state: outcome '{axis_name}' horizon must be >= 1, got {horizon}.")

        finite = np.isfinite(o)
        n_valid = int(finite.sum())
        label = (lambda k: state_names[k]) if state_names is not None else (lambda k: None)

        if n_valid == 0:
            for k in range(K):
                rows.append(_empty_row(axis_name, k, label(k), horizon))
            continue

        o_valid = o[finite]
        pooled_mean = float(o_valid.mean())
        map_valid = map_state[finite]

        for k in range(K):
            w = p[finite, k]
            coverage = float(w.sum())
            n_map = int((map_valid == k).sum())
            if coverage <= 0.0:
                rows.append(_empty_row(axis_name, k, label(k), horizon, n_valid=n_valid, n_map=n_map,
                                       pooled_mean=pooled_mean, coverage=0.0))
                continue

            w_mean = float((w * o_valid).sum() / coverage)
            q25, q50, q75 = _weighted_quantiles(o_valid, w, _QUANTILE_LEVELS)
            # Contrast contributions: delta = sum_t (w_t/coverage - 1/n_valid) * (o_t - pooled) == w_mean - pooled.
            contributions = (w / coverage - 1.0 / n_valid) * (o_valid - pooled_mean)
            delta = float(contributions.sum())
            low_coverage = coverage < min_coverage
            if low_coverage:
                se_hac = float("nan")
                t_hac = float("nan")
            else:
                var_delta = bartlett_hac_variance(contributions, bandwidth=horizon - 1)
                se_hac = float(np.sqrt(var_delta)) if np.isfinite(var_delta) else float("nan")
                t_hac = float(delta / se_hac) if se_hac > 0.0 else float("nan")

            rows.append({"axis": axis_name, "state": k, "state_label": label(k), "horizon": horizon,
                         "n_valid": n_valid, "n_map": n_map, "coverage": coverage, "n_eff": coverage / horizon,
                         "w_mean": w_mean, "q25": float(q25), "q50": float(q50), "q75": float(q75),
                         "pooled_mean": pooled_mean, "delta_vs_pooled": delta, "se_hac": se_hac, "t_hac": t_hac,
                         "low_coverage": low_coverage})

    return pd.DataFrame(rows, columns=_COLUMNS)


def _empty_row(axis: str, state: int, label: str | None, horizon: int, *, n_valid: int = 0, n_map: int = 0,
               pooled_mean: float = float("nan"), coverage: float = float("nan")) -> dict:
    """A profile row with NaN statistics, used when a state has no posterior mass on valid rows."""
    return {"axis": axis, "state": state, "state_label": label, "horizon": horizon, "n_valid": n_valid, "n_map": n_map,
            "coverage": coverage, "n_eff": float("nan"), "w_mean": float("nan"), "q25": float("nan"),
            "q50": float("nan"), "q75": float("nan"), "pooled_mean": pooled_mean, "delta_vs_pooled": float("nan"),
            "se_hac": float("nan"), "t_hac": float("nan"), "low_coverage": True}
