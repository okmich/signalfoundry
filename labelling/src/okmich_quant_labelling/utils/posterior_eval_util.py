"""Posterior-native analogues of the hard-label evaluation trio.

The hard-label utilities in :mod:`okmich_quant_labelling.utils.label_eval_util` operate on ``argmax(gamma)`` —
they collapse the per-bar posterior ``gamma_t in simplex(K-1)`` to a single integer and discard the confidence.
This module mirrors that trio for the *uncollapsed* posterior:

- :func:`all_posteriors_dynamics_statistics` mirrors
  ``all_labels_path_structure_statistics`` (per-state characterisation).
- :func:`evaluate_all_posteriors_returns_potentials` mirrors
  ``evaluate_all_labels_regime_returns_potentials`` (strategy tradeability).

Both compose the single-series primitives already living in
:mod:`okmich_quant_ml.posterior_inference` (``entropy``,
``summarize_posterior_dynamics``, ``posterior_calibration_report``) — this module only adds the *across-variant* workspace layer.

Design contract (confirmed with research lead, 2026-05-31)
----------------------------------------------------------
1. **Source-agnostic.** These evaluators never run inference and never choose filtered/smoothed/fixed-lag. They consume
    whatever ``gamma`` they are handed. Provenance (which inference mode produced each variant) is carried as an
   optional tag and surfaced in the output so a smoothed/oracle row is never silently compared against a fixed-lag row.
2. **Data layout.** One wide DataFrame. Each variant's posterior lives in columns ``post_{variant}_s{k}`` (k = 0..K-1),
    mirroring the ``lbl_{variant}`` convention. Variants and K are auto-discovered by parsing column names.
3. **State alignment is axis-relative.** States are ordered/labelled by their posterior-weighted mean of a chosen
    market-axis feature (``a_k = sum_t gamma_{t,k} x_t / sum_t gamma_{t,k}``) — e.g. ``smoothed_atr`` for a volatility
    axis, ``dbl_smoothed_log_rets`` for a momentum axis. This is how multiple variants are made comparable at all.
4. **Trade direction comes from return-sign mapping, not the axis.** The sign ``sign_k`` used to build positions is
    taken from the existing
   :func:`map_label_to_trend_direction` applied to ``argmax(gamma)`` — the same rule the hard-label trio uses, which
   keeps the posterior returns table apples-to-apples with the hard one. The axis only orders/labels states. Because
   argmax-mapping discards confidence, every state also carries a ``sign_divergence`` flag comparing it against the sign
   implied by the posterior-weighted mean return (see :func:`evaluate_all_posteriors_returns_potentials`).
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from okmich_quant_ml.posterior_inference import entropy, summarize_posterior_dynamics, posterior_calibration_report
from okmich_quant_labelling.utils.label_util import map_label_to_trend_direction

POSTERIOR_COL_PATTERN = re.compile(r"^post_(?P<variant>.+)_s(?P<state>\d+)$")
_SIMPLEX_TOL = 1e-6


# ---------------------------------------------------------------------------
# Discovery / extraction helpers
# ---------------------------------------------------------------------------

def discover_posterior_variants(df: pd.DataFrame, pattern: re.Pattern = POSTERIOR_COL_PATTERN) -> Dict[str, List[str]]:
    """Map each variant name to its ordered list of ``post_{variant}_s{k}`` columns.

    Columns are ordered by state index ``k``. Variants whose state indices are not a contiguous ``0..K-1`` run raise,
    since a gap means a missing state column rather than a genuinely sparse posterior.
    """
    grouped: Dict[str, Dict[int, str]] = {}
    for col in df.columns:
        match = pattern.match(str(col))
        if match is None:
            continue
        grouped.setdefault(match.group("variant"), {})[int(match.group("state"))] = col

    variants: Dict[str, List[str]] = {}
    for variant, state_to_col in grouped.items():
        states = sorted(state_to_col)
        if states != list(range(len(states))):
            raise ValueError(f"Variant '{variant}' has non-contiguous state columns {states}; expected 0..{len(states) - 1}.")
        variants[variant] = [state_to_col[k] for k in states]
    return variants


def extract_posteriors(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Return the ``(T, K)`` posterior matrix for ``cols`` and validate the simplex."""
    probs = df[list(cols)].to_numpy(dtype=float)
    if probs.size == 0:
        raise ValueError("Empty posterior matrix.")
    if probs.shape[1] < 2:
        raise ValueError("Posterior must have K >= 2 states; a single-state posterior carries no regime information.")
    if np.isnan(probs).any():
        raise ValueError("Posterior matrix contains NaN; slice off warmup/seam rows before evaluating.")
    if (probs < -_SIMPLEX_TOL).any() or (probs > 1.0 + _SIMPLEX_TOL).any():
        raise ValueError("Posterior entries must lie in [0, 1].")
    row_sums = probs.sum(axis=1)
    if np.abs(row_sums - 1.0).max() > _SIMPLEX_TOL:
        raise ValueError(f"Posterior rows must sum to 1 (max drift {np.abs(row_sums - 1.0).max():.2e}).")
    return probs


def posterior_weighted_mean(probs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-state posterior-weighted mean ``sum_t g_{t,k} x_t / sum_t g_{t,k}``, shape ``(K,)``.

    NaN entries in ``values`` are masked per state (their posterior mass is dropped from both numerator and denominator).
    """
    K = probs.shape[1]
    out = np.full(K, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    x = values[finite]
    w = probs[finite]
    denom = w.sum(axis=0)
    num = (w * x[:, None]).sum(axis=0)
    nonzero = denom > 0
    out[nonzero] = num[nonzero] / denom[nonzero]
    return out


def _posterior_weighted_std(probs: np.ndarray, values: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Per-state posterior-weighted standard deviation, shape ``(K,)``."""
    K = probs.shape[1]
    out = np.full(K, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    x = values[finite]
    w = probs[finite]
    denom = w.sum(axis=0)
    for k in range(K):
        if denom[k] <= 0 or not np.isfinite(means[k]):
            continue
        var = (w[:, k] * (x - means[k]) ** 2).sum() / denom[k]
        out[k] = float(np.sqrt(max(var, 0.0)))
    return out


def _effective_sample_size(probs: np.ndarray) -> np.ndarray:
    """Per-state effective sample size ``(sum g)^2 / sum g^2``, shape ``(K,)``."""
    s1 = probs.sum(axis=0)
    s2 = (probs ** 2).sum(axis=0)
    out = np.zeros(probs.shape[1], dtype=float)
    nonzero = s2 > 0
    out[nonzero] = (s1[nonzero] ** 2) / s2[nonzero]
    return out


def axis_ranks(axis_scores: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Dense rank of each state along an axis (0 = lowest score when ``ascending``)."""
    order = np.argsort(axis_scores if ascending else -axis_scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def _argmax_label(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def count_sign_divergence(signs: Dict[int, int], pw_mean_return: np.ndarray) -> int:
    """Number of states whose argmax-mapped sign disagrees with their posterior-weighted-mean-return sign.

    Only counts genuine disagreements: a state is divergent iff both signs are non-zero (a defined direction) and they
    point opposite ways. States mapped to flat, or with undefined/zero posterior-weighted mean, are not divergences.
    """
    divergent = 0
    for k, mapped in signs.items():
        if k >= len(pw_mean_return):
            continue
        pw = pw_mean_return[k]
        pw_sign = int(np.sign(pw)) if np.isfinite(pw) else 0
        if mapped != 0 and pw_sign != 0 and mapped != pw_sign:
            divergent += 1
    return divergent


def _state_signs(df: pd.DataFrame, probs: np.ndarray, price_col: str, method: str) -> Dict[int, int]:
    """Trade direction per state via ``map_label_to_trend_direction`` on ``argmax(gamma)``.

    Returns ``{state: sign}`` with sign in ``{-1, 0, 1}``. Uses the same mapper the hard-label trio uses so the
    posterior returns table stays comparable.
    """
    tmp = pd.DataFrame({
        "_argmax_state": _argmax_label(probs),
        "_log_ret": np.log(df[price_col].to_numpy(dtype=float) / df[price_col].shift(1).to_numpy(dtype=float)),
    }, index=df.index)
    mapping = map_label_to_trend_direction(tmp, state_col="_argmax_state", return_col="_log_ret", method=method)
    return {int(k): int(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# 1. Dynamics statistics (mirror of all_labels_path_structure_statistics)
# ---------------------------------------------------------------------------

def posterior_dynamics_statistics(df: pd.DataFrame, posterior_cols: Sequence[str], axis_col: Optional[str] = None,
                                  returns_col: str = "returns", price_col: str = "close", window: int = 20,
                                  tau: float = 0.5, ascending_axis: bool = True,
                                  rank_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-state posterior characterisation for one variant.

    Returns one row per state with both per-state metrics and variant-level scalars (duplicated across the variant's
    rows, mirroring how the hard-label table repeats discriminability). State rows are ordered by ``axis_col`` rank
    when an axis is supplied.

    Per-state columns
    -----------------
    - ``state`` : raw state index in the posterior matrix.
    - ``axis_rank`` / ``axis_label`` / ``axis_score`` : alignment along ``axis_col``
      (posterior-weighted mean of the axis feature). Present only when ``axis_col`` given.
    - ``occupancy`` : time-averaged posterior mass ``mean_t gamma_{t,k}`` (soft frequency).
    - ``argmax_freq_pct`` : fraction of bars where the state is argmax (hard frequency, %).
    - ``eff_n`` : posterior effective sample size ``(sum g)^2 / sum g^2``.
    - ``pw_mean_return`` / ``pw_vol`` / ``pw_sharpe`` : posterior-weighted return stats.

    Variant-level columns (constant within the variant)
    ---------------------------------------------------
    - ``mean_entropy`` : average Shannon entropy of ``gamma_t`` (nats).
    - ``perplexity`` : ``exp(mean_entropy)`` — effective number of states actually used.
    - ``decisiveness`` : fraction of bars with ``max gamma_t > tau``.
    - ``mean_flip_rate`` / ``mean_dwell_length`` / ``mean_step_kl`` : argmax-path
      dynamics from :func:`summarize_posterior_dynamics`.
    - ``occupancy_spread`` : std of occupancy across states (how unevenly mass is used).
    """
    probs = extract_posteriors(df, posterior_cols)
    T, K = probs.shape

    if returns_col in df.columns:
        returns = df[returns_col].to_numpy(dtype=float)
    elif price_col in df.columns:
        returns = np.log(df[price_col].to_numpy(dtype=float) / df[price_col].shift(1).to_numpy(dtype=float))
    else:
        raise ValueError(f"Neither returns_col '{returns_col}' nor price_col '{price_col}' found in DataFrame.")

    occupancy = probs.mean(axis=0)
    eff_n = _effective_sample_size(probs)
    pw_mean_ret = posterior_weighted_mean(probs, returns)
    pw_vol = _posterior_weighted_std(probs, returns, pw_mean_ret)
    pw_sharpe = np.divide(pw_mean_ret, pw_vol, out=np.full(K, np.nan), where=pw_vol > 0)
    argmax = _argmax_label(probs)
    argmax_freq = np.array([(argmax == k).mean() for k in range(K)]) * 100.0

    per_bar_entropy = entropy(probs)
    mean_entropy = float(per_bar_entropy.mean())
    dynamics = summarize_posterior_dynamics(probs, window=window)
    decisiveness = float((probs.max(axis=1) > tau).mean())

    if axis_col is not None:
        if axis_col not in df.columns:
            raise ValueError(f"axis_col '{axis_col}' not found in DataFrame.")
        axis_scores = posterior_weighted_mean(probs, df[axis_col].to_numpy(dtype=float))
        axis_rank = axis_ranks(axis_scores, ascending=ascending_axis)
    else:
        axis_scores = np.full(K, np.nan)
        axis_rank = np.arange(K)

    rows = []
    for k in range(K):
        row = {"state": k}
        if axis_col is not None:
            row["axis_rank"] = int(axis_rank[k])
            if rank_labels is not None:
                row["axis_label"] = rank_labels[int(axis_rank[k])] if int(axis_rank[k]) < len(rank_labels) else None
            row["axis_score"] = axis_scores[k]
        row.update({
            "occupancy": occupancy[k],
            "argmax_freq_pct": round(float(argmax_freq[k]), 4),
            "eff_n": round(float(eff_n[k]), 1),
            "pw_mean_return": pw_mean_ret[k],
            "pw_vol": pw_vol[k],
            "pw_sharpe": pw_sharpe[k],
            # variant-level scalars (constant across rows)
            "mean_entropy": mean_entropy,
            "perplexity": float(np.exp(mean_entropy)),
            "decisiveness": decisiveness,
            "mean_flip_rate": dynamics.mean_flip_rate,
            "mean_dwell_length": dynamics.mean_dwell_length,
            "mean_step_kl": dynamics.mean_step_kl,
            "occupancy_spread": float(occupancy.std()),
        })
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    sort_col = "axis_rank" if axis_col is not None else "occupancy"
    return stats_df.sort_values(sort_col, ascending=axis_col is not None).reset_index(drop=True)


def all_posteriors_dynamics_statistics(df: pd.DataFrame, variants: Optional[Sequence[str]] = None,
                                       axis_col: Optional[str] = None, returns_col: str = "returns",
                                       price_col: str = "close", window: int = 20, tau: float = 0.5,
                                       ascending_axis: bool = True, rank_labels: Optional[Sequence[str]] = None,
                                       provenance: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Across-variant posterior dynamics table (mirror of ``all_labels_path_structure_statistics``).

    Discovers variants from ``post_{variant}_s{k}`` columns (or uses the supplied ``variants`` subset), runs :func:
    `posterior_dynamics_statistics` per variant, and concatenates with an ``algo`` column. ``provenance``
    (``{variant: inference_mode}``) is attached as a column so smoothed/fixed-lag rows are never silently mixed.
    """
    discovered = discover_posterior_variants(df)
    selected = list(variants) if variants is not None else list(discovered)

    result = None
    for variant in selected:
        if variant not in discovered:
            raise ValueError(f"Variant '{variant}' has no post_{variant}_s* columns.")
        stats = posterior_dynamics_statistics(df, discovered[variant], axis_col=axis_col, returns_col=returns_col,
                                              price_col=price_col, window=window, tau=tau,
                                              ascending_axis=ascending_axis, rank_labels=rank_labels)
        stats.insert(0, "algo", variant)
        if provenance is not None:
            stats.insert(1, "provenance", provenance.get(variant, "unknown"))
        result = stats if result is None else pd.concat([result, stats])
    return result.reset_index(drop=True) if result is not None else None


# ---------------------------------------------------------------------------
# 2. Returns potentials (mirror of evaluate_all_labels_regime_returns_potentials)
# ---------------------------------------------------------------------------

def _soft_position(probs: np.ndarray, signs: Dict[int, int]) -> np.ndarray:
    """Graded position ``pos_t = sum_k gamma_{t,k} sign_k`` in ``[-1, 1]``."""
    sign_vec = np.array([signs.get(k, 0) for k in range(probs.shape[1])], dtype=float)
    return probs @ sign_vec


def _gated_position(probs: np.ndarray, signs: Dict[int, int], tau: float) -> np.ndarray:
    """Hard position: full size in the argmax direction iff ``max gamma_t > tau``, else flat."""
    argmax = probs.argmax(axis=1)
    top = probs.max(axis=1)
    sign_vec = np.array([signs.get(k, 0) for k in range(probs.shape[1])], dtype=float)
    pos = sign_vec[argmax]
    pos[top <= tau] = 0.0
    return pos


def _backtest_position(position: np.ndarray, fwd_returns: np.ndarray, whipsaw_cost: float,
                       progressive_skip: int) -> Dict[str, float]:
    """Per-bar soft backtest of a continuous position against forward returns.

    Unlike the hard-label evaluator (which blocks returns into contiguous regimes), a continuous position is evaluated
    bar-by-bar: ``pnl_t = pos_t * r_{t+1}``. Entry lag is modelled by shifting the position forward ``progressive_skip``
    bars; cost is charged on traded notional change (turnover).
    """
    if progressive_skip < 0:
        raise ValueError(f"progressive_skip must be >= 0, got {progressive_skip}.")
    pos = np.roll(position, progressive_skip).astype(float)
    pos[:progressive_skip] = 0.0
    valid = np.isfinite(fwd_returns)
    pos = pos[valid]
    r = fwd_returns[valid]

    gross = pos * r
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = turnover * whipsaw_cost
    net = gross - cost
    cum = np.cumsum(net)

    in_market = np.abs(pos) > 1e-9
    traded = gross[in_market]
    wins = traded[traded > 0]
    losses = traded[traded < 0]
    running_max = np.maximum.accumulate(cum) if cum.size else np.array([0.0])
    drawdown = running_max - cum

    if losses.sum() < 0:
        profit_factor = float(wins.sum() / -losses.sum())
    elif wins.sum() > 0:
        profit_factor = np.inf  # only winners observed
    else:
        profit_factor = np.nan  # no trades taken — undefined, not infinite

    return {
        "n_bars": int(pos.size),
        "exposure": float(in_market.mean()) if pos.size else 0.0,
        "avg_abs_position": float(np.abs(pos).mean()) if pos.size else 0.0,
        "turnover": float(turnover.sum()),
        "n_position_changes": int((turnover > 1e-9).sum()),
        "gross_return": float(gross.sum()),
        "total_whipsaw_cost": float(cost.sum()),
        "net_return": float(net.sum()),
        "mean_pnl": float(net.mean()) if net.size else 0.0,
        "sharpe_ratio": float(net.mean() / net.std()) if net.size and net.std() > 0 else 0.0,
        "win_rate": float((traded > 0).mean()) if traded.size else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": float(drawdown.max()) if drawdown.size else 0.0,
        "mean_drawdown": float(drawdown.mean()) if drawdown.size else 0.0,
    }


def evaluate_posterior_returns_potentials(df: pd.DataFrame, posterior_cols: Sequence[str], price_col: str = "close",
                                          whipsaw_cost: float = 0.0, tau: float = 0.5,
                                          sign_mapping_method: str = "sharpe", progressive_skip: int = 1,
                                          truth_col: Optional[str] = None,
                                          sign_map: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """Tradeability *potential* of one variant's posterior under soft vs confidence-gated positions.

    Returns two rows — ``position='soft'`` and ``position='gated'`` — each with its own turnover/whipsaw/Sharpe/drawdown
    so the cost of graded sizing vs a hard gate is directly readable. Trade direction per state comes from the argmax
    return-sign mapping; a ``n_sign_divergent`` column flags states whose argmax-mapped sign disagrees with the sign of
    their posterior-weighted mean return (the cheap safety net against argmax contamination by ambiguous/transition bars).

    IN-SAMPLE BY CONSTRUCTION. When ``sign_map`` is not supplied the per-state direction is *learned from the same
    sample it is then evaluated on* (identical to the hard-label ``evaluate_regime_returns_potentials``). The numbers are
    therefore upper-bound *potential* diagnostics, NOT a deployable out-of-sample strategy estimate. For an honest OOS
    figure, pass a ``sign_map`` (``{state: sign in {-1,0,1}}``) prefit on a disjoint training window.

    ``truth_col`` (an integer state-ground-truth column) enables a calibration audit (``ece``/``brier_score``/``nll``).
    It is off by default because calibrating a posterior against its own argmax is circular (per the upstream package warning).
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in DataFrame.")
    probs = extract_posteriors(df, posterior_cols)
    K = probs.shape[1]
    log_ret = np.log(df[price_col].to_numpy(dtype=float) / df[price_col].shift(1).to_numpy(dtype=float))
    fwd_returns = np.roll(log_ret, -1)
    fwd_returns[-1] = np.nan

    signs = dict(sign_map) if sign_map is not None else _state_signs(df, probs, price_col, sign_mapping_method)

    # sign-divergence flag: argmax-mapped sign vs posterior-weighted-mean-return sign
    pw_mean_ret = posterior_weighted_mean(probs, log_ret)
    n_divergent = count_sign_divergence({k: signs.get(k, 0) for k in range(K)}, pw_mean_ret)

    calibration = {}
    if truth_col is not None:
        report = posterior_calibration_report(probs, df[truth_col].to_numpy(dtype=np.int64))
        calibration = {"ece": report.ece, "brier_score": report.brier_score, "nll": report.nll}

    positions = {
        "soft": _soft_position(probs, signs),
        "gated": _gated_position(probs, signs, tau),
    }
    rows = []
    for name, pos in positions.items():
        stats = _backtest_position(pos, fwd_returns, whipsaw_cost, progressive_skip)
        stats = {"position": name, "n_states": K, "n_sign_divergent": n_divergent,
                 "sign_map": {k: signs.get(k, 0) for k in range(K)}, **stats, **calibration}
        rows.append(stats)
    return pd.DataFrame(rows)


def evaluate_all_posteriors_returns_potentials(df: pd.DataFrame, variants: Optional[Sequence[str]] = None,
                                               price_col: str = "close", whipsaw_cost: float = 0.0, tau: float = 0.5,
                                               sign_mapping_method: str = "sharpe", progressive_skip: int = 1,
                                               truth_cols: Optional[Dict[str, str]] = None,
                                               sign_maps: Optional[Dict[str, Dict[int, int]]] = None,
                                               provenance: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Across-variant soft tradeability table (mirror of ``evaluate_all_labels_regime_returns_potentials``).

    One pair of rows (soft, gated) per variant. ``truth_cols`` optionally maps a variant to its ground-truth state
    column for the calibration audit; ``sign_maps`` optionally maps a variant to a prefit ``{state: sign}`` (for
    out-of-sample direction — see :func:`evaluate_posterior_returns_potentials` on the in-sample default); ``provenance``
    tags each row with the inference mode that produced the posterior.
    """
    discovered = discover_posterior_variants(df)
    selected = list(variants) if variants is not None else list(discovered)
    truth_cols = truth_cols or {}
    sign_maps = sign_maps or {}

    result = None
    for variant in selected:
        if variant not in discovered:
            raise ValueError(f"Variant '{variant}' has no post_{variant}_s* columns.")
        stats = evaluate_posterior_returns_potentials(df, discovered[variant], price_col=price_col,
                                                      whipsaw_cost=whipsaw_cost, tau=tau,
                                                      sign_mapping_method=sign_mapping_method,
                                                      progressive_skip=progressive_skip,
                                                      truth_col=truth_cols.get(variant),
                                                      sign_map=sign_maps.get(variant))
        stats.insert(0, "algo", variant)
        if provenance is not None:
            stats.insert(1, "provenance", provenance.get(variant, "unknown"))
        result = stats if result is None else pd.concat([result, stats])
    return result.reset_index(drop=True) if result is not None else None
