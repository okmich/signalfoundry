"""Posterior-asymmetry validation — the Stage-2 *judge* (source-agnostic).

Given a causal ``PosteriorStream`` (from a walk-forward refit, a frozen artifact, or a live log — the validator does not
care which) and the price frame it was generated over, decide whether any state separates a market axis with edge that is
**incremental over the trivial trailing baseline**, overlap-corrected, and (when folds are present) stable across them.

Layering:
  * ``incremental_residual`` — remove the linear trailing-baseline component of a forward outcome (pure).
  * ``validate_outcomes`` — the judge over pre-built probes (pure; fully unit-testable with planted edges).
  * ``validate_stream`` — convenience wrapper that builds the per-axis probes from a price frame, then judges.

This is the back end of the screener funnel, but it has no dependency on the screener: any posterior source composes with it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from .profiler import ForwardOutcome, forward_outcome_by_state
from .forward_axes import MarketAxis, forward_axis_series


@dataclass(frozen=True)
class PosteriorStream:
    """A causal posterior matrix plus the metadata the judge needs. The seam every posterior source produces."""

    probs: NDArray                       # (T, K) causal, stitched across folds in time order
    state_names: list[str]               # length K, economic identity (e.g. ["low_vol", "mid_vol", "high_vol"])
    index: pd.DatetimeIndex              # length T, decision-time timestamp of each row
    fold_ids: NDArray | None = None      # length T fold index per row; None / single value => no fold-stability test

    def __post_init__(self) -> None:
        p = np.asarray(self.probs)
        if p.ndim != 2:
            raise ValueError(f"PosteriorStream.probs must be 2D (T, K), got shape {p.shape}.")
        if len(self.state_names) != p.shape[1]:
            raise ValueError(f"state_names has {len(self.state_names)} entries, expected K={p.shape[1]}.")
        if len(self.index) != p.shape[0]:
            raise ValueError(f"index has {len(self.index)} entries, expected T={p.shape[0]}.")
        if self.fold_ids is not None and len(self.fold_ids) != p.shape[0]:
            raise ValueError(f"fold_ids has {len(self.fold_ids)} entries, expected T={p.shape[0]}.")


class ValidationVerdict(enum.StrEnum):
    CONFIRMED = "confirmed"        # incremental, overlap-corrected, fold-stable edge on the axis
    REJECTED = "rejected"          # no incremental edge (separation, if any, is just baseline persistence)
    INCONCLUSIVE = "inconclusive"  # no judgeable incremental cell (coverage), or folds present but none assessable


class AxisProbe(NamedTuple):
    """One unit of validation work: an axis at a horizon, with its forward outcome and trailing baseline.

    The per-axis verdict searches the best-incremental cell across *all* a given axis's probes (states x horizons), so no
    horizon is privileged — every probe for an axis is a candidate cell.
    """

    axis: str
    horizon: int
    forward: NDArray
    baseline: NDArray


@dataclass(frozen=True)
class ValidationReport:
    table: pd.DataFrame                       # per (axis, horizon, kind in {raw, incremental}, state) contrast + t_hac
    focal_summary: pd.DataFrame               # per axis: best-incremental (focal) cell, its deflated bar, fold-frac, verdict
    per_fold: pd.DataFrame                    # per (axis, fold) incremental focal-cell t_hac
    verdicts: dict[str, ValidationVerdict]    # axis -> verdict
    summary: str


def incremental_residual(forward: NDArray, baseline: NDArray) -> NDArray:
    """Forward outcome minus its linear trailing-baseline fit — the part trailing persistence cannot explain.

    A deliberately weak (linear) control: if even this removes the state's apparent edge, the state was re-deriving
    persistence. NaN-safe and NaN-preserving (rows where either input is NaN stay NaN).
    """
    f = np.asarray(forward, dtype=float)
    b = np.asarray(baseline, dtype=float)
    resid = np.full(f.shape, np.nan)
    mask = np.isfinite(f) & np.isfinite(b)
    if mask.sum() >= 2 and np.ptp(b[mask]) > 0:
        slope, intercept = np.polyfit(b[mask], f[mask], 1)
        resid[mask] = f[mask] - (slope * b[mask] + intercept)
    elif mask.sum() >= 1:
        resid[mask] = f[mask] - f[mask].mean()  # degenerate baseline -> just de-mean
    return resid


def _contrast_table(probs: NDArray, probe: AxisProbe, kind: str, state_names: list[str], min_coverage: float,
                    values: NDArray) -> pd.DataFrame:
    df = forward_outcome_by_state(probs, {probe.axis: ForwardOutcome(values, probe.horizon)},
                                  state_names=state_names, min_coverage=min_coverage)
    df = df.rename(columns={"axis": "market_axis"})
    df["kind"] = kind
    return df


def _fold_stability(probs: NDArray, fold_ids: NDArray, unique_folds: NDArray, axis: str, horizon: int,
                    resid: NDArray, focal: int, state_names: list[str], min_coverage: float,
                    t_threshold: float) -> tuple[float, int, list[dict]]:
    """Per-fold incremental significance of the focal state. Returns (stable_fraction, n_evaluable, fold_rows).

    ``stable_fraction`` is over evaluable folds only (a fold whose focal cell is low-coverage or has an undefined t is
    skipped, not counted as a failure); ``nan`` when no fold is evaluable.
    """
    rows: list[dict] = []
    passes = evaluable = 0
    for f in unique_folds:
        mask = fold_ids == f
        fold_table = forward_outcome_by_state(probs[mask], {axis: ForwardOutcome(resid[mask], horizon)},
                                              state_names=state_names, min_coverage=min_coverage)
        row = fold_table[fold_table.state == focal].iloc[0]
        if row.low_coverage or not np.isfinite(row.t_hac):
            continue
        evaluable += 1
        passes += int(abs(row.t_hac) > t_threshold)
        rows.append({"market_axis": axis, "fold": int(f), "state": focal,
                     "t_hac": float(row.t_hac), "delta_vs_pooled": float(row.delta_vs_pooled)})
    return (passes / evaluable if evaluable else float("nan")), evaluable, rows


def _bonferroni_t(t_threshold: float, n_tests: int) -> float:
    """Two-sided Bonferroni-adjusted ``|t|`` bar a best-of-``n_tests`` statistic must clear to hold the family-wise error
    at the level implied by ``t_threshold`` (``n_tests=1`` returns ``t_threshold``). Deflates the best-cell search so a
    near-miss has to be real, not the largest of many draws."""
    n = max(int(n_tests), 1)
    alpha = 2.0 * float(norm.sf(t_threshold))            # two-sided per-comparison alpha implied by t_threshold
    return float(norm.isf((alpha / n) / 2.0))            # two-sided t for that alpha spread over n comparisons


def _inconclusive_focal_row(axis: str) -> dict:
    return {"market_axis": axis, "focal_state": -1, "focal_state_label": None, "horizon": -1,
            "delta_vs_pooled": float("nan"), "t_hac": float("nan"), "deflated_t": float("nan"),
            "n_cells_tested": 0, "fold_fraction": float("nan"), "verdict": ValidationVerdict.INCONCLUSIVE.value}


def validate_outcomes(stream: PosteriorStream, probes: list[AxisProbe], *, min_coverage: float = 200.0,
                      t_threshold: float = 2.0, min_stable_fraction: float = 0.6) -> ValidationReport:
    """Judge pre-built probes. Pure: no price/HMM dependency, so it is unit-testable with planted edges."""
    if not probes:
        raise ValueError("validate_outcomes: probes must be non-empty.")
    probs = np.asarray(stream.probs, dtype=float)

    raw_parts, inc_parts = [], []
    residuals: dict[tuple[str, int], NDArray] = {}
    for pr in probes:
        resid = incremental_residual(pr.forward, pr.baseline)
        residuals[(pr.axis, pr.horizon)] = resid
        raw_parts.append(_contrast_table(probs, pr, "raw", stream.state_names, min_coverage, pr.forward))
        inc_parts.append(_contrast_table(probs, pr, "incremental", stream.state_names, min_coverage, resid))
    table = pd.concat(raw_parts + inc_parts, ignore_index=True)

    fold_rows: list[dict] = []
    focal_rows: list[dict] = []
    verdicts: dict[str, ValidationVerdict] = {}
    fold_ids = np.asarray(stream.fold_ids) if stream.fold_ids is not None else None
    unique_folds = np.unique(fold_ids) if fold_ids is not None else np.array([])
    can_test_folds = unique_folds.size >= 2

    inc_table = table[table.kind == "incremental"]
    for axis in dict.fromkeys(pr.axis for pr in probes):                       # unique axes, order preserved
        cells = inc_table[(inc_table.market_axis == axis) & (~inc_table.low_coverage) & inc_table.t_hac.notna()]
        if cells.empty:                                                        # no judgeable incremental cell
            verdicts[axis] = ValidationVerdict.INCONCLUSIVE
            focal_rows.append(_inconclusive_focal_row(axis))
            continue

        best = cells.loc[cells.t_hac.abs().idxmax()]                           # best incremental over (state, horizon)
        focal_state, focal_h, n_cells = int(best.state), int(best.horizon), int(len(cells))
        deflated_t = _bonferroni_t(t_threshold, n_cells)                       # deflate the best-of-n_cells search
        pooled_pass = abs(best.t_hac) > deflated_t

        fraction = float("nan")
        if not pooled_pass:
            verdict = ValidationVerdict.REJECTED                               # no incremental edge beyond the baseline
        elif not can_test_folds:
            verdict = ValidationVerdict.CONFIRMED                              # single-fold stream: pooled-only
        else:
            fraction, evaluable, rows = _fold_stability(probs, fold_ids, unique_folds, axis, focal_h,
                                                         residuals[(axis, focal_h)], focal_state, stream.state_names,
                                                         min_coverage, t_threshold)
            fold_rows.extend(rows)
            if evaluable == 0:
                verdict = ValidationVerdict.INCONCLUSIVE
            else:
                verdict = (ValidationVerdict.CONFIRMED if fraction >= min_stable_fraction
                           else ValidationVerdict.REJECTED)                    # deflated-significant but not fold-stable
        verdicts[axis] = verdict
        focal_rows.append({"market_axis": axis, "focal_state": focal_state, "focal_state_label": best.state_label,
                           "horizon": focal_h, "delta_vs_pooled": float(best.delta_vs_pooled),
                           "t_hac": float(best.t_hac), "deflated_t": deflated_t, "n_cells_tested": n_cells,
                           "fold_fraction": fraction, "verdict": verdict.value})

    confirmed = [a for a, v in verdicts.items() if v == ValidationVerdict.CONFIRMED]
    summary = (f"confirmed={confirmed or 'none'}; verdicts=" +
               ", ".join(f"{a}:{v.value}" for a, v in verdicts.items()))
    focal_cols = ["market_axis", "focal_state", "focal_state_label", "horizon", "delta_vs_pooled", "t_hac",
                  "deflated_t", "n_cells_tested", "fold_fraction", "verdict"]
    per_fold = pd.DataFrame(fold_rows, columns=["market_axis", "fold", "state", "t_hac", "delta_vs_pooled"])
    return ValidationReport(table=table, focal_summary=pd.DataFrame(focal_rows, columns=focal_cols),
                            per_fold=per_fold, verdicts=verdicts, summary=summary)


def validate_stream(stream: PosteriorStream, prices: pd.DataFrame, *, axes: list[MarketAxis], horizons: list[int],
                    min_coverage: float = 200.0, t_threshold: float = 2.0,
                    min_stable_fraction: float = 0.6) -> ValidationReport:
    """Build per-axis probes from ``prices`` (aligned to ``stream.index``), then judge via ``validate_outcomes``.

    ``prices`` must contain ``stream.index`` (axis series are computed on the full frame for correct forward/trailing
    windows, then sliced to the stream's rows). Each axis's verdict is the best deflated, fold-stable incremental cell
    across all (state x horizon) — no horizon is privileged.
    """
    if not horizons:
        raise ValueError("validate_stream: horizons must be non-empty.")
    pos = prices.index.get_indexer(stream.index)
    if (pos < 0).any():
        raise ValueError("validate_stream: stream.index is not fully contained in prices.index.")

    probes: list[AxisProbe] = []
    for axis in axes:
        for h in horizons:
            forward_full, baseline_full = forward_axis_series(prices, axis, h)
            probes.append(AxisProbe(axis.value, h, forward_full[pos], baseline_full[pos]))
    return validate_outcomes(stream, probes, min_coverage=min_coverage, t_threshold=t_threshold,
                             min_stable_fraction=min_stable_fraction)
