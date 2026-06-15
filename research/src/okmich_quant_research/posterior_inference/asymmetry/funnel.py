"""The asymmetry confirmer — Stage 2 of the screener funnel.

Validates candidate feature-subsets: per subset, generate a walk-forward causal posterior stream and judge it with
``validate_stream``. **Decoupled from the screener** — it takes feature-subsets, not the screener object — so the
``hmm_screener`` stays independently runnable and the screener→confirmer handoff is a thin driver / lab example.

Per-candidate failures (a fit that doesn't converge, too few rows, etc.) are captured, not raised, so one bad subset
never sinks the batch — the same robustness the screener uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .forward_axes import MarketAxis
from .sources import HmmFitSpec, WalkForwardWindow, walk_forward_filtered_posteriors
from .validation import ValidationReport, ValidationVerdict, validate_stream


@dataclass(frozen=True)
class CandidateResult:
    """One candidate's verdict. ``report`` is ``None`` when generation/validation failed (see ``error``)."""

    features: tuple[str, ...]
    report: ValidationReport | None
    error: str | None = None

    @property
    def confirmed_axes(self) -> list[str]:
        if self.report is None:
            return []
        return [axis for axis, verdict in self.report.verdicts.items() if verdict == ValidationVerdict.CONFIRMED]


def confirm_candidates(candidates: list[list[str]], data: pd.DataFrame, *, fit: HmmFitSpec, window: WalkForwardWindow,
                       axes: list[MarketAxis], horizons: list[int], min_coverage: float = 200.0,
                       t_threshold: float = 2.0, min_stable_fraction: float = 0.6,
                       identity_axis: MarketAxis = MarketAxis.VOLATILITY) -> list[CandidateResult]:
    """Walk-forward + validate each candidate feature-subset; return per-candidate results.

    Winners are ``[r for r in results if r.confirmed_axes]`` — candidates with an incremental, fold-stable, overlap-
    corrected edge on at least one axis. ``data`` supplies both the feature columns (for the HMM source) and the price
    columns (for the forward-axis outcomes the judge builds).
    """
    results: list[CandidateResult] = []
    for subset in candidates:
        features = tuple(subset)
        try:
            stream = walk_forward_filtered_posteriors(data, feature_columns=list(subset), fit=fit, window=window,
                                                      identity_axis=identity_axis)
            report = validate_stream(stream, data, axes=axes, horizons=horizons, min_coverage=min_coverage,
                                     t_threshold=t_threshold, min_stable_fraction=min_stable_fraction)
            results.append(CandidateResult(features, report))
        except Exception as exc:  # one bad subset shouldn't sink the batch
            results.append(CandidateResult(features, None, error=f"{type(exc).__name__}: {exc}"))
    return results
