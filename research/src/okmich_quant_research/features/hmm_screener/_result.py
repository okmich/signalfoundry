"""Result types for HmmFeatureScreener."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from ..screener._result import StageReport
from ._config import ScreenStrategy
from ._pareto import ParetoStatus


class BaselineRole(StrEnum):
    """What a supplied baseline is DOING in a given screen.

    The same list of features means opposite things under the two strategies, and conflating them
    is how an anchor's output gets read as a search finding:

    ``ANCHOR`` (``ABLATION``) -- it seeds every subset, so it is a PRIOR on the result.
    ``BENCHMARK`` (``GREEDY_FORWARD``) -- it seeds nothing; it is fitted once purely so the
    unanchored winner can be scored against the hand-frozen set it replaces.
    """
    ANCHOR = "anchor"
    BENCHMARK = "benchmark"


class GreedyStopReason(StrEnum):
    """Why a ``GREEDY_FORWARD`` beam search stopped growing."""
    DEPTH_CAP = "depth_cap"
    MIN_GAIN = "min_gain"
    POOL_EXHAUSTED = "pool_exhausted"
    NO_ELIGIBLE_CANDIDATES = "no_eligible_candidates"


@dataclass(frozen=True)
class GreedyStep:
    """One step of the beam search: what it fitted, what it kept, and what it bought.

    The sequence of ``relative_gain`` values IS the marginal-gain curve -- the generalisation of
    ``base_frac`` to an unanchored search. Where ``base_frac`` gives one ratio against a hand-picked
    anchor, this shows how much each additional feature actually added and where the curve flattens.
    """
    step: int
    n_fitted: int
    beam: tuple[tuple[str, ...], ...]
    best_subset: tuple[str, ...]
    best_separation: float
    gain: float
    relative_gain: float
    used_trap_fallback: bool = False
    """True when EVERY candidate at this step was a confidence trap and the step had to rank on
    the merely-non-fragile pool. The gain curve keeps climbing when this fires -- because traps are
    where the high separations live -- so without this flag a trapped search reads as a successful
    one. Any step with this set produces a best_subset that the Pareto layer will class TRAP."""


class WinnerPool(StrEnum):
    """Which pool of subsets the baseline prior was measured against.

    Falls back down this list because the ranked pool can be empty: on many axes the whole
    frontier lands in the confidence-trap quadrant, and a candidates-only definition of
    "winner" would return ``None`` on exactly the runs that most need auditing.
    """
    ASYMMETRY_CANDIDATES = "asymmetry_candidates"
    NON_FRAGILE = "non_fragile"
    ALL_EVALUATED = "all_evaluated"


@dataclass(frozen=True)
class BaselinePrior:
    """How much of a screen's answer the hand-supplied baseline already contained.

    ``ScreenStrategy.ABLATION`` enumerates ``baseline + drop-one + add-one``, so every add-one
    subset is ``baseline + candidate``: the search is a 1-neighbourhood of a fixed point and no
    subset ever pairs two non-baseline features. That makes the baseline a **prior on the
    result**, not merely a starting point — and an unreported prior reads as a search finding.
    This dataclass makes it auditable.

    Read ``base_frac`` as the fraction of the winner's ``axis_separation`` already delivered by the
    baseline alone. High values mean the search added little to what the anchor asserted;
    ``base_frac == 1.0`` means nothing beat the anchor and the "winner" IS the baseline.

    Diagnostic only — it measures anchor dominance, it does not remove it. Removing it needs a
    baseline-free enumeration (greedy-forward-from-empty or ``EXHAUSTIVE``).
    """
    baseline: tuple[str, ...]
    baseline_separation: float | None
    winner: tuple[str, ...] | None
    winner_separation: float | None
    base_frac: float | None
    winner_keeps_baseline: bool | None
    subsets_containing_baseline: int
    n_subsets: int
    winner_pool: WinnerPool | None
    role: BaselineRole = BaselineRole.ANCHOR

    def __str__(self) -> str:
        label = "baseline prior" if self.role == BaselineRole.ANCHOR else "baseline benchmark"
        if self.base_frac is None:
            return (f"{label}: NOT MEASURABLE (baseline={list(self.baseline)}, "
                    f"baseline_sep={self.baseline_separation}, winner_sep={self.winner_separation})")
        pool = self.winner_pool.value if self.winner_pool is not None else "n/a"
        head = (f"{label}: base_frac={self.base_frac:.3f} "
                f"(baseline_sep={self.baseline_separation:.4f} / winner_sep={self.winner_separation:.4f}, "
                f"pool={pool})")
        if self.role == BaselineRole.ANCHOR:
            keeps = "yes" if self.winner_keeps_baseline else "no"
            return (f"{head}; winner keeps full baseline: {keeps}; "
                    f"{self.subsets_containing_baseline}/{self.n_subsets} subsets contain the full baseline")
        # BENCHMARK: the baseline seeded nothing, so containment counts are not a dominance signal.
        # base_frac < 1 means the unanchored search beat the frozen set; >= 1 means it did not.
        verdict = ("unanchored search BEAT the frozen baseline" if self.base_frac < 1.0
                   else "unanchored search did NOT beat the frozen baseline")
        return f"{head}; {verdict}"


@dataclass(frozen=True)
class AxisEvaluation:
    """Axis-specific output from a single evaluator call.

    Returned by each ``evaluate_*`` function in ``_evaluators.py`` and folded
    into the per-subset ``SubsetEvaluation`` below.

    ``axis_separation`` is the **population-weighted standard deviation of
    per-state medians** of the axis target — a small-population outlier state
    contributes proportional to its mass, not its value. ``axis_separation_range``
    is the legacy ``max(median) - min(median)`` across states, preserved for
    back-comparison and for diagnosing when state-population skew is hiding
    or inflating the separation signal.
    """
    axis_separation: float
    secondary_robustness: float
    secondary_label: str
    axis_separation_range: float = 0.0
    raw_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubsetEvaluation:
    """One row of the screener's output, per candidate feature subset."""
    features: tuple[str, ...]
    n_features: int
    axis_separation: float
    secondary_robustness: float
    secondary_label: str
    honesty: float
    state_balance_ratio: float
    pareto_status: ParetoStatus
    axis_separation_range: float = 0.0
    warnings: tuple[str, ...] = ()
    raw_details: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: str | None = None  # populated if the fit / evaluation raised


@dataclass
class HmmScreenerResult:
    """Output of ``HmmFeatureScreener.screen()``.

    Combines per-subset evaluations, a tidy DataFrame view for ranking, and the stage reports describing what was pre-filtered.
    """
    evaluations: list[SubsetEvaluation]
    results_: pd.DataFrame
    stage_reports: list[StageReport] = field(default_factory=list)
    baseline: tuple[str, ...] | None = None
    """Baseline ABLATION actually anchored on, after intersecting with the stage-0 survivors.

    ``None`` when the strategy carries no anchor (``EXHAUSTIVE``). This is the *effective*
    baseline, not what the caller requested — a requested feature dropped by a stage-0
    pre-filter is absent here, and ``screen()`` warns when that happens.
    """
    strategy: ScreenStrategy | None = None
    search_trace: tuple[GreedyStep, ...] = ()
    """Per-step record of a ``GREEDY_FORWARD`` beam search; empty for the other strategies."""
    stop_reason: GreedyStopReason | None = None

    @property
    def asymmetry_candidates(self) -> list[SubsetEvaluation]:
        """Pareto-optimal, non-trap subsets (structural asymmetry candidates), ordered by axis_separation descending.

        Stage-1 output: these still need Stage-2 confirmation (walk-forward + incremental) before any is *confirmed*
        asymmetry. This is the list that feeds the confirmer funnel.
        """
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.ASYMMETRY_CANDIDATE),
            key=lambda e: e.axis_separation, reverse=True,
        )

    @property
    def traps(self) -> list[SubsetEvaluation]:
        """Subsets in the confidence-trap quadrant, ordered by axis_separation descending."""
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.TRAP),
            key=lambda e: e.axis_separation, reverse=True,
        )

    @property
    def fragile(self) -> list[SubsetEvaluation]:
        """Subsets flagged FRAGILE for structural degeneracy.

        These failed the Phase-A quality gate (missing states, balance ratio beyond ``config.max_balance_ratio``, or fewer than
        ``config.min_significant_states`` distinguished states). They never reached the Pareto check and should be
        investigated as model-structure failures before any decision is made about the feature subset.
        """
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.FRAGILE),
            key=lambda e: e.axis_separation, reverse=True,
        )

    @property
    def baseline_prior(self) -> BaselinePrior | None:
        """Audit of how much of this result the baseline anchor already contained.

        ``None`` when the screen had no anchor. See :class:`BaselinePrior` for why an ABLATION
        verdict is not interpretable without it.
        """
        if not self.baseline:
            return None
        baseline_key = tuple(sorted(self.baseline))
        n_baseline = len(baseline_key)
        baseline_set = set(baseline_key)

        scored = [e for e in self.evaluations if e.error is None]
        contains = sum(1 for e in scored if baseline_set.issubset(e.features))

        baseline_ev = next((e for e in scored if tuple(sorted(e.features)) == baseline_key), None)
        baseline_sep = baseline_ev.axis_separation if baseline_ev is not None else None

        pool, winner_pool = self.asymmetry_candidates, WinnerPool.ASYMMETRY_CANDIDATES
        if not pool:
            pool = sorted((e for e in scored if e.pareto_status != ParetoStatus.FRAGILE),
                          key=lambda e: e.axis_separation, reverse=True)
            winner_pool = WinnerPool.NON_FRAGILE
        if not pool:
            pool = sorted(scored, key=lambda e: e.axis_separation, reverse=True)
            winner_pool = WinnerPool.ALL_EVALUATED
        winner = pool[0] if pool else None

        winner_sep = winner.axis_separation if winner is not None else None
        # Guard the ratio: axis_separation is a population-weighted std so it cannot be negative,
        # but a degenerate fit can return exactly 0 and a 0-denominator ratio is not a prior.
        base_frac = None
        if baseline_sep is not None and winner_sep is not None and winner_sep > 0:
            base_frac = baseline_sep / winner_sep

        # Under GREEDY_FORWARD the baseline seeded nothing, so it is a benchmark, not a prior.
        role = (BaselineRole.BENCHMARK if self.strategy == ScreenStrategy.GREEDY_FORWARD
                else BaselineRole.ANCHOR)
        return BaselinePrior(
            baseline=baseline_key, baseline_separation=baseline_sep,
            winner=tuple(winner.features) if winner is not None else None, winner_separation=winner_sep,
            base_frac=base_frac,
            winner_keeps_baseline=(baseline_set.issubset(winner.features) if winner is not None else None),
            subsets_containing_baseline=contains, n_subsets=len(scored),
            winner_pool=winner_pool if winner is not None else None,
            role=role,
        )

    @property
    def marginal_gain_curve(self) -> pd.DataFrame:
        """Per-step gain of a ``GREEDY_FORWARD`` search; empty frame for the other strategies.

        The unanchored generalisation of ``base_frac``: instead of one ratio against a hand-picked
        anchor, it shows what each additional feature bought and where the curve flattens. Read the
        flattening point as the honest subset size -- features added past it are noise-fitting.
        """
        return pd.DataFrame([
            {"step": st.step, "n_fitted": st.n_fitted, "best_subset": ",".join(st.best_subset),
             "best_separation": st.best_separation, "gain": st.gain,
             "relative_gain": st.relative_gain, "beam_width": len(st.beam),
             "used_trap_fallback": st.used_trap_fallback}
            for st in self.search_trace
        ])

    @property
    def base_frac(self) -> float | None:
        """Shortcut for ``baseline_prior.base_frac`` — the anchor's share of the winner's separation."""
        prior = self.baseline_prior
        return prior.base_frac if prior is not None else None

    def __repr__(self) -> str:
        n = len(self.evaluations)
        n_candidates = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.ASYMMETRY_CANDIDATE)
        n_trap = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.TRAP)
        n_frag = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.FRAGILE)
        n_err = sum(1 for e in self.evaluations if e.error is not None)
        # base_frac rides in the repr on purpose: under ABLATION the verdict is only readable next
        # to its prior, and a repr is what a notebook shows when the result is displayed.
        bf = self.base_frac
        prior = "" if bf is None else f", base_frac={bf:.3f}"
        greedy = ""
        if self.search_trace:
            reason = self.stop_reason.value if self.stop_reason is not None else "?"
            greedy = (f", greedy depth={self.search_trace[-1].step} "
                      f"(stopped: {reason})")
        return (f"HmmScreenerResult({n} subsets, {n_candidates} candidates, {n_trap} traps, "
                f"{n_frag} fragile, {n_err} errors{prior}{greedy})")
