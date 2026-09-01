"""HmmFeatureScreener — research-time feature-subset selection for an axis HMM.

Sibling to the ML ``FeatureScreener`` in ``..screener``. Where the ML screener asks "which features predict a known target,"
this one asks "which features produce a coherent latent state structure for *this semantic axis*."

Workflow:
    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> from okmich_quant_research.features.hmm_screener import (
    ...     HmmFeatureScreener, HmmScreenerConfig, ScreenStrategy,
    ... )
    >>> reg = FeatureRegistry()
    >>> candidates = reg.candidates_for("regime", min_relevance="HIGH").names()
    >>> config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4)
    >>> screener = HmmFeatureScreener(config, raw_data, feature_engineering_fn)
    >>> result = screener.screen(candidates, strategy=ScreenStrategy.GREEDY_FORWARD)
    >>> result.results_             # ranked DataFrame
    >>> result.marginal_gain_curve  # what each greedy step bought
    >>> result.asymmetry_candidates # Pareto-optimal non-trap subsets (Stage-1 candidates)

Implementation notes:
    * Evaluator state labels are ``argmax(filtering gamma)`` (causal MAP), not the offline Viterbi path.
      This keeps axis-quality scoring consistent with what a live system would actually observe.
    * Evaluator OHLC inputs are joined from ``self.raw_data`` on index, not read from the ``feature_engineering`` output,
      so a "clean" feature function that returns only engineered columns still works with axes that need
      ``high``/``low`` (e.g. price_structure).
    * Pareto classification is preceded by a structural quality gate (``min_significant_states``, ``max_balance_ratio``);
      subsets failing either are marked ``FRAGILE`` and excluded from the frontier.
    * Off-axis coherence warnings are computed per-subset, so a warning names only the feature(s) actually contaminating that subset.
"""
from __future__ import annotations

import time
import traceback
import warnings
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_ml.posterior_inference import top_prob

from ..registry import FeatureRegistry
from ..screener._stage0 import stage0_variance_filter
from ..screener._result import StageReport
from ._config import HmmScreenerConfig, ScreenStrategy, build_hmm
from ._evaluators import get_evaluator
from ._persistence import stage0b_persistence_filter
from ._collinearity import stage0c_collinearity_filter
from ._pareto import ParetoStatus, classify_pareto
from ._result import (AxisEvaluation, GreedyStep, GreedyStopReason, HmmScreenerResult,
                      SubsetEvaluation)


_PASSTHROUGH_COLUMNS = ("open", "high", "low", "close", "tick_volume", "volume")


class HmmFeatureScreener:
    """Screen feature subsets for an axis-specific HMM.

    Composes:
      - Feature-engineering function (user-supplied; runs once per screen)
      - Stage-0 variance pre-filter (reused from the ML screener)
      - Per-subset off-axis coherence check via ``FeatureRegistry``
      - HMM fit + posterior-honesty diagnostic + axis-matched evaluator
      - Phase-A structural quality gate (state collapse / significance)
      - Phase-B Pareto classification on ``(axis_separation, honesty)``

    Parameters
    ----------
    config : HmmScreenerConfig
        Axis / algo / n_states + thresholds.
    raw_data : pd.DataFrame
        OHLCV bars. Must contain at least a ``close`` column. Tail
        ``config.data_size`` rows are used.
    feature_engineering : Callable[[pd.DataFrame], pd.DataFrame]
        Produces a DataFrame containing the candidate feature columns. Must be deterministic; the screener applies it once
            per call to ``screen()``.
        OHLC columns from ``raw_data`` are joined separately for evaluators, so the feature function does not need to preserve them.
    registry : FeatureRegistry, optional
        Used to validate off-axis features. Defaults to ``FeatureRegistry()``.
    """

    def __init__(self, config: HmmScreenerConfig, raw_data: pd.DataFrame,
                 feature_engineering: Callable[[pd.DataFrame], pd.DataFrame],
                 registry: FeatureRegistry | None = None):
        if "close" not in raw_data.columns:
            raise ValueError("raw_data must contain a 'close' column for axis evaluators.")
        self.config = config
        self.raw_data = (raw_data.iloc[-config.data_size:].copy()
                         if len(raw_data) > config.data_size else raw_data.copy())
        self.feature_engineering = feature_engineering
        self.registry = registry if registry is not None else FeatureRegistry()
        self._allowed_signal_types = config.effective_allowed_signal_types
        # Pre-compute the passthrough frame (OHLC + volume) used to join evaluator inputs.
        # Volume columns are accepted under either of two common names so the liquidity
        # evaluator works with the broader 5-min-OHLCV data conventions used in this repo.
        passthrough_cols = [c for c in _PASSTHROUGH_COLUMNS if c in self.raw_data.columns]
        self._passthrough = self.raw_data[passthrough_cols].copy()


    def screen(self, candidate_features: list[str], strategy: ScreenStrategy = ScreenStrategy.ABLATION,
               baseline: list[str] | None = None, max_subset_size: int | None = None) -> HmmScreenerResult:
        """Run the screen.

        Parameters
        ----------
        candidate_features : list[str]
            Features to screen. Must be column names produced by ``feature_engineering``.
        strategy : ScreenStrategy
            ``ABLATION`` (default): baseline + drop-one + add-one ablation. Anchored -- read
            ``result.base_frac`` beside the verdict.
            ``GREEDY_FORWARD``: unanchored beam search up from the empty set, width
            ``config.beam_width``. Stops at the depth cap or when a step's gain falls below
            ``config.greedy_min_relative_gain``; ``result.marginal_gain_curve`` shows what each
            feature bought.
            ``EXHAUSTIVE``: all non-empty subsets up to ``max_subset_size``. ``2**n - 1`` fits --
            tractable only on small pools.
        baseline : list[str], optional
            Under ``ABLATION`` this is the ANCHOR: it seeds every subset, so it is a prior on the
            result (defaults to all surviving candidates). Under ``GREEDY_FORWARD`` the same
            argument is a BENCHMARK: fitted once for comparison, it seeds nothing, and omitting it
            simply means the run reports no benchmark. Ignored by ``EXHAUSTIVE``.
        max_subset_size : int, optional
            Cap for ``EXHAUSTIVE`` (defaults to ``len(candidate_features)``) and depth cap for
            ``GREEDY_FORWARD`` (defaults to ``config.greedy_max_depth``).
        """
        df = self.feature_engineering(self.raw_data.copy())
        missing = [c for c in candidate_features if c not in df.columns]
        if missing:
            raise ValueError(f"feature_engineering did not produce columns: {missing}")

        # Stage 0: variance pre-filter on the candidate columns. Re-use the ML screener's helper.
        candidate_df = df[list(candidate_features)].dropna(how="all")
        filtered_X, stage0_report = stage0_variance_filter(candidate_df, verbose=False)
        # Stage 0b: marginal-persistence diagnostic (removal opt-in via config.min_persistence, default
        # 0.0 = report-only). Variance asks "does it move?"; persistence asks "does its movement have
        # structure?". When enabled it runs BEFORE subset generation so a memoryless feature never
        # reaches a fit. It is per-feature and marginal, so it cannot see covariance or tail-shape
        # regimes the joint emission can use -- which is why it does not remove by default.
        filtered_X, persistence_report = stage0b_persistence_filter(
            filtered_X, min_persistence=self.config.min_persistence, verbose=False)
        # Stage 0c: set-wise collinearity (removal opt-in via config.max_vif, default inf = report-only).
        # Variance asks "does it move?"; persistence "does its movement have structure?"; collinearity
        # "does its structure duplicate another's?". Runs BEFORE subset generation so a near-duplicate
        # never ill-conditions a fit -- and so every subset ablation builds is non-redundant by
        # construction. Like 0b it is opt-in: VIF is static and linear, blind to regime-switching
        # covariance the joint emission uses, so it must stay a high near-duplicate floor (see _collinearity.py).
        filtered_X, collinearity_report = stage0c_collinearity_filter(
            filtered_X, max_vif=self.config.max_vif, verbose=False)
        surviving = list(filtered_X.columns)

        # A hand-supplied baseline is privileged: under ABLATION every add-one subset is
        # ``baseline + candidate``, so a bad baseline feature contaminates almost the whole search
        # space. ``_generate_subsets`` intersects the baseline with ``surviving``, so a rejected
        # feature cannot sneak back in that way -- but the caller asserted it was baseline-worthy and
        # the data disagreed, which they should hear about rather than have silently corrected.
        if baseline is not None:
            dropped = [c for c in baseline if c in candidate_features and c not in surviving]
            if dropped:
                warnings.warn(
                    f"Baseline features removed by the stage-0 pre-filters and excluded from every "
                    f"subset: {dropped}. They did not earn their place (near-constant, or no persistent "
                    f"structure in any moment). See result.stage_reports for which filter rejected them.",
                    UserWarning, stacklevel=2)

        effective_baseline = self._effective_baseline(surviving, strategy, baseline)
        trace: tuple[GreedyStep, ...] = ()
        stop_reason: GreedyStopReason | None = None
        if strategy == ScreenStrategy.GREEDY_FORWARD:
            # Adaptive: each step's candidates depend on the previous step's fits, so generation
            # cannot be hoisted out of the fit loop the way ABLATION/EXHAUSTIVE do it.
            evaluations, trace, stop_reason = self._greedy_forward(
                surviving, df, max_subset_size, effective_baseline)
        else:
            subsets = self._generate_subsets(surviving, strategy, baseline, max_subset_size)
            if not subsets:
                raise ValueError("No subsets to screen. Check candidate_features and strategy.")
            evaluations = [self._evaluate_subset(subset, df) for subset in subsets]
        if not evaluations:
            raise ValueError("No subsets to screen. Check candidate_features and strategy.")

        statuses = self._classify(evaluations)
        evaluations = [self._with_status(ev, status) for ev, status in zip(evaluations, statuses)]

        results_df = self._build_results_df(evaluations, effective_baseline)
        return HmmScreenerResult(
            evaluations=evaluations,
            results_=results_df,
            stage_reports=[stage0_report, persistence_report, collinearity_report],
            baseline=effective_baseline,
            strategy=strategy,
            search_trace=trace,
            stop_reason=stop_reason,
        )

    # -------------------------------------------------------- subset generation

    @staticmethod
    def _effective_baseline(surviving: list[str], strategy: ScreenStrategy,
                            baseline: list[str] | None) -> tuple[str, ...] | None:
        """The anchor ABLATION actually uses, after intersecting the request with the survivors.

        Single source of truth for both subset generation and the ``BaselinePrior`` audit, so the
        prior can never be measured against a baseline different from the one that was screened.
        Returns ``None`` for strategies that carry no anchor.
        """
        if not surviving:
            return None
        if strategy == ScreenStrategy.GREEDY_FORWARD:
            # Benchmark, not an anchor: it is fitted once so the unanchored winner can be scored
            # against the frozen set it replaces, but it seeds nothing. Absent means no benchmark;
            # unlike ABLATION there is no "default to everything" fallback, because there is no
            # subset the search needs to start from.
            if baseline is None:
                return None
            benchmark = tuple(c for c in baseline if c in surviving)
            return benchmark or None
        if strategy != ScreenStrategy.ABLATION:
            return None
        requested = list(surviving) if baseline is None else baseline
        baseline_set = tuple(c for c in requested if c in surviving)
        if not baseline_set:
            raise ValueError(f"baseline {baseline} has no overlap with surviving candidates {surviving}.")
        return baseline_set

    def _rank_for_beam(self, evaluations: list[SubsetEvaluation]
                       ) -> tuple[list[SubsetEvaluation], bool]:
        """Order candidates for beam selection: axis_separation among quality-gated subsets.

        Returns ``(ranked, used_trap_fallback)``. The flag is the caller's only signal that the
        step ranked on trapped subsets, which is otherwise invisible in a rising gain curve.

        Both gates are ABSOLUTE per-subset -- the Phase-A thresholds and ``honesty_trap_rate`` are
        config constants, not population statistics like the Pareto frontier. A subset's
        eligibility therefore does not shift as the search grows, which is what makes the beam
        reproducible and the trace replayable.

        Trap exclusion is the load-bearing part. Ranking on raw ``axis_separation`` would walk the
        search straight into the confidence-trap quadrant, which is where most high-separation
        subsets live; the fallback to merely non-fragile fires only when EVERY candidate at a step
        is a trap, so a trapped axis still returns a best-effort path instead of stopping dead.
        """
        healthy = [e for e in evaluations
                   if e.error is None and np.isfinite(e.axis_separation) and np.isfinite(e.honesty)
                   and not self._is_fragile(e)]
        non_trap = [e for e in healthy if e.honesty <= self.config.honesty_trap_rate]
        pool = non_trap or healthy
        return sorted(pool, key=lambda e: e.axis_separation, reverse=True), not non_trap

    def _greedy_forward(self, surviving: list[str], df: pd.DataFrame, max_subset_size: int | None,
                        benchmark: tuple[str, ...] | None
                        ) -> tuple[list[SubsetEvaluation], tuple[GreedyStep, ...], GreedyStopReason]:
        """Beam search up from the empty set. Returns (all evaluations, per-step trace, stop reason).

        Every subset fitted along the way is returned, not just the beam -- the frontier is the
        audit trail, and dropping the non-beam fits would make the result look like a far narrower
        search than the one actually paid for.
        """
        depth_cap = max_subset_size if max_subset_size is not None else self.config.greedy_max_depth
        depth_cap = min(depth_cap, len(surviving))
        evaluated: dict[tuple[str, ...], SubsetEvaluation] = {}

        def _fit(key: tuple[str, ...]) -> SubsetEvaluation:
            if key not in evaluated:
                evaluated[key] = self._evaluate_subset(key, df)
            return evaluated[key]

        trace: list[GreedyStep] = []
        beam: tuple[tuple[str, ...], ...] = ()
        best_sep = 0.0
        stop_reason = GreedyStopReason.DEPTH_CAP
        for step in range(1, depth_cap + 1):
            if step == 1:
                candidates = [(c,) for c in surviving]
            else:
                seen: set[tuple[str, ...]] = set()
                candidates = []
                for path in beam:
                    for col in surviving:
                        if col in path:
                            continue
                        key = tuple(sorted(path + (col,)))
                        # Two beam paths routinely propose the same superset; fit it once.
                        if key in seen or key in evaluated:
                            continue
                        seen.add(key)
                        candidates.append(key)
            if not candidates:
                stop_reason = GreedyStopReason.POOL_EXHAUSTED
                break

            step_evals = [_fit(key) for key in candidates]
            ranked, trap_fallback = self._rank_for_beam(step_evals)
            if not ranked:
                # Every candidate at this depth was structurally degenerate or errored.
                stop_reason = GreedyStopReason.NO_ELIGIBLE_CANDIDATES
                break

            step_best = ranked[0]
            gain = step_best.axis_separation - best_sep
            # Step 1 has no predecessor, so its gain is unbounded by construction and can never
            # trip the min-gain stop -- the search always fits at least one full step.
            relative_gain = (gain / best_sep) if best_sep > 0 else float("inf")
            beam = tuple(tuple(e.features) for e in ranked[: self.config.beam_width])
            trace.append(GreedyStep(
                step=step, n_fitted=len(candidates), beam=beam,
                best_subset=tuple(step_best.features), best_separation=step_best.axis_separation,
                gain=gain, relative_gain=relative_gain, used_trap_fallback=trap_fallback,
            ))
            best_sep = max(best_sep, step_best.axis_separation)
            if relative_gain < self.config.greedy_min_relative_gain:
                # Includes the negative-gain case: adding a feature made the best path worse.
                stop_reason = GreedyStopReason.MIN_GAIN
                break

        # The benchmark seeds nothing, but it must be fitted or `base_frac` has no numerator.
        if benchmark:
            _fit(tuple(sorted(benchmark)))
        return list(evaluated.values()), tuple(trace), stop_reason

    def _generate_subsets(self, surviving: list[str], strategy: ScreenStrategy,
                         baseline: list[str] | None, max_subset_size: int | None) -> list[tuple[str, ...]]:
        if not surviving:
            return []
        if strategy == ScreenStrategy.EXHAUSTIVE:
            cap = max_subset_size if max_subset_size is not None else len(surviving)
            out: list[tuple[str, ...]] = []
            for size in range(1, cap + 1):
                for combo in combinations(surviving, size):
                    out.append(combo)
            return out

        # ABLATION: baseline + drop-one (for each in baseline) + add-one (for each not in baseline).
        baseline_set = list(self._effective_baseline(surviving, strategy, baseline))
        seen: set[tuple[str, ...]] = set()
        out: list[tuple[str, ...]] = []

        def _add(subset: list[str]) -> None:
            key = tuple(sorted(subset))
            if key not in seen and len(key) >= 1:
                seen.add(key)
                out.append(key)

        _add(baseline_set)
        for col in baseline_set:
            _add([c for c in baseline_set if c != col])
        for col in surviving:
            if col not in baseline_set:
                _add(baseline_set + [col])
        return out

    # ----------------------------------------------------------- per-subset fit

    def _evaluate_subset(self, subset: tuple[str, ...], df: pd.DataFrame) -> SubsetEvaluation:
        t0 = time.time()
        # Coherence check raises immediately if config.raise_on_off_axis is set;
        # otherwise the offending feature names accumulate as warnings on this subset.
        per_subset_warnings = self._validate_subset_coherence(subset)
        try:
            # Join engineered features to raw OHLC on index; drop rows where any of
            # the required columns is NaN. This guarantees evaluators receive the
            # OHLC they need regardless of what feature_engineering preserves.
            feature_block = df[list(subset)]
            # Avoid duplicate column names when a candidate feature shares a name with a
            # reserved passthrough column (e.g. 'close', 'tick_volume'). The engineered
            # version wins — that's the user's explicit feature. The raw passthrough
            # column is dropped from the join. A per-subset warning surfaces the override
            # so it's never silent.
            colliding = [c for c in subset if c in self._passthrough.columns]
            if colliding:
                per_subset_warnings.append(
                    f"candidate feature(s) {colliding} collide with reserved passthrough "
                    f"column names; using engineered values, raw passthrough is dropped"
                )
            non_colliding_passthrough_cols = [c for c in self._passthrough.columns if c not in subset]
            joined = pd.concat(
                [feature_block, self._passthrough[non_colliding_passthrough_cols]],
                axis=1, join="inner",
            ).dropna()
            if len(joined) < self.config.n_states * 30:
                raise ValueError(
                    f"too few rows after join+dropna ({len(joined)}) for n_states={self.config.n_states}."
                )
            X = joined[list(subset)].values

            model = build_hmm(self.config.algo, self.config.n_states,
                              self.config.mm_n_components, self.config.random_state)
            model.fit(X)

            model.inference_mode = InferenceMode.FILTERING
            gamma = model.predict_proba(X)
            # Use causal MAP labels (argmax of filtering gamma) — not offline Viterbi —
            # so axis quality is scored on the same posterior a live system would act on.
            state_labels = np.argmax(gamma, axis=1)

            tp = top_prob(gamma)
            honesty = float((tp > self.config.honesty_threshold).mean())
            balance_ratio = self._state_balance_ratio(state_labels, self.config.n_states)
            axis_eval = self._call_evaluator(subset, gamma, state_labels, joined)

            return SubsetEvaluation(
                features=subset,
                n_features=len(subset),
                axis_separation=axis_eval.axis_separation,
                secondary_robustness=axis_eval.secondary_robustness,
                secondary_label=axis_eval.secondary_label,
                honesty=honesty,
                state_balance_ratio=balance_ratio,
                pareto_status=ParetoStatus.DOMINATED,  # placeholder; set in _classify
                axis_separation_range=axis_eval.axis_separation_range,
                warnings=tuple(per_subset_warnings),
                raw_details=axis_eval.raw_details,
                elapsed_sec=float(time.time() - t0),
                error=None,
            )
        except Exception as exc:  # capture per-subset failure, continue
            per_subset_warnings.append(f"evaluation_error: {type(exc).__name__}: {exc}")
            return SubsetEvaluation(
                features=subset,
                n_features=len(subset),
                axis_separation=float("nan"),
                secondary_robustness=float("nan"),
                secondary_label="error",
                honesty=float("nan"),
                state_balance_ratio=float("nan"),
                pareto_status=ParetoStatus.DOMINATED,
                axis_separation_range=float("nan"),
                warnings=tuple(per_subset_warnings),
                raw_details={"traceback": traceback.format_exc()},
                elapsed_sec=float(time.time() - t0),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _call_evaluator(self, subset: tuple[str, ...], gamma: np.ndarray,
                       state_labels: np.ndarray, evaluator_df: pd.DataFrame) -> AxisEvaluation:
        evaluator = get_evaluator(self.config.signal_type)
        kwargs: dict = {"respect_sessions": self.config.respect_session_boundaries}
        if self.config.signal_type == "momentum":
            kwargs["is_directional"] = self._infer_is_directional(subset)
        return evaluator(gamma=gamma, state_labels=state_labels, raw_data=evaluator_df,
                         horizons=self.config.horizons, **kwargs)

    def _infer_is_directional(self, subset: tuple[str, ...]) -> bool:
        """Read the registry: if >=50% of subset features are directional, treat as directional axis."""
        flags = []
        for name in subset:
            try:
                entry = self.registry.get(name)
                flags.append(entry.directional)
            except (KeyError, ValueError):
                continue
        if not flags:
            return True  # default to directional when nothing is registered
        return (sum(flags) / len(flags)) >= 0.5

    # ----------------------------------------------------------- axis coherence

    def _validate_subset_coherence(self, subset: tuple[str, ...]) -> list[str]:
        """Per-subset off-axis check.

        Iterates only over features actually in ``subset`` so the resulting warnings name the contaminating feature,
        not the whole candidate pool.
        Raises ``ValueError`` immediately if ``config.raise_on_off_axis`` is set.
        """
        warns: list[str] = []
        for f in subset:
            try:
                entry = self.registry.get(f)
            except (KeyError, ValueError) as e:
                warns.append(f"'{f}' not in FeatureRegistry: {type(e).__name__}; skipping coherence check")
                continue
            if entry.signal_type not in self._allowed_signal_types:
                msg = (f"'{f}' has signal_type='{entry.signal_type}', not in allowed "
                       f"signal_types={sorted(self._allowed_signal_types)}")
                warns.append(msg)
                if self.config.raise_on_off_axis:
                    raise ValueError(msg)
        return warns

    # --------------------------------------------------------- helpers / output

    @staticmethod
    def _state_balance_ratio(state_labels: np.ndarray, n_states: int) -> float:
        """Return ``max_state_pop / min_state_pop`` over ``n_states`` configured states.

        Returns ``+inf`` if any configured state has zero population — that's the smoking-gun for state collapse and the
        Phase-A gate treats it as FRAGILE.
        """
        counts = pd.Series(state_labels).value_counts()
        if len(counts) < n_states or counts.min() == 0:
            return float("inf")
        return float(counts.max() / counts.min())

    def _is_fragile(self, ev: SubsetEvaluation) -> bool:
        """Phase-A gate: structural degeneracy that should pre-empt Pareto comparison."""
        if ev.error is not None:
            return False  # errors are routed separately to DOMINATED
        if not np.isfinite(ev.state_balance_ratio):
            return True
        if ev.state_balance_ratio > self.config.max_balance_ratio:
            return True
        if not np.isfinite(ev.secondary_robustness):
            return True
        if ev.secondary_robustness < self.config.min_significant_states:
            return True
        return False

    def _classify(self, evaluations: list[SubsetEvaluation]) -> list[ParetoStatus]:
        statuses: list[ParetoStatus] = [ParetoStatus.DOMINATED] * len(evaluations)
        # Phase A: structural gate. Errored subsets stay DOMINATED; fragile ones are tagged FRAGILE.
        for_pareto: list[int] = []
        for i, ev in enumerate(evaluations):
            if ev.error is not None or not np.isfinite(ev.axis_separation) or not np.isfinite(ev.honesty):
                statuses[i] = ParetoStatus.DOMINATED
                continue
            if self._is_fragile(ev):
                statuses[i] = ParetoStatus.FRAGILE
                continue
            for_pareto.append(i)

        # Phase B: Pareto check on healthy subsets only.
        measurements = [(evaluations[i].axis_separation, evaluations[i].honesty) for i in for_pareto]
        pareto_statuses = classify_pareto(measurements, self.config.honesty_trap_rate)
        for idx, status in zip(for_pareto, pareto_statuses):
            statuses[idx] = status
        return statuses

    @staticmethod
    def _with_status(ev: SubsetEvaluation, status: ParetoStatus) -> SubsetEvaluation:
        # SubsetEvaluation is frozen; rebuild with the post-classification status.
        # The placeholder-then-rebuild pattern is intentional: it keeps the dataclass
        # immutable while still letting the classifier own the final status decision.
        return SubsetEvaluation(
            features=ev.features, n_features=ev.n_features,
            axis_separation=ev.axis_separation,
            secondary_robustness=ev.secondary_robustness,
            secondary_label=ev.secondary_label,
            honesty=ev.honesty, state_balance_ratio=ev.state_balance_ratio,
            pareto_status=status,
            axis_separation_range=ev.axis_separation_range,
            warnings=ev.warnings, raw_details=ev.raw_details,
            elapsed_sec=ev.elapsed_sec, error=ev.error,
        )

    @staticmethod
    def _build_results_df(evaluations: list[SubsetEvaluation],
                          baseline: tuple[str, ...] | None = None) -> pd.DataFrame:
        rows = []
        # Under ABLATION all but the drop-one rows are supersets of the anchor, so `contains_baseline`
        # is what lets a reader see at a glance how little of the frontier is anchor-independent.
        baseline_set = set(baseline or ())
        for i, ev in enumerate(evaluations):
            rows.append({
                "subset_id": i,
                "features": ",".join(ev.features),
                "n_features": ev.n_features,
                "contains_baseline": bool(baseline_set) and baseline_set.issubset(ev.features),
                "axis_separation": ev.axis_separation,
                "axis_separation_range": ev.axis_separation_range,
                "secondary_robustness": ev.secondary_robustness,
                "secondary_label": ev.secondary_label,
                "honesty": ev.honesty,
                "state_balance_ratio": ev.state_balance_ratio,
                "pareto_status": ev.pareto_status.value,
                "warnings": "; ".join(ev.warnings),
                "error": ev.error or "",
                "elapsed_sec": ev.elapsed_sec,
            })
        df = pd.DataFrame(rows)
        status_order = {"asymmetry_candidate": 0, "trap": 1, "fragile": 2, "dominated": 3}
        df["_status_rank"] = df["pareto_status"].map(status_order).fillna(99)
        df = (df.sort_values(["_status_rank", "axis_separation"], ascending=[True, False])
                .drop(columns=["_status_rank"])
                .reset_index(drop=True))
        return df
