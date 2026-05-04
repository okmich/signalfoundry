"""
LeakageDiagnostics — meta-diagnostic on completed ScreenerResult runs.

This module owns the *cheap* leakage diagnostics: rank-correlation comparison of two screener runs (full vs pruned) on
the intersection of their surviving features, plus an opt-in *expensive* interaction-SHAP probe.

Design decision: the caller supplies both runs. LeakageDiagnostics never constructs a FeatureScreener internally — that
would re-couple the two classes the architecture deliberately separates.

Purity: all helper methods are referentially transparent (return values, no side effects). The notes list is built up by
the orchestrating method only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder

from .._result import ScreenerResult
from ._report import LeakageReport, Severity, classify
from ._sampling import SamplingTask, stratified_row_sample
from ._suspects import SuspectRegistry, resolve


# ── internal value types ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class _RankCorrResult:
    correlations: dict[str, float]
    rank_full_shap: pd.Series
    rank_other_shap: pd.Series
    rank_full_mda: pd.Series
    rank_other_mda: pd.Series
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _InteractionResult:
    interactions: pd.DataFrame
    notes: tuple[str, ...]


# ── main class ───────────────────────────────────────────────────────────────

class LeakageDiagnostics:
    """
    Post-hoc leakage analysis for a completed ScreenerResult.

    Usage
    -----
        full_run   = screener.screen_for_returns(X, y, horizon=5)
        pruned_run = screener.screen_for_returns(X.drop(columns=suspect_cols), y, horizon=5)

        diag   = LeakageDiagnostics(full_run, SuspectRegistry(prefixes=("tm_vol_",)))
        report = diag.compare_runs(pruned_run, label="vs_pruned")
        print(report)

    Parameters
    ----------
    full_run : ScreenerResult
        The screener run including suspect features.
    suspects : SuspectRegistry
        Analyst declaration of which features are suspect.
    """

    def __init__(self, full_run: ScreenerResult, suspects: SuspectRegistry):
        self.full_run = full_run
        self.suspects = suspects
        self._resolved = resolve(suspects, full_run)

        # Cache rank vectors and per-feature cluster lookup — they don't change
        # after construction and may be hit multiple times by repeated probe calls.
        self._full_shap_rank = full_run.shap_rank.rank(ascending=False)
        self._full_mda_rank  = full_run.mda_rank.rank(ascending=False)

        cluster_id: dict[str, int] = {}
        cluster_size: dict[str, int] = {}
        for cid, members in full_run.cluster_assignments.items():
            for m in members:
                cluster_id[m]   = int(cid)
                cluster_size[m] = len(members)
        self._partner_cluster_id   = cluster_id
        self._partner_cluster_size = cluster_size

        # Cached XGB model reused across probe() calls with the same feature_set.
        # Keyed by (task, tuple(sorted(feature_set))).
        self._model_cache: dict[tuple[str, tuple[str, ...]], object] = {}

    # ── module-level convenience ──────────────────────────────────────────────

    @classmethod
    def compare(cls, full: ScreenerResult, pruned: ScreenerResult, suspects: SuspectRegistry,
                label: str = "vs_pruned",
                thresholds: tuple[float, float] = (0.85, 0.70)
                ) -> tuple["LeakageDiagnostics", LeakageReport]:
        """
        One-call helper: build a diagnostics instance and run compare_runs.

        Returns ``(diagnostics, report)`` so the caller can immediately call ``diagnostics.probe(...)`` without rebuilding state.
        """
        diag = cls(full, suspects)
        return diag, diag.compare_runs(pruned, label=label, thresholds=thresholds)

    # ── compare_runs ──────────────────────────────────────────────────────────

    def compare_runs(self, other: ScreenerResult, label: str = "vs_pruned",
                     thresholds: tuple[float, float] = (0.85, 0.70),
                     top_movers_n: int = 20) -> LeakageReport:
        """
        Compare the full run against ``other`` (typically the pruned run).

        Computes Spearman rank correlation on the **intersection of features that received a SHAP/MDA score in both runs**.
        A feature missing from either ranking is excluded from the correlation but recorded as a large mover (``shap_abs_shift = +inf``).

        Parameters
        ----------
        other : ScreenerResult
            The comparison run. Typically produced by re-running the screener with the suspect family removed pre-Stage 0.
        label : str
            Free-text label for the comparison; surfaced in the report.
        thresholds : (high, low)
            Severity thresholds. Defaults (0.85, 0.70).
        top_movers_n : int
            Number of features to include in the top_movers table.

        Returns
        -------
        LeakageReport
            With ``suspect_interactions=None`` — that field is filled in by
            ``probe``.
        """
        notes: list[str] = []

        rank_corr = self._compute_rank_correlations(other)
        notes.extend(rank_corr.notes)

        top_movers = self._build_top_movers(
            rank_corr.rank_full_shap, rank_corr.rank_other_shap,
            rank_corr.rank_full_mda,  rank_corr.rank_other_mda,
            top_movers_n,
        )

        severity = classify(rank_corr.correlations, thresholds=thresholds)
        notes.append(self._severity_note(severity))

        if self._resolved.cluster_inherited_suspects:
            notes.append(
                f"{len(self._resolved.cluster_inherited_suspects)} cluster representative(s) "
                f"inherited a suspect's signal: {self._resolved.cluster_inherited_suspects}"
            )

        return LeakageReport(
            suspect_registry=self.suspects, label=label,
            cluster_lineage=self._resolved.cluster_lineage,
            rank_correlations=rank_corr.correlations,
            top_movers=top_movers, severity=severity,
            notes=tuple(notes), suspect_interactions=None,
        )

    @staticmethod
    def _severity_note(severity: Severity) -> str:
        if severity == Severity.CLEAN:
            return "Rank correlations clear the upper threshold — interaction probe not required."
        if severity == Severity.WATCH:
            return "Rank correlations in the watch band — consider running probe() for interaction SHAP."
        return "Rank correlations below lower threshold — probe() with interactions strongly recommended."

    # ── probe ─────────────────────────────────────────────────────────────────

    def probe(self, X: pd.DataFrame, y: pd.Series, task: SamplingTask | str,
              pruned_run: ScreenerResult | None = None, label: str = "vs_pruned",
              interaction_rows: int = 5000, tail_oversample: float = 2.0,
              top_k_pairs: int = 30, force_interactions: bool = False,
              thresholds: tuple[float, float] = (0.85, 0.70)) -> LeakageReport:
        """
        Run the full leakage probe: cheap rank-correlation diagnostic plus —
        when warranted — strategic-pair interaction SHAP.

        Cost model
        ----------
        - Rank correlation (when ``pruned_run`` provided): O(n_features), milliseconds.
        - Interaction SHAP: one XGBoost fit on the confirmed feature set, then
          ``TreeExplainer.shap_interaction_values`` on a stratified row subsample
          (``interaction_rows`` rows). Cost ~ O(n_rows × n_confirmed²). At 5k
          rows × 60 features this is seconds, not minutes; at 50k rows × 200
          features it is much heavier — keep ``interaction_rows`` modest.

        When the interaction step runs
        -------------------------------
        - ``force_interactions=True``: always.
        - Otherwise: only when severity from ``compare_runs`` is WATCH or
          INVESTIGATE. CLEAN runs skip interactions; the probe is a no-op
          beyond the cheap diagnostic.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix (the same one passed to the screener for the
            full run). Used to extract the confirmed-set columns and the
            stratified subsample.
        y : pd.Series
            Label series (regime labels or forward returns).
        task : SamplingTask or {"regime", "return"}
            Drives the model choice (XGBClassifier/Regressor) and the
            stratification scheme.
        pruned_run : ScreenerResult or None
            Comparison run with suspects removed. When provided, drives the
            rank-correlation diagnostic and severity. When None, the cheap
            diagnostic is skipped and ``force_interactions`` must be True
            for any work to happen.
        label : str
            Free-text label for the comparison.
        interaction_rows : int
            Stratified row subsample size for SHAP interactions. 0 disables
            interactions entirely. Default 5000.
        tail_oversample : float
            Tail-decile multiplier for return-task stratification. Ignored
            for regime task. Default 2.0.
        top_k_pairs : int
            Number of (suspect, partner) pairs to keep in the report. Default 30.
        force_interactions : bool
            Compute interactions regardless of severity. Default False.
        thresholds : (high, low)
            Severity thresholds.

        Returns
        -------
        LeakageReport
        """
        task = SamplingTask(task)

        # Cheap diagnostic first — drives severity and gates the expensive step
        if pruned_run is not None:
            base = self.compare_runs(pruned_run, label=label, thresholds=thresholds)
        else:
            base = LeakageReport(
                suspect_registry=self.suspects, label=label,
                cluster_lineage=self._resolved.cluster_lineage,
                rank_correlations={}, top_movers=pd.DataFrame(),
                severity=Severity.INVESTIGATE,
                notes=("No pruned_run supplied — cheap diagnostic skipped.",),
            )

        if interaction_rows == 0:
            return base.with_notes(("Interactions skipped: interaction_rows=0.",))
        if base.severity == Severity.CLEAN and not force_interactions:
            return base.with_notes(
                ("Interactions skipped: severity is CLEAN. Pass force_interactions=True to override.",)
            )

        targets = self._select_interaction_targets(X)
        if targets is None:
            return base.with_notes(("Interaction probe skipped: no targets.",))
        suspect_targets, confirmed_partners, skip_reason = targets
        if skip_reason is not None:
            return base.with_notes((skip_reason,))

        result = self._compute_interactions(
            X=X, y=y, task=task,
            suspect_targets=suspect_targets, confirmed_partners=confirmed_partners,
            interaction_rows=interaction_rows, tail_oversample=tail_oversample, top_k=top_k_pairs,
        )
        # Promote NaN-rank partners to a note (B4)
        nan_rank_note: tuple[str, ...] = ()
        if not result.interactions.empty:
            nan_partners = result.interactions[
                result.interactions["partner_marginal_shap_rank"].isna()
            ]["partner"].unique().tolist()
            if nan_partners:
                nan_rank_note = (f"Partners with undefined marginal SHAP rank "
                                 f"(check Stage 5 fit): {nan_partners}",)
        return base.with_interactions(
            result.interactions if not result.interactions.empty else None,
            extra_notes=result.notes + nan_rank_note,
        )

    # ── interaction-target selection ──────────────────────────────────────────

    def _select_interaction_targets(self, X: pd.DataFrame
                                    ) -> tuple[list[str], list[str], str | None] | None:
        """
        Return ``(suspect_targets, confirmed_partners, skip_reason)`` after
        intersecting with X.columns. ``skip_reason`` non-None signals an
        early exit with a note. Returns None only on impossible empty universes.
        """
        present = set(X.columns)
        suspect_universe = set(self._resolved.direct_suspects) | set(self._resolved.cluster_inherited_suspects)
        suspect_targets = sorted(s for s in suspect_universe if s in present)
        confirmed_partners = [c for c in self.full_run.confirmed
                              if c not in set(self._resolved.direct_suspects) and c in present]

        if not suspect_universe:
            return [], [], "No suspects (direct or inherited) present in the full run — interaction probe skipped."
        if not [c for c in self.full_run.confirmed if c not in set(self._resolved.direct_suspects)]:
            return [], [], "No confirmed non-suspect partners — interaction probe has nothing to test against."
        if not suspect_targets or not confirmed_partners:
            return [], [], "Interaction targets missing from X — probe skipped."
        return suspect_targets, confirmed_partners, None

    # ── interaction computation ───────────────────────────────────────────────

    def _compute_interactions(self, *, X: pd.DataFrame, y: pd.Series, task: SamplingTask,
                              suspect_targets: list[str], confirmed_partners: list[str],
                              interaction_rows: int, tail_oversample: float,
                              top_k: int) -> _InteractionResult:
        """
        Fit a single XGBoost on the union of suspects + partners, compute SHAP
        interaction values on a label-stratified subsample, and slice the
        resulting (n_rows, n_features, n_features) tensor down to suspect ×
        confirmed-partner cells.

        Returns the sorted top-k interaction table plus any notes accrued.
        """
        # Lazy import — shap & xgboost are heavy
        import shap

        # Align X and y, drop NaN-label rows (single source of truth for NaN handling)
        common = X.index.intersection(y.index)
        feature_set = sorted(set(suspect_targets) | set(confirmed_partners))
        X_local = X.loc[common, feature_set]
        y_local = y.loc[common]
        valid_y = y_local.notna()
        X_local = X_local[valid_y]
        y_local = y_local[valid_y]
        # B1: invariant — both sides must have identical positional indexing
        assert len(X_local) == len(y_local), "X/y row alignment broken in _compute_interactions"

        if len(X_local) < 50:
            return _InteractionResult(
                pd.DataFrame(),
                (f"Interaction probe: only {len(X_local)} aligned rows — too few to fit a model. Skipped.",),
            )

        notes_acc: list[str] = []

        # Sampler now contracts NaN-free y; we just dropped NaN-y above
        positions = stratified_row_sample(
            y_local, task=task, n_rows=interaction_rows, tail_oversample=tail_oversample,
        )
        notes_acc.append(
            f"Interaction probe: stratified subsample of {len(positions)} rows "
            f"(task={task.value}, tail_oversample={tail_oversample if task == SamplingTask.RETURN else 'n/a'})."
        )
        if task == SamplingTask.REGIME and tail_oversample != 2.0:
            notes_acc.append("tail_oversample is ignored for regime task.")

        X_filled = X_local.fillna(X_local.median())
        X_sample = X_filled.iloc[positions]

        # Train target encoding
        if task == SamplingTask.REGIME:
            y_train = LabelEncoder().fit_transform(y_local)
        else:
            y_train = y_local.values.astype(float)

        # Cached fit (P2) — keyed by task and feature_set order
        cache_key = (task.value, tuple(feature_set))
        model = self._model_cache.get(cache_key)
        if model is None:
            model = self._build_xgb(task, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_filled.values, y_train)
            self._model_cache[cache_key] = model

        # SHAP interaction values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.TreeExplainer(model)
                interaction = explainer.shap_interaction_values(X_sample.values)
        except Exception as e:
            return _InteractionResult(pd.DataFrame(), tuple(notes_acc) + (
                f"Interaction probe: TreeExplainer.shap_interaction_values failed ({e}). Skipped.",
            ))

        # SHAP returns one of three shapes for the interaction tensor:
        #   - regression:                 ndarray (n_rows, n_features, n_features)
        #   - multiclass (legacy SHAP):   list[ndarray (n_rows, n_features, n_features)] per class
        #   - multiclass (newer SHAP):    ndarray (n_rows, n_features, n_features, n_classes)
        # Collapse all three into a single (n_rows, n_features, n_features) tensor of mean |.|.
        if isinstance(interaction, list):
            interaction_abs = np.mean([np.abs(arr) for arr in interaction], axis=0)
        else:
            arr = np.asarray(interaction)
            if arr.ndim == 4:
                # Average |.| across the class axis (last axis).
                interaction_abs = np.abs(arr).mean(axis=-1)
            elif arr.ndim == 3:
                interaction_abs = np.abs(arr)
            else:
                return _InteractionResult(pd.DataFrame(), tuple(notes_acc) + (
                    f"Interaction probe: unexpected SHAP tensor shape {arr.shape}. Skipped.",
                ))

        # B5: index from the same object SHAP saw
        feat_to_idx = {f: i for i, f in enumerate(X_sample.columns)}
        suspect_set = set(suspect_targets)
        partner_set = [p for p in confirmed_partners if p not in suspect_set]
        suspect_idx = [feat_to_idx[s] for s in suspect_targets]
        partner_idx = [feat_to_idx[p] for p in partner_set]

        # P3: slice first, then mean — only |suspects| × |partners| cells
        sliced = interaction_abs[:, suspect_idx, :][:, :, partner_idx]   # (n_rows, |s|, |p|)
        mean_sliced = sliced.mean(axis=0)                                 # (|s|, |p|)

        # P1: vectorised correlations via DataFrame.corr
        partner_meta = self._partner_metadata(X_local, y_local, suspect_targets, partner_set)

        rows = []
        for si, s in enumerate(suspect_targets):
            for pi, p in enumerate(partner_set):
                rows.append({
                    "suspect":                       s,
                    "partner":                       p,
                    "mean_abs_interaction":          float(mean_sliced[si, pi]),
                    "partner_marginal_shap_rank":    self._full_shap_rank.get(p, float("nan")),
                    "partner_corr_with_suspect":     partner_meta["corr_with_suspect"].get((s, p), float("nan")),
                    "partner_corr_with_label":       partner_meta["corr_with_label"].get(p, float("nan")),
                    "partner_cluster_id":            self._partner_cluster_id.get(p, -1),
                    "partner_cluster_size":          self._partner_cluster_size.get(p, 1),
                })

        if not rows:
            return _InteractionResult(pd.DataFrame(), tuple(notes_acc))

        df = (pd.DataFrame(rows)
                .sort_values("mean_abs_interaction", ascending=False)
                .head(top_k)
                .reset_index(drop=True))
        return _InteractionResult(df, tuple(notes_acc))

    @staticmethod
    def _build_xgb(task: SamplingTask, y_train: np.ndarray):
        """Match Stage 5 hyperparameters so the diagnostic and ranking stay comparable."""
        from xgboost import XGBClassifier, XGBRegressor
        if task == SamplingTask.REGIME:
            n_classes = len(np.unique(y_train))
            obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
            return XGBClassifier(objective=obj, n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, eval_metric="logloss", random_state=42, verbosity=0)
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                            random_state=42, verbosity=0)

    def _partner_metadata(self, X_local: pd.DataFrame, y_local: pd.Series,
                          suspects: list[str], partners: list[str]) -> dict:
        """
        Pre-compute partner-side classification metadata in one BLAS-backed
        Spearman pass. Replaces the previous O(|s|×|p|) per-pair loop (P1).
        """
        # corr_with_label: Spearman = Pearson on ranks. One vectorised call beats
        # per-column scipy.stats.spearmanr.
        partner_ranks = X_local[partners].rank()
        label_rank = y_local.rank()
        corr_with_label_series = partner_ranks.corrwith(label_rank, method="pearson")
        corr_with_label = corr_with_label_series.to_dict()

        # corr_with_suspect: full Spearman matrix on suspects ∪ partners, then slice.
        all_cols = list(dict.fromkeys(list(suspects) + list(partners)))  # preserve order, dedup
        corr_matrix = X_local[all_cols].corr(method="spearman")
        corr_with_suspect: dict[tuple[str, str], float] = {}
        for s in suspects:
            if s not in corr_matrix.index:
                continue
            for p in partners:
                if p not in corr_matrix.columns:
                    corr_with_suspect[(s, p)] = float("nan")
                    continue
                v = corr_matrix.at[s, p]
                corr_with_suspect[(s, p)] = float(v) if not pd.isna(v) else float("nan")

        return {"corr_with_label": corr_with_label, "corr_with_suspect": corr_with_suspect}

    # ── top movers ────────────────────────────────────────────────────────────

    def _compute_rank_correlations(self, other: ScreenerResult) -> _RankCorrResult:
        """
        Compute Spearman correlations between full and other on the
        intersection of features ranked in both runs.
        """
        notes: list[str] = []

        rank_other_shap = other.shap_rank.rank(ascending=False)
        rank_other_mda  = other.mda_rank.rank(ascending=False)

        shap_intersection = self._full_shap_rank.index.intersection(rank_other_shap.index)
        mda_intersection  = self._full_mda_rank.index.intersection(rank_other_mda.index)

        shap_corr = self._spearman_on_intersection(self._full_shap_rank, rank_other_shap, shap_intersection)
        mda_corr  = self._spearman_on_intersection(self._full_mda_rank,  rank_other_mda,  mda_intersection)

        if pd.isna(shap_corr) and len(shap_intersection) < 2:
            notes.append(f"SHAP rank-correlation undefined: intersection has {len(shap_intersection)} feature(s).")
        if pd.isna(mda_corr) and len(mda_intersection) < 2:
            notes.append(f"MDA rank-correlation undefined: intersection has {len(mda_intersection)} feature(s).")

        return _RankCorrResult(
            correlations={"shap_full_vs_pruned": shap_corr, "mda_full_vs_pruned": mda_corr},
            rank_full_shap=self._full_shap_rank, rank_other_shap=rank_other_shap,
            rank_full_mda=self._full_mda_rank,   rank_other_mda=rank_other_mda,
            notes=tuple(notes),
        )

    @staticmethod
    def _spearman_on_intersection(rank_a: pd.Series, rank_b: pd.Series, idx: pd.Index) -> float:
        if len(idx) < 2:
            return float("nan")
        rho, _ = spearmanr(rank_a.loc[idx].values, rank_b.loc[idx].values)
        return float(rho) if not np.isnan(rho) else float("nan")

    def _build_top_movers(self, rank_full_shap: pd.Series, rank_other_shap: pd.Series,
                          rank_full_mda: pd.Series, rank_other_mda: pd.Series,
                          n: int) -> pd.DataFrame:
        """
        Top-N features by absolute SHAP rank shift between the two runs.
        Now also reports the parallel MDA shift (S8) — agreement between
        the two distinguishes interaction-driven leakage (SHAP shifts more)
        from broad reranking (both shift together).

        Vectorised via pandas (P5) instead of a per-feature loop.
        """
        all_features = (rank_full_shap.index.union(rank_other_shap.index)
                                       .union(rank_full_mda.index)
                                       .union(rank_other_mda.index))
        if len(all_features) == 0:
            return pd.DataFrame(
                columns=["feature", "rank_full", "rank_pruned", "shap_abs_shift",
                         "mda_abs_shift", "in_suspect_cluster"],
            )

        # Members of any cluster that contained a suspect (full run only)
        if not self._resolved.cluster_lineage.empty:
            suspect_cluster_features: set[str] = set()
            for members in self._resolved.cluster_lineage["members"]:
                suspect_cluster_features.update(members)
        else:
            suspect_cluster_features = set()

        df = pd.DataFrame(index=all_features)
        df["rank_full"]      = rank_full_shap.reindex(all_features)
        df["rank_pruned"]    = rank_other_shap.reindex(all_features)
        df["shap_abs_shift"] = (df["rank_full"] - df["rank_pruned"]).abs()
        # Disappearance → +inf
        df["shap_abs_shift"] = df["shap_abs_shift"].where(
            df["rank_full"].notna() & df["rank_pruned"].notna(), float("inf"),
        )

        mda_shift = (rank_full_mda.reindex(all_features) - rank_other_mda.reindex(all_features)).abs()
        mda_shift = mda_shift.where(
            rank_full_mda.reindex(all_features).notna() & rank_other_mda.reindex(all_features).notna(),
            float("inf"),
        )
        df["mda_abs_shift"] = mda_shift
        df["in_suspect_cluster"] = df.index.isin(suspect_cluster_features)

        df = (df.sort_values("shap_abs_shift", ascending=False)
                .head(n)
                .reset_index()
                .rename(columns={"index": "feature"}))
        return df
