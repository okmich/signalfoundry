"""
FeatureScreener
===============
Orchestrates the empirical feature selection funnel.

    Stage 0 — Prefix deduplication (tm_* vs feat_*, zero-compute, optional)
    Stage 0 — Variance filter
    Stage 1 — Hygiene (MI + KS for regime; MI + dcor for return)
    Stage 2 — Temporal stability (IC-IR)
    Stage 3 — Redundancy reduction (hierarchical clustering)
    Stage 4 — Boruta (confirmed / tentative / rejected)
    Stage 5 — Model-based ranking (SHAP + permutation importance)

Usage
-----
    from okmich_quant_research.features.screener import FeatureScreener

    screener = FeatureScreener()
    result = screener.screen_for_regimes(X, regime_labels)
    result = screener.screen_for_returns(X, forward_returns, horizon=5)
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ._result import ScreenerResult, StageReport
from ._stage0_dedup import stage0_prefix_dedup
from ._stage0 import stage0_variance_filter
from ._stage1_regime import stage1_regime
from ._stage1_return import stage1_return
from ._stage2 import stage2_temporal_stability
from ._stage3 import stage3_redundancy
from ._stage4 import stage4_boruta
from ._stage5 import stage5_model_ranking


class FeatureScreener:
    """
    End-to-end empirical feature selection funnel.

    This is Layer 2 of the Feature Intelligence system. It should be called
    after Layer 1 (FeatureRegistry) has pre-screened candidates using domain
    knowledge, so that only relevant features enter the expensive empirical stages.

    Parameters
    ----------
    cv_threshold : float
        Stage 0: minimum coefficient of variation. Default 1e-4.
    const_pct_threshold : float
        Stage 0: maximum fraction of bars with the modal value. Default 0.95.
    mi_threshold : float
        Stage 1: minimum mutual information score. Default 0.02.
    ks_threshold : float
        Stage 1 (regime): minimum KS statistic. Default 0.10.
    dcor_threshold : float
        Stage 1 (return): minimum distance correlation. Default 0.05.
    ic_window : int
        Stage 2: rolling window for IC computation. Default 252.
    regime_icir_threshold : float
        Stage 2 (regime): minimum IC-IR. Default 0.20.
    return_icir_threshold : float
        Stage 2 (return): minimum IC-IR. Default 0.30.
    corr_threshold : float
        Stage 3: Spearman |corr| above which features are considered redundant.
        Default 0.75.
    boruta_max_iter : int
        Stage 4: Boruta iterations. More = more reliable. Default 100.
    boruta_perc : int
        Stage 4: percentile of shadow importance to beat. Default 100.
    n_cv_splits : int
        Stage 5: number of CV folds for SHAP/MDA ranking. Default 5.
    walk_forward_pct : float
        Stage 2: minimum fraction of rolling IC windows where IC must be positive
        (return task) or > 0.05 (regime task). Default 0.55.
        Lower this for high-frequency / intraday data where per-window IC is noisy.
    stage1_min_passes : int
        Stage 1 (return): number of tests (MI, dcor) a feature must pass to survive.
        Default 2 (AND logic). Set to 1 for OR logic — useful when vol-normalised
        labels suppress dcor/MI scores below their natural thresholds.
    prefix_dedup : bool
        If True, run a zero-compute prefix deduplication step before Stage 0.
        Removes cross-prefix duplicates (e.g. ``tm_rsi`` vs ``feat_rsi``) by name
        matching, keeping the feature from the highest-priority prefix.
        Default True.
    prefix_priority : tuple[str, ...]
        Ordered list of feature prefixes; leftmost = most preferred.
        Default ``("tm_", "feat_")``.
    use_block_ic : bool
        Stage 2: if True, use non-overlapping block IC — O(n) and much faster than
        the default sliding-window O(n × window). Default False (original behaviour).
    max_samples : int
        If > 0, subsample this many rows (time-stratified) before Stages 1, 4, and 5.
        Stage 2 always uses the full series for temporal coverage.
        Default 0 (disabled — original behaviour).
    verbose : bool
        Print progress at each stage. Default True.
    """

    def __init__(self, cv_threshold: float = 1e-4, const_pct_threshold: float = 0.95, mi_threshold: float = 0.02,
                 ks_threshold: float = 0.10, dcor_threshold: float = 0.05, ic_window: int = 252,
                 regime_icir_threshold: float = 0.20, return_icir_threshold: float = 0.30, corr_threshold: float = 0.75,
                 boruta_max_iter: int = 100, boruta_perc: int = 100, n_cv_splits: int = 5,
                 walk_forward_pct: float = 0.55, stage1_min_passes: int = 2,
                 stage1_mi_pct: float | None = None, stage1_ks_pct: float | None = None,
                 stage1_dcor_pct: float | None = None,
                 stage2_icir_pct: float | None = None, stage2_wf_threshold: float | None = None,
                 prefix_dedup: bool = True, prefix_priority: tuple = ("tm_", "feat_"),
                 use_block_ic: bool = True, max_samples: int = 0, verbose: bool = True):
        self.cv_threshold = cv_threshold
        self.const_pct_threshold = const_pct_threshold
        self.mi_threshold = mi_threshold
        self.ks_threshold = ks_threshold
        self.dcor_threshold = dcor_threshold
        self.ic_window = ic_window
        self.regime_icir_threshold = regime_icir_threshold
        self.return_icir_threshold = return_icir_threshold
        self.corr_threshold = corr_threshold
        self.boruta_max_iter = boruta_max_iter
        self.boruta_perc = boruta_perc
        self.n_cv_splits = n_cv_splits
        self.walk_forward_pct = walk_forward_pct
        self.stage1_min_passes = stage1_min_passes
        self.stage1_mi_pct = stage1_mi_pct
        self.stage1_ks_pct = stage1_ks_pct
        self.stage1_dcor_pct = stage1_dcor_pct
        self.stage2_icir_pct = stage2_icir_pct
        self.stage2_wf_threshold = stage2_wf_threshold
        self.prefix_dedup = prefix_dedup
        self.prefix_priority = prefix_priority
        self.use_block_ic = use_block_ic
        self.max_samples = max_samples
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Subsampling ───────────────────────────────────────────────────────────

    @staticmethod
    def _subsample(X: pd.DataFrame, y: pd.Series, max_samples: int) -> tuple[pd.DataFrame, pd.Series]:
        """
        Time-stratified subsample: draw equally from 10 temporal blocks so
        the subsample covers the full history rather than one market regime.
        """
        import numpy as np
        n = len(X)
        if max_samples <= 0 or n <= max_samples:
            return X, y
        rng = np.random.default_rng(42)
        n_blocks = 10
        per_block = max_samples // n_blocks
        block_size = n // n_blocks
        idx: list[int] = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size if i < n_blocks - 1 else n
            block_idx = np.arange(start, end)
            k = min(per_block, len(block_idx))
            chosen = rng.choice(block_idx, size=k, replace=False)
            idx.extend(chosen.tolist())
        idx = sorted(idx)
        return X.iloc[idx], y.iloc[idx]

    # ── Common pre-processing ─────────────────────────────────────────────────

    @staticmethod
    def _align(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Align X and y safely.

        With duplicate index labels, label-based alignment can explode into
        many-to-many matches and silently create length mismatches. For
        feature screening we only need row-wise pairing, so when duplicates
        are present we fall back to positional alignment.
        """
        if X.index.has_duplicates or y.index.has_duplicates:
            n = min(len(X), len(y))
            return (
                X.iloc[:n].reset_index(drop=True),
                y.iloc[:n].reset_index(drop=True),
            )

        common = X.index.intersection(y.index)
        X_aligned = X.loc[common]
        y_aligned = y.loc[common]

        if len(X_aligned) != len(y_aligned):
            n = min(len(X_aligned), len(y_aligned))
            return (
                X_aligned.iloc[:n].reset_index(drop=True),
                y_aligned.iloc[:n].reset_index(drop=True),
            )

        return X_aligned, y_aligned

    # ── Public API ────────────────────────────────────────────────────────────

    def screen_for_regimes(self, features_df: pd.DataFrame, labels: pd.Series,
                           starting_features: Optional[List[str]] = None) -> ScreenerResult:
        """
        Run the full 5-stage funnel for regime classification.

        Parameters
        ----------
        features_df : pd.DataFrame
            Pre-computed feature matrix (rows = bars, cols = features).
        labels : pd.Series
            Discrete regime labels (integer or string). Must share index with features_df.
        starting_features : list[str] or None
            If provided, subsets features_df to these columns before Stage 0.
            Use the FeatureRegistry to generate this list.

        Returns
        -------
        ScreenerResult
        """
        self._log("\n== FeatureScreener: screen_for_regimes ==")
        return self._run(features_df=features_df, y=labels, task="regime", starting_features=starting_features)

    def screen_for_returns(self, features_df: pd.DataFrame, forward_returns: pd.Series, horizon: int = 5,
                           embargo_pct: float = 0.01, starting_features: Optional[List[str]] = None) -> ScreenerResult:
        """
        Run the full 5-stage funnel for return prediction.

        Parameters
        ----------
        features_df : pd.DataFrame
            Pre-computed feature matrix.
        forward_returns : pd.Series
            Forward log-returns: ``log(close.shift(-horizon) / close)``.
            NaN at the tail is handled automatically.
        horizon : int
            Bars ahead for the label. Used for PurgedKFold purge width.
        embargo_pct : float
            Embargo fraction after each test fold.
        starting_features : list[str] or None

        Returns
        -------
        ScreenerResult
        """
        self._log("\n== FeatureScreener: screen_for_returns ==")
        return self._run(features_df=features_df, y=forward_returns, task="return", starting_features=starting_features,
                         horizon=horizon, embargo_pct=embargo_pct)

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run(self, features_df: pd.DataFrame, y: pd.Series, task: str, starting_features: Optional[List[str]] = None,
             horizon: int = 1, embargo_pct: float = 0.01) -> ScreenerResult:

        # Optional pre-filter from registry
        if starting_features is not None:
            cols = [c for c in starting_features if c in features_df.columns]
            features_df = features_df[cols]
            self._log(f"  Registry pre-filter: using {len(cols)} of {len(starting_features)} requested features")

        X, y = self._align(features_df, y)
        self._log(f"  Input: {X.shape[1]} features x {len(X)} bars")

        # Prepare subsampled view for expensive stages (Stages 1, 4, 5).
        # Stage 2 always uses the full series — it needs temporal coverage.
        if self.max_samples > 0 and len(X) > self.max_samples:
            X_sub, y_sub = self._subsample(X, y, self.max_samples)
            self._log(f"  Subsampling: {len(X)} -> {len(X_sub)} rows for Stages 1, 4, 5")
        else:
            X_sub, y_sub = X, y

        stage_reports: list[StageReport] = []

        if self.prefix_dedup:
            self._log("\n[Stage 0] Prefix deduplication")
            X, r_dedup = stage0_prefix_dedup(X, prefix_priority=self.prefix_priority, verbose=self.verbose)
            X_sub = X_sub[X_sub.columns.intersection(X.columns)]
            stage_reports.append(r_dedup)
            if X.shape[1] == 0:
                return self._empty_result(stage_reports)

        self._log("\n[Stage 0] Near-zero variance filter")
        X, r0 = stage0_variance_filter(X, cv_threshold=self.cv_threshold, const_pct_threshold=self.const_pct_threshold,
                                       verbose=self.verbose)
        X_sub = X_sub[X.columns]
        stage_reports.append(r0)
        if X.shape[1] == 0:
            return self._empty_result(stage_reports)

        self._log(f"\n[Stage 1] Hygiene filter ({task})")
        if task == "regime":
            X_sub, r1, s1_scores = stage1_regime(X_sub, y_sub, mi_threshold=self.mi_threshold,
                                                 ks_threshold=self.ks_threshold,
                                                 mi_pct=self.stage1_mi_pct, ks_pct=self.stage1_ks_pct,
                                                 min_passes=self.stage1_min_passes, verbose=self.verbose)
        else:
            X_sub, r1, s1_scores = stage1_return(X_sub, y_sub, mi_threshold=self.mi_threshold,
                                                 dcor_threshold=self.dcor_threshold,
                                                 mi_pct=self.stage1_mi_pct, dcor_pct=self.stage1_dcor_pct,
                                                 min_passes=self.stage1_min_passes, verbose=self.verbose)
        X = X[X_sub.columns]
        stage_reports.append(r1)
        if X.shape[1] == 0:
            return self._empty_result(stage_reports)

        self._log("\n[Stage 2] Temporal stability (IC-IR)")
        min_icir = self.regime_icir_threshold if task == "regime" else self.return_icir_threshold
        X, r2, icir_scores = stage2_temporal_stability(X, y, task=task, window=self.ic_window, min_icir=min_icir,
                                                       walk_forward_pct=self.walk_forward_pct,
                                                       icir_pct=self.stage2_icir_pct,
                                                       wf_threshold=self.stage2_wf_threshold,
                                                       use_block_ic=self.use_block_ic, verbose=self.verbose)
        X_sub = X_sub[X_sub.columns.intersection(X.columns)]
        stage_reports.append(r2)
        if X.shape[1] == 0:
            return self._empty_result(stage_reports)

        # ── Stage 3 ──────────────────────────────────────────────────────────
        self._log("\n[Stage 3] Redundancy reduction (clustering)")
        X, r3 = stage3_redundancy(X, icir_scores=icir_scores, corr_threshold=self.corr_threshold, verbose=self.verbose)
        X_sub = X_sub[X_sub.columns.intersection(X.columns)]
        stage_reports.append(r3)
        if X.shape[1] == 0:
            return self._empty_result(stage_reports)

        self._log("\n[Stage 4] Boruta")
        X_boruta_sub, r4, boruta_groups = stage4_boruta(X_sub, y_sub, task=task, max_iter=self.boruta_max_iter,
                                                        perc=self.boruta_perc, verbose=self.verbose)
        stage_reports.append(r4)

        confirmed = boruta_groups["confirmed"]
        tentative = boruta_groups["tentative"]
        rejected_total = r0.removed + r1.removed + r2.removed + r3.removed + boruta_groups["rejected"]

        # If Boruta confirmed nothing, fall back to all Stage-3 survivors
        if len(confirmed) + len(tentative) == 0:
            self._log("  Boruta confirmed nothing — keeping all Stage-3 survivors for ranking")
            X_sub_for_ranking = X_sub
            confirmed = list(X.columns)
            tentative = []
        else:
            X_sub_for_ranking = X_boruta_sub

        # ── Stage 5 ──────────────────────────────────────────────────────────
        self._log("\n[Stage 5] Model ranking (SHAP + MDA)")
        shap_rank, mda_rank = stage5_model_ranking(X_sub_for_ranking, y_sub, task=task, n_splits=self.n_cv_splits,
                                                   horizon=horizon, embargo_pct=embargo_pct, verbose=self.verbose)

        self._log(f"\n== Done. Confirmed: {len(confirmed)}, Tentative: {len(tentative)} ==")

        return ScreenerResult(confirmed=confirmed, tentative=tentative, rejected=rejected_total,
                              shap_rank=shap_rank, mda_rank=mda_rank, stage_reports=stage_reports,
                              icir_scores=icir_scores)

    @staticmethod
    def _empty_result(stage_reports: list[StageReport]) -> ScreenerResult:
        """Return an empty result when all features were removed."""
        rejected_total: list[str] = []
        for r in stage_reports:
            rejected_total.extend(r.removed)
        return ScreenerResult(confirmed=[], tentative=[], rejected=rejected_total,
                              shap_rank=pd.Series(dtype=float), mda_rank=pd.Series(dtype=float),
                              stage_reports=stage_reports, icir_scores={})
