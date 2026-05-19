"""
Period-Stack Audit Framework for Multi-Period (Implicit Multi-Timeframe) Feature Exploration

**EXPLORATORY TOOL.** Audit a stack of one feature family computed at multiple periods
to answer: "how much independent information is in this stack, and what is the cleanest
construction to inspect?" The OLS / Gram-Schmidt math in :meth:`PeriodStackAudit.build_variants`
fits coefficients on the full input series — appropriate for characterising the stack
during exploration.

If you later want to apply the same orthogonalisation to a separate window without
re-fitting, an escape hatch is provided: :meth:`PeriodStackAudit.fit_orthogonalisation`
learns coefficients on a window of your choice; :meth:`PeriodStackAudit.transform_orthogonalisation`
applies them to any other window. The audit itself does not impose any IS/OOS split.

This is the in-place equivalent of multiple-timeframe analysis: stacking
``norm_ema(close, period=24/72/200)`` on 5-minute bars is implicitly a 2h/6h/16h
view of the same series. The auditor reports:

- Naive stack pairwise correlations (typically very high; the trap to detect).
- Spread construction: level + alignment + curvature (often partially decoupled).
- Static OLS orthogonalisation against an anchor period (anchor clean, residuals
  may remain coupled).
- Full Gram-Schmidt orthogonalisation (mathematically orthogonal triple).
- Variance retention after orthogonalisation (signal magnitude survives the
  decoupling — too-tight period spacings strip most of the variance and leave
  noise).
- A spacing verdict (TOO_TIGHT / ADEQUATE / WIDE) and a recommended construction.

Empirical anchor (5m EURUSD): adjacent-period ratios of (0.6, 2.0) produce
6%/3% GS variance retention and are TOO_TIGHT for HMM emission separation;
ratios of (0.33, 2.78) produce 25%/14% retention and are ADEQUATE.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# ENUMS
# ============================================================================


class StackVariant(StrEnum):
    """The four feature-construction variants the auditor compares."""

    NAIVE = "naive"
    SPREADS = "spreads"
    OLS_ORTHO = "ols_ortho"
    GRAM_SCHMIDT = "gram_schmidt"


class SpacingVerdict(StrEnum):
    """Verdict on whether the period spacing carries enough independent information."""

    TOO_TIGHT = "too_tight"
    ADEQUATE = "adequate"
    WIDE = "wide"


# ============================================================================
# CORE AUDITOR
# ============================================================================


class PeriodStackAudit:
    """
    Audit a multi-period stack of one feature family for the four standard constructions
    (naive / spreads / static OLS / Gram-Schmidt) and report which one carries usable
    independent information.

    Parameters
    ----------
    series : pd.Series
        Underlying input series (typically close prices) that ``feature_fn`` consumes.
    feature_fn : Callable[[pd.Series, int], pd.Series]
        Function mapping ``(series, period)`` to a feature Series.
        Examples: ``okmich_quant_features.trend.normalized_ma.norm_ema``,
        ``okmich_quant_features.trend.normalized_ma.norm_sma``.
    periods : Tuple[int, int, int]
        Exactly three periods (short, anchor, long). The auditor is specialised to a 3-period
        stack because that is where the level / alignment / curvature semantics map cleanly.
        The anchor (or middle period) is always ``sorted(periods)[1]``.
    bar_seconds : int, default=300
        Seconds per bar — used only to translate periods to human-readable timescales
        (e.g. ``period=72`` on 5m → ``72 * 300 = 21600s = 6h``).
    feature_name : str, default='feature'
        Short name used for output column labels (e.g. 'norm_ema').
    verbose : bool, default=True
        If ``True``, the analysis methods print their reports (matching the
        ``eda.FeatureEDA`` notebook UX). Set ``False`` for tests, batch pipelines,
        and model-selection loops.
    too_tight_short_retention : float, default=0.10
        GS short-residual variance retention below this is classified TOO_TIGHT.
    too_tight_long_retention : float, default=0.05
        GS long-residual variance retention below this is classified TOO_TIGHT.
    wide_short_retention : float, default=0.35
        GS short-residual variance retention above this is classified WIDE.
    wide_long_retention : float, default=0.20
        GS long-residual variance retention above this is classified WIDE.

    Notes
    -----
    The orthogonalisation is Pearson/linear (OLS β solves a least-squares projection).
    Reporting the same Pearson correlation everywhere keeps the orthogonality claim
    consistent with the displayed matrix.
    """

    def __init__(self, series: pd.Series, feature_fn: Callable[[pd.Series, int], pd.Series],
                 periods: Tuple[int, int, int], bar_seconds: int = 300,
                 feature_name: str = "feature", verbose: bool = True,
                 too_tight_short_retention: float = 0.10, too_tight_long_retention: float = 0.05,
                 wide_short_retention: float = 0.35, wide_long_retention: float = 0.20):
        if len(periods) != 3:
            raise ValueError(f"periods must contain exactly 3 values, got {len(periods)}")
        if any(p <= 0 for p in periods):
            raise ValueError(f"periods must be positive, got {periods}")
        if len(set(periods)) != 3:
            raise ValueError(f"periods must be distinct, got {periods}")
        if bar_seconds <= 0:
            raise ValueError(f"bar_seconds must be positive, got {bar_seconds}")

        sorted_periods = tuple(sorted(periods))
        # Anchor is always the median period. Curvature ``f_short − 2·f_anchor + f_long``
        # only encodes a meaningful second-difference if the anchor sits between short and long;
        # allowing min/max anchors would silently produce incoherent "spreads" semantics.
        anchor_period = sorted_periods[1]

        self.series = series.copy()
        self.feature_fn = feature_fn
        self.periods = sorted_periods
        self.anchor_period = anchor_period
        self.short_period = sorted_periods[0]
        self.long_period = sorted_periods[2]
        self.bar_seconds = bar_seconds
        self.feature_name = feature_name
        self.verbose = verbose
        # Exact column name for the anchor — used everywhere that needs to locate the anchor column without scanning by
        # prefix (which can collide if feature_name happens to be a substring of another column label like "alignment"/"curvature").
        self.anchor_col_name = f"{feature_name}_{anchor_period}"
        self._verdict_thresholds = {
            "too_tight_short": too_tight_short_retention,
            "too_tight_long": too_tight_long_retention,
            "wide_short": wide_short_retention,
            "wide_long": wide_long_retention,
        }
        # Minimum number of finite-aligned bars required after dropna in build_variants.
        # Below this, OLS / variance estimates are unreliable.
        self._min_samples = 30

        # Results storage (populated lazily by the analysis methods)
        self.variants: Dict[StackVariant, pd.DataFrame] = {}
        self.correlation_results: Dict[StackVariant, pd.DataFrame] = {}
        self.variance_retention: Dict[str, float] = {}
        self.spacing_results: Dict[str, object] = {}
        self.recommendation: Dict[str, object] = {}
        self._beta_short: Optional[float] = None
        self._beta_long: Optional[float] = None
        self._gamma: Optional[float] = None

    # ========================================================================
    # 1. VARIANT CONSTRUCTION
    # ========================================================================

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str = "") -> None:
        """Gate verbose printing on ``self.verbose``."""
        if self.verbose:
            print(msg)

    @staticmethod
    def _validate_finite_frame(df: pd.DataFrame, context: str, min_samples: int) -> None:
        """Reject empty / inf-containing / too-small / constant-column frames.

        Run after ``dropna()`` on every joined frame before the OLS / variance math
        downstream, so failures surface as clear ``ValueError`` rather than silent
        ``nan`` / ``inf`` leaking into the verdict.
        """
        if df.empty:
            raise ValueError(f"{context}: frame is empty after dropna; check that "
                             "feature_fn produces non-trivial output for these periods.")
        if not np.all(np.isfinite(df.values)):
            raise ValueError(f"{context}: frame contains non-finite values (inf or NaN) "
                             "after dropna; check the feature function for division-by-zero "
                             "or log-of-non-positive paths.")
        if len(df) < min_samples:
            raise ValueError(f"{context}: only {len(df)} rows after dropna, need >= "
                             f"{min_samples}; widen the input series or use shorter periods.")
        zero_var = [c for c in df.columns if df[c].var() == 0.0]
        if zero_var:
            raise ValueError(f"{context}: columns {zero_var} are constant (zero variance); "
                             "cannot compute correlations or OLS coefficients.")

    @staticmethod
    def _ols_beta(x: pd.Series, y: pd.Series) -> float:
        """OLS slope ``cov(x, y) / var(y)`` with consistent ddof=0 across both estimators.

        Mixing ``np.cov``'s default ``ddof=1`` with ``np.var``'s default ``ddof=0``
        introduces a (N-1)/N bias that leaks anchor correlation back into the
        residuals at ~1/N magnitude. ddof=0 throughout gives the unbiased OLS slope.
        """
        return float(np.cov(x, y, ddof=0)[0, 1] / np.var(y))

    # ========================================================================
    # 1. VARIANT CONSTRUCTION (exploratory — full-series OLS fit)
    # ========================================================================

    def build_variants(self) -> Dict[StackVariant, pd.DataFrame]:
        """
        Compute all four candidate feature constructions from the underlying series.

        **Exploratory.** The OLS β and Gram-Schmidt γ are fit on the entire input
        series — appropriate for characterising the stack but not for use as model
        features without separate train/test handling. The escape hatch for taking
        an exploration insight into a train/test workflow is
        :meth:`fit_orthogonalisation` + :meth:`transform_orthogonalisation`.

        Populates ``self.variants`` and returns the same dict for chaining.
        Each variant is a 3-column DataFrame aligned on the same index after ``dropna()``.

        Raises
        ------
        ValueError
            If the joined feature frame is empty, contains inf/NaN, has fewer than
            ``self._min_samples`` rows, or any column has zero variance.
        """
        f_short = self.feature_fn(self.series, self.short_period)
        f_anchor = self.feature_fn(self.series, self.anchor_period)
        f_long = self.feature_fn(self.series, self.long_period)

        col_s = f"{self.feature_name}_{self.short_period}"
        col_a = self.anchor_col_name
        col_l = f"{self.feature_name}_{self.long_period}"

        # Variant 1: naive stack
        naive = pd.DataFrame({col_s: f_short, col_a: f_anchor, col_l: f_long}).dropna()
        self._validate_finite_frame(naive, "build_variants/naive", self._min_samples)

        # Variant 2: spreads — level, alignment, curvature
        spreads = pd.DataFrame({
            col_a: f_anchor,
            "alignment": f_short - f_long,
            "curvature": f_short - 2 * f_anchor + f_long,
        }).dropna()
        self._validate_finite_frame(spreads, "build_variants/spreads", self._min_samples)

        # Common frame for orthogonalisation
        common = pd.DataFrame({"short": f_short, "anchor": f_anchor, "long": f_long}).dropna()
        self._validate_finite_frame(common, "build_variants/orthogonalisation", self._min_samples)
        anchor_vals = common["anchor"]

        # Variant 3: static OLS — each residual orthogonal to anchor.
        self._beta_short = self._ols_beta(common["short"], anchor_vals)
        self._beta_long = self._ols_beta(common["long"], anchor_vals)
        short_resid = common["short"] - self._beta_short * anchor_vals
        long_resid_static = common["long"] - self._beta_long * anchor_vals

        ols_ortho = pd.DataFrame({
            col_a: anchor_vals,
            "short_resid": short_resid,
            "long_resid": long_resid_static,
        })

        # Variant 4: Gram-Schmidt — long_resid additionally orthogonal to short_resid
        if short_resid.var() == 0.0:
            raise ValueError("build_variants/gram_schmidt: short_resid has zero variance "
                             "after OLS step; periods are degenerate.")
        self._gamma = self._ols_beta(long_resid_static, short_resid)
        long_resid_gs = long_resid_static - self._gamma * short_resid

        gram_schmidt = pd.DataFrame({
            col_a: anchor_vals,
            "short_resid": short_resid,
            "long_resid": long_resid_gs,
        })

        self.variants = {
            StackVariant.NAIVE: naive,
            StackVariant.SPREADS: spreads,
            StackVariant.OLS_ORTHO: ols_ortho,
            StackVariant.GRAM_SCHMIDT: gram_schmidt,
        }

        # Variance retention is the central diagnostic for the spacing verdict.
        # Mathematically: short_resid_over_source = Var(short - β·anchor) / Var(short) = 1 - R²(short ~ anchor).
        # Read as "fraction of short-period variance NOT explained by the anchor period" — i.e. the genuinely independent signal
        # the residual carries. Tight period spacings collapse this toward 0.
        #
        # The long_resid_over_intermediate denominator can legitimately collapse if the long-on-anchor projection has stripped all
        # variance (e.g. long ≈ a linear function of anchor on this window); report NaN rather than divide-by-zero.
        long_inter_var = float(long_resid_static.var())
        self.variance_retention = {
            "short_resid_over_source": float(short_resid.var() / common["short"].var()),
            "long_resid_over_source": float(long_resid_gs.var() / common["long"].var()),
            "long_resid_over_intermediate": (float(long_resid_gs.var() / long_inter_var)
                                              if long_inter_var > 0.0 else float("nan")),
            "short_resid_std_vs_anchor": float(short_resid.std() / anchor_vals.std()),
            "long_resid_std_vs_anchor": float(long_resid_gs.std() / anchor_vals.std()),
        }
        return self.variants

    # ========================================================================
    # 2. CORRELATION ANALYSIS
    # ========================================================================

    def analyze_correlations(self) -> Dict[StackVariant, pd.DataFrame]:
        """
        Compute the within-variant correlation matrix for each of the 4 variants
        plus max and mean off-diagonal absolute correlations.

        Populates ``self.correlation_results`` and returns it.
        """
        if not self.variants:
            self.build_variants()

        self._log("=" * 80)
        self._log(f"PERIOD-STACK CORRELATION AUDIT — {self.feature_name} at periods {self.periods}")
        self._log("=" * 80)

        self.correlation_results = {}
        for variant, df in self.variants.items():
            # Pearson-only: the orthogonalisation (variants 3 and 4) is linear/least-squares,
            # so reporting Pearson keeps the orthogonality claim consistent with the matrix.
            corr = df.corr(method="pearson")
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            off = corr.where(mask).abs().stack()
            max_off = float(off.max()) if len(off) else float("nan")
            mean_off = float(off.mean()) if len(off) else float("nan")
            self.correlation_results[variant] = corr

            self._log(f"\n--- {variant.value} ---")
            self._log(corr.round(4).to_string())
            self._log(f"(max off-diag |r| = {max_off:.4f}, mean off-diag |r| = {mean_off:.4f})")

        return self.correlation_results

    # ========================================================================
    # 3. SPACING + VARIANCE RETENTION ANALYSIS
    # ========================================================================

    def analyze_spacing(self) -> Dict[str, object]:
        """
        Translate periods into implied timescales, compute spacing ratios, classify the period stack as TOO_TIGHT /
        ADEQUATE / WIDE based on the Gram-Schmidt variance retention thresholds set at construction.

        Populates ``self.spacing_results`` and returns it.
        """
        if not self.variants:
            self.build_variants()

        self._log("\n" + "=" * 80)
        self._log("PERIOD SPACING & IMPLIED TIMESCALES")
        self._log("=" * 80)

        implied_seconds = {p: p * self.bar_seconds for p in self.periods}
        implied_human = {p: _seconds_to_human(s) for p, s in implied_seconds.items()}
        spacing_ratios = (self.short_period / self.anchor_period,
                          self.long_period / self.anchor_period)

        self._log(f"\nBar duration: {self.bar_seconds}s ({_seconds_to_human(self.bar_seconds)})")
        self._log("\nImplied timescales:")
        for p in self.periods:
            tag = " (anchor)" if p == self.anchor_period else ""
            self._log(f"  period={p:<5d} → {implied_seconds[p]:>7d}s = {implied_human[p]}{tag}")
        self._log(f"\nSpacing ratios (short/anchor, long/anchor): "
                  f"({spacing_ratios[0]:.3f}, {spacing_ratios[1]:.3f})")

        short_ret = self.variance_retention["short_resid_over_source"]
        long_ret = self.variance_retention["long_resid_over_source"]

        if (short_ret < self._verdict_thresholds["too_tight_short"]
                or long_ret < self._verdict_thresholds["too_tight_long"]):
            verdict = SpacingVerdict.TOO_TIGHT
        elif (short_ret > self._verdict_thresholds["wide_short"]
              and long_ret > self._verdict_thresholds["wide_long"]):
            verdict = SpacingVerdict.WIDE
        else:
            verdict = SpacingVerdict.ADEQUATE

        self._log("\nGram-Schmidt variance retention:")
        self._log(f"  short_resid / source short variance: {short_ret:.3f} ({100*short_ret:.1f}%)")
        self._log(f"  long_resid  / source long variance:  {long_ret:.3f} ({100*long_ret:.1f}%)")
        self._log(f"  std(short_resid) / std(anchor):      "
                  f"{self.variance_retention['short_resid_std_vs_anchor']:.3f}")
        self._log(f"  std(long_resid)  / std(anchor):      "
                  f"{self.variance_retention['long_resid_std_vs_anchor']:.3f}")
        self._log(f"\nVerdict: {verdict.value.upper()}")
        if verdict == SpacingVerdict.TOO_TIGHT:
            self._log("  → adjacent periods carry the same one-dimensional signal; widen the "
                      "spacing or drop the stack approach.")
        elif verdict == SpacingVerdict.ADEQUATE:
            self._log("  → residuals retain enough variance to be informative; the orthogonalised "
                      "triple is worth inspecting further.")
        else:
            self._log("  → very wide spacing; residuals carry substantial independent information.")

        self.spacing_results = {
            "implied_seconds": implied_seconds,
            "implied_human": implied_human,
            "spacing_ratios": spacing_ratios,
            "verdict": verdict,
            "short_retention": short_ret,
            "long_retention": long_ret,
        }
        return self.spacing_results

    # ========================================================================
    # 4. RECOMMENDATION
    # ========================================================================

    def recommend_construction(self) -> Dict[str, object]:
        """
        Recommend which candidate construction to inspect further, based on the correlation and spacing analysis.

        Cascade:
        1. NAIVE if its max off-diag |r| < 0.7 (no transform needed).
        2. SPREADS if its max |r| < 0.5.
        3. GRAM_SCHMIDT if SpacingVerdict is ADEQUATE or WIDE.
        4. Else recommend a 2-feature subset of SPREADS (drop the column most-correlated with anchor).

        Populates ``self.recommendation`` and returns it.
        """
        if not self.correlation_results:
            self.analyze_correlations()
        if not self.spacing_results:
            self.analyze_spacing()

        self._log("\n" + "=" * 80)
        self._log("RECOMMENDED CONSTRUCTION")
        self._log("=" * 80)

        def _max_off(variant: StackVariant) -> float:
            corr = self.correlation_results[variant]
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            return float(corr.where(mask).abs().stack().max())

        naive_max = _max_off(StackVariant.NAIVE)
        spreads_max = _max_off(StackVariant.SPREADS)
        verdict = self.spacing_results["verdict"]

        if naive_max < 0.7:
            recommended_variant = StackVariant.NAIVE
            recommended_features = list(self.variants[StackVariant.NAIVE].columns)
            rationale = (f"Naive stack max |r| = {naive_max:.3f} < 0.7 — periods carry distinct "
                         "signal as-is; no transform needed.")
        elif spreads_max < 0.5:
            recommended_variant = StackVariant.SPREADS
            recommended_features = list(self.variants[StackVariant.SPREADS].columns)
            rationale = (f"Spreads max |r| = {spreads_max:.3f} < 0.5 — level/alignment/curvature "
                         "construction is sufficient.")
        elif verdict in (SpacingVerdict.ADEQUATE, SpacingVerdict.WIDE):
            recommended_variant = StackVariant.GRAM_SCHMIDT
            recommended_features = list(self.variants[StackVariant.GRAM_SCHMIDT].columns)
            rationale = ("Naive and spread constructions remain coupled; SpacingVerdict is "
                         f"{verdict.value} so Gram-Schmidt orthogonalisation produces a "
                         "usable triple.")
        else:
            # Fall back: 2-feature subset of SPREADS dropping the column most-correlated with anchor.
            # Use the exact anchor column name (not prefix-match) so feature_name collisions
            # with the literal "alignment"/"curvature" column labels cannot pick the wrong column.
            spreads_corr = self.correlation_results[StackVariant.SPREADS]
            anchor_col = self.anchor_col_name
            other_cols = [c for c in spreads_corr.columns if c != anchor_col]
            corrs_with_anchor = {c: abs(spreads_corr.loc[anchor_col, c]) for c in other_cols}
            drop_col = max(corrs_with_anchor, key=corrs_with_anchor.get)
            keep_cols = [c for c in spreads_corr.columns if c != drop_col]
            recommended_variant = StackVariant.SPREADS
            recommended_features = keep_cols
            rationale = (f"SpacingVerdict is TOO_TIGHT — no full triple is salvageable. "
                         f"Best 2-feature subset of SPREADS is {keep_cols} (dropped "
                         f"'{drop_col}' as most-correlated with anchor at "
                         f"|r| = {corrs_with_anchor[drop_col]:.3f}).")

        self._log(f"\nVariant: {recommended_variant.value}")
        self._log(f"Features: {recommended_features}")
        self._log(f"Rationale: {rationale}")

        self.recommendation = {
            "variant": recommended_variant,
            "features": recommended_features,
            "rationale": rationale,
        }
        return self.recommendation

    # ========================================================================
    # 5. PLOTTING
    # ========================================================================

    def plot_correlation_matrices(self, figsize: Tuple[int, int] = (16, 4)):
        """
        Plot the 4 within-variant correlation heatmaps side-by-side.

        Returns the matplotlib Figure. Imports matplotlib and seaborn lazily so the core
        analysis path does not pull plotting dependencies on import.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.correlation_results:
            self.analyze_correlations()

        fig, axes = plt.subplots(1, 4, figsize=figsize)
        for ax, (variant, corr) in zip(axes, self.correlation_results.items()):
            sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                        square=True, cbar=False, ax=ax, linewidths=0.5)
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            max_off = float(corr.where(mask).abs().stack().max())
            ax.set_title(f"{variant.value}\nmax |r| = {max_off:.3f}", fontsize=10)
            ax.tick_params(axis="x", rotation=30)

        title = f"{self.feature_name} at periods {self.periods}"
        plt.suptitle(f"Period-stack correlation audit — {title}", fontsize=13, y=1.02)
        plt.tight_layout()
        return fig

    def plot_feature_distributions(self, variant: StackVariant = StackVariant.GRAM_SCHMIDT,
                                   figsize: Tuple[int, int] = (15, 4)):
        """
        Plot histograms for the columns of the chosen variant — useful for inspecting whether Gram-Schmidt residuals
        retain enough signal to model with.

        Returns the matplotlib Figure. Imports matplotlib lazily so the core analysis
        path does not pull plotting dependencies on import.
        """
        import matplotlib.pyplot as plt

        if not self.variants:
            self.build_variants()

        df = self.variants[variant]
        n_cols = df.shape[1]
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, df.columns):
            data = df[col].dropna()
            ax.hist(data, bins=80, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
            ax.set_title(f"{col}\nstd={data.std():.4g}", fontsize=10)
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            ax.grid(alpha=0.3)

        plt.suptitle(f"{variant.value} variant distributions — {self.feature_name} "
                     f"at {self.periods}", fontsize=13, y=1.02)
        plt.tight_layout()
        return fig

    # ========================================================================
    # 6. LEAKAGE-SAFE FIT / TRANSFORM (for shipping the orthogonalised triple into a model)
    # ========================================================================

    def fit_orthogonalisation(self, train_series: pd.Series) -> "PeriodStackAudit":
        """
        Fit β_short, β_long, γ Gram-Schmidt coefficients on a window of your choice.
        Pair with :meth:`transform_orthogonalisation` to apply the same coefficients to a different window without re-fitting.

        Use this as an escape hatch when you want to carry an exploration insight into a separate-window workflow
        (e.g. fit coefficients on one period of data and inspect the residuals on another) without reusing :meth:`build_variants`,
        which fits on the full input series.

        Parameters
        ----------
        train_series : pd.Series
            The underlying input restricted to the training window. Same dtype /
            interpretation as ``self.series``.

        Returns
        -------
        PeriodStackAudit
            ``self`` (for chaining).

        Raises
        ------
        ValueError
            If the joined feature frame on the train slice fails the same finite /
            non-constant / min-sample checks as ``build_variants``.
        """
        f_short = self.feature_fn(train_series, self.short_period)
        f_anchor = self.feature_fn(train_series, self.anchor_period)
        f_long = self.feature_fn(train_series, self.long_period)
        common = pd.DataFrame({"short": f_short, "anchor": f_anchor, "long": f_long}).dropna()
        self._validate_finite_frame(common, "fit_orthogonalisation", self._min_samples)
        anchor_vals = common["anchor"]

        self._beta_short = self._ols_beta(common["short"], anchor_vals)
        self._beta_long = self._ols_beta(common["long"], anchor_vals)
        short_resid = common["short"] - self._beta_short * anchor_vals
        long_resid_static = common["long"] - self._beta_long * anchor_vals
        if short_resid.var() == 0.0:
            raise ValueError("fit_orthogonalisation: short_resid has zero variance on the train "
                             "slice; periods are degenerate on this window.")
        self._gamma = self._ols_beta(long_resid_static, short_resid)
        return self

    def transform_orthogonalisation(self, series: pd.Series) -> pd.DataFrame:
        """
        Apply previously-fitted β/γ coefficients to produce the Gram-Schmidt orthogonal
        triple on a given series window, without re-fitting.

        Parameters
        ----------
        series : pd.Series
            The underlying input (e.g. close prices) on the window to be transformed.

        Returns
        -------
        pd.DataFrame
            A 3-column DataFrame with the anchor level, short residual, and Gram-Schmidt
            long residual. NaNs from feature-function warmup are preserved (not dropped)
            so the caller can choose alignment policy.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_orthogonalisation` (or :meth:`build_variants`) has
            populated the coefficients.
        """
        if self._beta_short is None or self._beta_long is None or self._gamma is None:
            raise RuntimeError("transform_orthogonalisation called before fit; "
                               "call fit_orthogonalisation(train_series) first.")
        f_short = self.feature_fn(series, self.short_period)
        f_anchor = self.feature_fn(series, self.anchor_period)
        f_long = self.feature_fn(series, self.long_period)
        short_resid = f_short - self._beta_short * f_anchor
        long_resid_static = f_long - self._beta_long * f_anchor
        long_resid_gs = long_resid_static - self._gamma * short_resid
        return pd.DataFrame({
            self.anchor_col_name: f_anchor,
            "short_resid": short_resid,
            "long_resid": long_resid_gs,
        })

    # ========================================================================
    # COMPREHENSIVE REPORT
    # ========================================================================

    def generate_comprehensive_report(self) -> Dict[str, object]:
        """
        Run the full audit: build variants, analyse correlations, classify spacing,
        recommend construction. Returns a dict carrying all intermediate artefacts.
        """
        self._log("\n" + "=" * 80)
        self._log(f"PERIOD-STACK AUDIT — {self.feature_name} at periods {self.periods}")
        self._log("=" * 80)

        self.build_variants()
        self.analyze_correlations()
        self.analyze_spacing()
        self.recommend_construction()

        self._log("\n" + "=" * 80)
        self._log("AUDIT COMPLETE")
        self._log("=" * 80)

        return {
            "variants": self.variants,
            "correlations": self.correlation_results,
            "spacing": self.spacing_results,
            "recommendation": self.recommendation,
            "variance_retention": self.variance_retention,
            "betas": {"short": self._beta_short, "long": self._beta_long, "gamma": self._gamma},
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_period_stack_audit(series: pd.Series, feature_fn: Callable[[pd.Series, int], pd.Series],
                              periods: Tuple[int, int, int], **kwargs) -> PeriodStackAudit:
    """
    Quick audit with default settings — equivalent to ``eda.quick_eda``.

    Parameters
    ----------
    series : pd.Series
        Underlying input (e.g. close prices).
    feature_fn : Callable[[pd.Series, int], pd.Series]
        Feature constructor.
    periods : Tuple[int, int, int]
        Three periods to audit.
    **kwargs
        Forwarded to ``PeriodStackAudit``.

    Returns
    -------
    PeriodStackAudit
        Auditor instance with all results populated.
    """
    auditor = PeriodStackAudit(series, feature_fn, periods, **kwargs)
    auditor.generate_comprehensive_report()
    return auditor


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _seconds_to_human(seconds: int) -> str:
    """Format seconds as a readable timescale string (e.g. 21600 -> '6h', 60000 -> '16.7h')."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m = seconds / 60
        return f"{m:.1f}m" if m % 1 else f"{int(m)}m"
    if seconds < 86400:
        h = seconds / 3600
        return f"{h:.1f}h" if h % 1 else f"{int(h)}h"
    d = seconds / 86400
    return f"{d:.1f}d" if d % 1 else f"{int(d)}d"
