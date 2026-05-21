"""Tests for okmich_quant_research.period_stack_audit.

Covers:
 - Math correctness on synthetic series with controllable orthogonality structure
 - Validation behaviour (bad periods, finite-data + non-constant input contract)
 - Spacing-verdict logic across TOO_TIGHT / ADEQUATE / WIDE regimes
 - Recommendation cascade across each branch
 - Leakage-safe fit_orthogonalisation / transform_orthogonalisation path
 - Real-data smoke test locking in the empirical EURUSD 5m finding:
   (36,60,120) → TOO_TIGHT, (24,72,200) → ADEQUATE.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless plotting for CI

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.period_stack_audit import (
    PeriodStackAudit,
    SpacingVerdict,
    StackVariant,
    quick_period_stack_audit,
)


# ============================================================================
# SYNTHETIC HELPERS
# ============================================================================


def _norm_ema_like(series: pd.Series, period: int) -> pd.Series:
    """Synthetic ``norm_ema``-shaped feature: log(price / EMA(period))."""
    ema = series.ewm(span=period, adjust=False).mean()
    return np.log(series / ema)


def _wide_signal_series(n: int = 5000, seed: int = 7) -> pd.Series:
    """Generate a series whose short / long EMA deviations carry distinct information.

    Random walk + sinusoidal trend at a long timescale → wide ema-period spacing should
    leave substantial independent variance in the residuals.
    """
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.001, n))
    slow = 0.05 * np.sin(np.linspace(0.0, 8 * np.pi, n))
    price = 100.0 * np.exp(walk + slow)
    return pd.Series(price)


# ============================================================================
# CONSTRUCTION + VALIDATION
# ============================================================================


def test_audit_init_validates_periods_length() -> None:
    series = _wide_signal_series(500)
    with pytest.raises(ValueError, match="exactly 3 values"):
        PeriodStackAudit(series, _norm_ema_like, (10, 20), verbose=False)


def test_audit_init_validates_positive_periods() -> None:
    series = _wide_signal_series(500)
    with pytest.raises(ValueError, match="positive"):
        PeriodStackAudit(series, _norm_ema_like, (10, 20, -1), verbose=False)


def test_audit_init_validates_distinct_periods() -> None:
    series = _wide_signal_series(500)
    with pytest.raises(ValueError, match="distinct"):
        PeriodStackAudit(series, _norm_ema_like, (10, 10, 20), verbose=False)


def test_audit_init_validates_bar_seconds() -> None:
    series = _wide_signal_series(500)
    with pytest.raises(ValueError, match="bar_seconds"):
        PeriodStackAudit(series, _norm_ema_like, (10, 20, 40), bar_seconds=0, verbose=False)


def test_audit_anchor_is_always_the_median_period() -> None:
    """Anchor must be the median period — non-median anchors would make curvature semantics incoherent."""
    series = _wide_signal_series(500)
    audit = PeriodStackAudit(series, _norm_ema_like, (40, 10, 20), verbose=False)
    assert audit.anchor_period == 20
    assert audit.short_period == 10
    assert audit.long_period == 40
    assert audit.periods == (10, 20, 40)  # sorted


# ============================================================================
# BUILD_VARIANTS INPUT-CONTRACT VALIDATION
# ============================================================================


def test_build_variants_rejects_constant_series() -> None:
    """A constant series produces zero-variance features → must fail closed, not produce NaN."""
    series = pd.Series([1.0] * 5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240), verbose=False)
    with pytest.raises(ValueError, match="zero variance|constant"):
        audit.build_variants()


def test_build_variants_rejects_too_few_samples() -> None:
    """Below the min-sample threshold the OLS / variance estimates are unreliable."""
    # 20 bars is below the default min_samples=30; _norm_ema_like preserves length, so the
    # joined frame has 20 rows after dropna.
    series = _wide_signal_series(20)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240), verbose=False)
    with pytest.raises(ValueError, match="rows after dropna|need >="):
        audit.build_variants()


def test_build_variants_rejects_inf_in_feature_output() -> None:
    """If the feature function emits inf (e.g. log(0)), build_variants must reject the frame."""
    def broken_feature_fn(s: pd.Series, period: int) -> pd.Series:
        out = s.ewm(span=period, adjust=False).mean()
        out.iloc[100] = np.inf  # inject a single non-finite value
        return out

    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, broken_feature_fn, (20, 60, 240), verbose=False)
    with pytest.raises(ValueError, match="non-finite|inf"):
        audit.build_variants()


# ============================================================================
# MATH: ORTHOGONALITY
# ============================================================================


def test_gram_schmidt_produces_orthogonal_triple() -> None:
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.build_variants()

    gs = audit.variants[StackVariant.GRAM_SCHMIDT]
    corr = gs.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    max_off = corr.where(mask).abs().stack().max()
    assert max_off < 1e-6, f"Gram-Schmidt should produce orthogonal triple, max |r| = {max_off}"


def test_ols_residuals_are_orthogonal_to_anchor() -> None:
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.build_variants()

    ortho = audit.variants[StackVariant.OLS_ORTHO]
    anchor_col = "ema_60"
    assert abs(ortho.corr().loc[anchor_col, "short_resid"]) < 1e-6
    assert abs(ortho.corr().loc[anchor_col, "long_resid"]) < 1e-6


def test_spreads_variant_includes_three_distinct_signals() -> None:
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.build_variants()

    spreads = audit.variants[StackVariant.SPREADS]
    assert set(spreads.columns) == {"ema_60", "alignment", "curvature"}
    assert spreads.shape[1] == 3


# ============================================================================
# VERDICT LOGIC
# ============================================================================


def test_tight_spacing_yields_too_tight_verdict() -> None:
    """Very-close periods produce highly correlated stacks; verdict should flag this."""
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (58, 60, 62),
                              feature_name="ema", verbose=False)
    result = audit.generate_comprehensive_report()
    assert result["spacing"]["verdict"] == SpacingVerdict.TOO_TIGHT


def test_wide_spacing_yields_adequate_or_wide_verdict() -> None:
    """Well-separated periods retain variance after orthogonalisation."""
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (10, 60, 360),
                              feature_name="ema", verbose=False)
    result = audit.generate_comprehensive_report()
    assert result["spacing"]["verdict"] in (SpacingVerdict.ADEQUATE, SpacingVerdict.WIDE)


def test_implied_timescales_translation_is_correct() -> None:
    series = _wide_signal_series(500)
    audit = PeriodStackAudit(series, _norm_ema_like, (24, 72, 200),
                              bar_seconds=300, feature_name="ema", verbose=False)
    audit.build_variants()
    audit.analyze_spacing()

    assert audit.spacing_results["implied_seconds"][24] == 24 * 300
    assert audit.spacing_results["implied_seconds"][72] == 72 * 300
    assert audit.spacing_results["implied_seconds"][200] == 200 * 300


# ============================================================================
# RECOMMENDATION CASCADE
# ============================================================================


def test_recommendation_returns_valid_variant_and_features() -> None:
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.generate_comprehensive_report()

    rec = audit.recommendation
    assert rec["variant"] in StackVariant
    assert isinstance(rec["features"], list)
    assert len(rec["features"]) >= 2
    assert isinstance(rec["rationale"], str) and rec["rationale"]


def test_recommendation_falls_back_to_subset_when_too_tight() -> None:
    """For TOO_TIGHT spacing the cascade can land on SPREADS (full or 2-feature subset),
    but never on NAIVE (would imply periods carry distinct signal) or GRAM_SCHMIDT
    (cascade branch 3 requires SpacingVerdict.ADEQUATE or WIDE).
    """
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (58, 60, 62),
                              feature_name="ema", verbose=False)
    result = audit.generate_comprehensive_report()
    assert result["spacing"]["verdict"] == SpacingVerdict.TOO_TIGHT
    rec = result["recommendation"]
    assert rec["variant"] != StackVariant.NAIVE, (
        f"TOO_TIGHT must not be recommended as NAIVE; got {rec['variant']}"
    )
    assert rec["variant"] != StackVariant.GRAM_SCHMIDT, (
        f"TOO_TIGHT must not land on GRAM_SCHMIDT (cascade branch 3 requires ADEQUATE/WIDE); "
        f"got {rec['variant']}"
    )
    if "TOO_TIGHT" in rec["rationale"]:
        assert rec["variant"] == StackVariant.SPREADS
        assert len(rec["features"]) == 2


# ============================================================================
# LEAKAGE-SAFE FIT / TRANSFORM
# ============================================================================


def test_transform_orthogonalisation_requires_prior_fit() -> None:
    """Calling transform before fit (or build_variants) is a hard error."""
    series = _wide_signal_series(5000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    with pytest.raises(RuntimeError, match="fit"):
        audit.transform_orthogonalisation(series)


def test_fit_then_transform_produces_orthogonal_triple_on_train_slice() -> None:
    """Fitting on train and transforming on the same train slice should produce orthogonality."""
    series = _wide_signal_series(5000)
    train = series.iloc[: int(0.7 * len(series))]
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.fit_orthogonalisation(train)
    out = audit.transform_orthogonalisation(train).dropna()
    corr = out.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    assert corr.where(mask).abs().stack().max() < 1e-6


def test_fit_on_train_transform_on_oos_does_not_refit_coefficients() -> None:
    """The β / γ coefficients must be the ones fitted on train, not re-derived from OOS."""
    series = _wide_signal_series(5000)
    train = series.iloc[: int(0.7 * len(series))]
    oos = series.iloc[int(0.7 * len(series)):]

    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.fit_orthogonalisation(train)
    b_short, b_long, gamma = audit._beta_short, audit._beta_long, audit._gamma

    _ = audit.transform_orthogonalisation(oos)
    assert audit._beta_short == b_short
    assert audit._beta_long == b_long
    assert audit._gamma == gamma


def test_fitted_coefficients_returns_none_before_fit() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240), verbose=False)
    assert audit.fitted_coefficients is None


def test_fitted_coefficients_returns_dict_after_build_variants() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240), verbose=False)
    audit.build_variants()
    coefs = audit.fitted_coefficients
    assert coefs is not None
    assert set(coefs.keys()) == {"beta_short", "beta_long", "gamma"}
    assert all(isinstance(v, float) for v in coefs.values())


def test_fitted_coefficients_match_fit_orthogonalisation() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240), verbose=False)
    audit.fit_orthogonalisation(series.iloc[:1500])
    coefs = audit.fitted_coefficients
    assert coefs["beta_short"] == audit._beta_short
    assert coefs["beta_long"] == audit._beta_long
    assert coefs["gamma"] == audit._gamma


def test_fit_orthogonalisation_returns_self_for_chaining() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    returned = audit.fit_orthogonalisation(series)
    assert returned is audit


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def test_quick_period_stack_audit_runs_full_report() -> None:
    series = _wide_signal_series(5000)
    audit = quick_period_stack_audit(series, _norm_ema_like, (20, 60, 240),
                                      feature_name="ema", verbose=False)
    assert audit.variants
    assert audit.correlation_results
    assert audit.spacing_results
    assert audit.recommendation


# ============================================================================
# VERBOSE GATE
# ============================================================================


def test_verbose_false_suppresses_prints(capsys) -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.generate_comprehensive_report()
    captured = capsys.readouterr()
    assert captured.out == "", f"verbose=False should produce no stdout, got: {captured.out!r}"


def test_verbose_true_emits_output(capsys) -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=True)
    audit.generate_comprehensive_report()
    captured = capsys.readouterr()
    assert "PERIOD-STACK AUDIT" in captured.out
    assert "Verdict" in captured.out


# ============================================================================
# PLOTTING (smoke — no assertion beyond return type)
# ============================================================================


def test_plot_correlation_matrices_returns_figure() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.generate_comprehensive_report()
    fig = audit.plot_correlation_matrices()
    assert fig is not None


def test_plot_feature_distributions_returns_figure() -> None:
    series = _wide_signal_series(2000)
    audit = PeriodStackAudit(series, _norm_ema_like, (20, 60, 240),
                              feature_name="ema", verbose=False)
    audit.generate_comprehensive_report()
    fig = audit.plot_feature_distributions()
    assert fig is not None


# ============================================================================
# REAL-DATA SMOKE TEST
# ============================================================================

_EURUSD_PARQUET = Path(r"D:/data_dump/market_data/raw/FXPIG-Server/5/EURUSD.r.parquet")


@pytest.mark.skipif(
    not _EURUSD_PARQUET.exists(),
    reason=f"EURUSD 5m parquet not present at {_EURUSD_PARQUET}; skipping data-dependent smoke test"
)
def test_smoke_eurusd_5m_tight_periods_yield_too_tight() -> None:
    """Locks in the empirical finding from the May 2026 audit session:
    (36, 60, 120) on EURUSD 5m has GS variance retention ~6%/3% and is TOO_TIGHT.
    """
    from okmich_quant_features.trend.normalized_ma import norm_ema

    raw = pd.read_parquet(_EURUSD_PARQUET).iloc[-60_000:].copy()
    raw.columns = [c.lower() for c in raw.columns]

    audit = PeriodStackAudit(raw["close"], norm_ema, (36, 60, 120),
                              bar_seconds=300, feature_name="norm_ema", verbose=False)
    result = audit.generate_comprehensive_report()

    assert result["spacing"]["verdict"] == SpacingVerdict.TOO_TIGHT
    assert result["variance_retention"]["short_resid_over_source"] < 0.10
    assert result["variance_retention"]["long_resid_over_source"] < 0.05


@pytest.mark.skipif(
    not _EURUSD_PARQUET.exists(),
    reason=f"EURUSD 5m parquet not present at {_EURUSD_PARQUET}; skipping data-dependent smoke test"
)
def test_smoke_eurusd_5m_wider_periods_yield_adequate_or_wide() -> None:
    """Locks in the empirical finding: (24, 72, 200) on EURUSD 5m has GS variance
    retention ~25%/14% and is ADEQUATE.
    """
    from okmich_quant_features.trend.normalized_ma import norm_ema

    raw = pd.read_parquet(_EURUSD_PARQUET).iloc[-60_000:].copy()
    raw.columns = [c.lower() for c in raw.columns]

    audit = PeriodStackAudit(raw["close"], norm_ema, (24, 72, 200),
                              bar_seconds=300, feature_name="norm_ema", verbose=False)
    result = audit.generate_comprehensive_report()

    assert result["spacing"]["verdict"] in (SpacingVerdict.ADEQUATE, SpacingVerdict.WIDE)
    assert result["variance_retention"]["short_resid_over_source"] > 0.10
    assert result["variance_retention"]["long_resid_over_source"] > 0.05