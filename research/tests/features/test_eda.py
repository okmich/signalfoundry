"""Tests for okmich_quant_research.features.eda.

Covers:
 - The leakage contract: no default analysis path reads a single holdout row
 - Partition geometry: train-edge purge width, boundary immune to holdout content
 - Walk-forward folds: strictly forward, purged and embargoed, inside train only
 - Frozen transform parameters (fit on train, apply anywhere) and their causality
 - HAC correction actually deflating an overlapping-label t-statistic
 - Backward compatibility of the pre-audit call signatures
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")  # headless plotting for CI

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.eda import (
    EDAMode,
    EDAScope,
    FeatureEDA,
    LeakageManifest,
    Transformation,
    TargetType,
    WFScheme,
    quick_eda,
)


# ============================================================================
# SYNTHETIC HELPERS
# ============================================================================


def _panel(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame with one real predictor, one pure-noise column, one skewed column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    real = pd.Series(rng.normal(size=n), index=idx)
    features = pd.DataFrame({
        "real": real,
        "fake": rng.normal(size=n),
        "skewed": rng.lognormal(0.0, 1.2, n),
        "bounded": rng.uniform(0.0, 1.0, n),
    }, index=idx)
    target = pd.Series(0.25 * real.to_numpy() + rng.normal(size=n), index=idx, name="fwd")
    return features, target


def _poison(features: pd.DataFrame, target: pd.Series, from_pos: int) -> tuple[pd.DataFrame, pd.Series]:
    """Replace everything from ``from_pos`` onward with values that would wreck any statistic."""
    bad_features = features.copy()
    bad_target = target.copy()
    bad_features.iloc[from_pos:] = np.inf
    bad_features.iloc[from_pos::3] = np.nan
    bad_features.iloc[from_pos + 1::7] = -1e18
    bad_target.iloc[from_pos:] = 1e18
    bad_target.iloc[from_pos::5] = np.nan
    return bad_features, bad_target


def _eda(features: pd.DataFrame, target: pd.Series, **kwargs) -> FeatureEDA:
    defaults = dict(horizon=12, verbose=False)
    defaults.update(kwargs)
    return FeatureEDA(features, target, **defaults)


# ============================================================================
# THE LEAKAGE CONTRACT
# ============================================================================


def test_no_default_analysis_reads_a_holdout_row() -> None:
    """Poisoning every holdout row must not move a single number in any default output.

    This is the load-bearing test: it is what turns "reads train only" from a comment in
    the docstring into a property of the code.
    """
    features, target = _panel()
    clean = _eda(features, target)
    poisoned_features, poisoned_target = _poison(features, target, from_pos=len(features) * 3 // 4)
    dirty = _eda(poisoned_features, poisoned_target)

    pd.testing.assert_frame_equal(clean.analyze_feature_relevance(), dirty.analyze_feature_relevance())
    pd.testing.assert_frame_equal(clean.analyze_distributions(), dirty.analyze_distributions())
    pd.testing.assert_frame_equal(clean.compute_vif(), dirty.compute_vif())
    pd.testing.assert_frame_equal(clean.recommend_transformations(), dirty.recommend_transformations())
    pd.testing.assert_frame_equal(clean.analyze_model_based_importance(), dirty.analyze_model_based_importance())

    clean_corr, clean_pairs = clean.analyze_correlation()
    dirty_corr, dirty_pairs = dirty.analyze_correlation()
    pd.testing.assert_frame_equal(clean_corr, dirty_corr)
    assert clean_pairs == dirty_pairs


def test_walk_forward_mode_also_reads_no_holdout_row() -> None:
    features, target = _panel()
    clean = _eda(features, target, mode=EDAMode.WALK_FORWARD)
    poisoned_features, poisoned_target = _poison(features, target, from_pos=len(features) * 3 // 4)
    dirty = _eda(poisoned_features, poisoned_target, mode=EDAMode.WALK_FORWARD)
    pd.testing.assert_frame_equal(clean.analyze_feature_relevance(), dirty.analyze_feature_relevance())


def test_holdout_nan_targets_do_not_move_the_split_boundary() -> None:
    """Regression: the pre-audit version dropped NaN targets *before* splitting, so the
    count of missing targets inside the holdout shifted the positional boundary."""
    features, target = _panel()
    baseline = _eda(features, target)

    holed = target.copy()
    holed.iloc[len(target) * 3 // 4::2] = np.nan  # blow away half the holdout targets
    perturbed = _eda(features, holed)

    pd.testing.assert_index_equal(baseline.train_index, perturbed.train_index)
    assert baseline.manifest.n_train == perturbed.manifest.n_train


def test_train_edge_is_purged_by_exactly_horizon_bars() -> None:
    features, target = _panel(n=1000)
    horizon = 20
    eda = _eda(features, target, horizon=horizon)
    first_holdout = eda.holdout_index[0]
    gap = features.index.get_loc(first_holdout) - features.index.get_loc(eda.train_index[-1])
    assert gap == horizon + 1
    assert eda.manifest.n_purged == horizon


def test_train_and_holdout_indices_are_disjoint() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    assert len(eda.train_index.intersection(eda.holdout_index)) == 0
    assert eda.train_index.max() < eda.holdout_index.min()


def test_explicit_holdout_scope_warns() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    with pytest.warns(UserWarning, match="contaminated"):
        eda.analyze_distributions(scope=EDAScope.HOLDOUT)


def test_explicit_holdout_scope_actually_changes_the_answer() -> None:
    """The opt-in escape hatch must be real, or the poison test proves nothing."""
    features, target = _panel()
    poisoned_features, poisoned_target = _poison(features, target, from_pos=len(features) * 3 // 4)
    eda = _eda(poisoned_features, poisoned_target)
    train_only = eda.analyze_distributions()
    with pytest.warns(UserWarning):
        full = eda.analyze_distributions(scope=EDAScope.FULL)
    assert not train_only.equals(full)


# ============================================================================
# PARTITION VALIDATION
# ============================================================================


def test_missing_horizon_warns_about_the_purge_width() -> None:
    features, target = _panel()
    with pytest.warns(UserWarning, match="horizon"):
        FeatureEDA(features, target, verbose=False)


def test_explicit_horizon_does_not_warn() -> None:
    features, target = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FeatureEDA(features, target, horizon=1, verbose=False)


def test_train_threshold_accepts_a_timestamp_boundary() -> None:
    features, target = _panel()
    boundary = features.index[1200]
    eda = _eda(features, target, train_threshold=boundary)
    assert eda.holdout_index[0] == boundary
    assert eda.train_index[-1] < boundary


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.2])
def test_degenerate_train_threshold_is_rejected(threshold: float) -> None:
    features, target = _panel(n=200)
    with pytest.raises(ValueError):
        _eda(features, target, train_threshold=threshold)


def test_horizon_larger_than_train_partition_is_rejected() -> None:
    features, target = _panel(n=200)
    with pytest.raises(ValueError, match="leaves no training rows"):
        _eda(features, target, horizon=1000)


def test_disjoint_indices_are_rejected() -> None:
    features, target = _panel(n=200)
    shifted = target.copy()
    shifted.index = shifted.index + pd.Timedelta(days=3650)
    with pytest.raises(ValueError, match="share no index"):
        _eda(features, shifted)


# ============================================================================
# WALK-FORWARD FOLDS
# ============================================================================


def test_walk_forward_folds_are_strictly_forward_and_gapped() -> None:
    features, target = _panel(n=4000)
    horizon = 12
    eda = _eda(features, target, horizon=horizon, mode=EDAMode.WALK_FORWARD, n_splits=5)
    assert len(eda.wf_folds) >= 2
    for train_pos, test_pos in eda.wf_folds:
        assert train_pos.max() < test_pos.min(), "a fold trained on rows at or after its test block"
        assert test_pos.min() - train_pos.max() >= horizon + eda.embargo_bars


def test_walk_forward_folds_never_leave_the_train_partition() -> None:
    features, target = _panel(n=4000)
    eda = _eda(features, target, mode=EDAMode.WALK_FORWARD)
    n_train = len(eda.train_index)
    for train_pos, test_pos in eda.wf_folds:
        assert test_pos.max() < n_train


def test_anchored_folds_grow_and_rolling_folds_do_not() -> None:
    features, target = _panel(n=4000)
    anchored = _eda(features, target, mode=EDAMode.WALK_FORWARD, wf_scheme=WFScheme.ANCHORED)
    rolling = _eda(features, target, mode=EDAMode.WALK_FORWARD, wf_scheme=WFScheme.ROLLING)

    anchored_sizes = [len(train_pos) for train_pos, _ in anchored.wf_folds]
    rolling_sizes = [len(train_pos) for train_pos, _ in rolling.wf_folds]
    assert anchored_sizes == sorted(anchored_sizes) and anchored_sizes[-1] > anchored_sizes[0]
    assert len(set(rolling_sizes)) == 1
    assert all(train_pos[0] == 0 for train_pos, _ in anchored.wf_folds)


def test_walk_forward_reports_is_and_oos_side_by_side() -> None:
    features, target = _panel(n=4000)
    eda = _eda(features, target, mode=EDAMode.WALK_FORWARD)
    result = eda.analyze_feature_relevance()
    for column in ("pearson_is_mean", "pearson_oos_mean", "pearson_oos_std", "decay",
                   "oos_sign_consistency", "n_folds"):
        assert column in result.columns
    real = result.set_index("feature").loc["real"]
    assert real["oos_sign_consistency"] == 1.0
    assert abs(real["pearson_oos_mean"]) > 0.1


def test_walk_forward_needs_enough_data_for_two_folds() -> None:
    # 400 rows -> 300 train rows, 200 purged at the edge -> 100 usable, which cannot carry a
    # single fold once the horizon+embargo gap is applied.
    features, target = _panel(n=400)
    with pytest.raises(ValueError, match="at least 2 constructible folds"):
        _eda(features, target, horizon=200, mode=EDAMode.WALK_FORWARD, n_splits=8)


# ============================================================================
# FROZEN TRANSFORM PARAMETERS
# ============================================================================


def test_fitted_spec_params_come_from_train_only() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    spec = eda.fit_transformations()
    assert len(spec.fitted_on) == len(eda.train_index)
    assert spec.fitted_on.max() < eda.holdout_index.min()


def test_standardize_uses_train_moments_not_full_sample_moments() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    transformed = eda.apply_transformation("real", Transformation.STANDARDIZE)

    train_slice = features.loc[eda.train_index, "real"]
    expected_mean, expected_std = train_slice.mean(), train_slice.std(ddof=1)
    reconstructed = transformed * expected_std + expected_mean
    pd.testing.assert_series_equal(reconstructed.rename("real"), features["real"], atol=1e-9)

    # And the train-fitted z-scores must NOT match a naive full-sample standardisation.
    naive = (features["real"] - features["real"].mean()) / features["real"].std(ddof=1)
    assert not np.allclose(transformed.to_numpy(), naive.to_numpy())


@pytest.mark.parametrize("transformation", [Transformation.RANK, Transformation.STANDARDIZE,
                                            Transformation.QUANTILE])
def test_transformed_value_does_not_depend_on_later_bars(transformation: Transformation) -> None:
    """A bar's transformed value must be identical whether or not the future exists yet."""
    features, target = _panel()
    eda = _eda(features, target)
    spec_transformer = eda.apply_transformation("real", transformation)

    prefix = features["real"].iloc[:500]
    prefix_transformed = eda.apply_transformation("real", transformation).iloc[:500]
    np.testing.assert_allclose(prefix_transformed.to_numpy(), spec_transformer.iloc[:500].to_numpy())
    assert len(prefix) == len(prefix_transformed)


def test_naive_rank_is_the_look_ahead_this_replaces() -> None:
    """Sanity check on the test above: pandas' own rank(pct=True) *does* depend on the future."""
    features, _ = _panel()
    series = features["real"]
    full_rank = series.rank(pct=True)
    prefix_rank = series.iloc[:500].rank(pct=True)
    assert not np.allclose(full_rank.iloc[:500].to_numpy(), prefix_rank.to_numpy())


def test_rank_transform_scores_against_the_frozen_train_cdf() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    ranks = eda.apply_transformation("real", Transformation.RANK)
    assert ranks.name == "real_rank"
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0
    # A value above every train observation saturates at 1.0 rather than re-ranking the sample.
    train_max = features.loc[eda.train_index, "real"].max()
    beyond = pd.Series([train_max + 10.0], index=[features.index[-1]])
    fitted = eda.fit_transformations(recommendations=pd.DataFrame(
        [{"feature": "real", "transformations": str(Transformation.RANK)}]))
    assert fitted.transform_feature(beyond, "real").iloc[0] == 1.0


def test_spec_transform_applies_to_the_holdout_without_refitting() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    spec = eda.fit_transformations()
    holdout = features.loc[eda.holdout_index]

    once = spec.transform(holdout)
    # Transforming a subset must give the same numbers as transforming the whole block.
    subset = spec.transform(holdout.iloc[:100])
    pd.testing.assert_frame_equal(subset, once.iloc[:100])


def test_spec_transform_can_add_suffixed_columns() -> None:
    features, target = _panel()
    spec = _eda(features, target).fit_transformations()
    out = spec.transform(features.loc[_eda(features, target).holdout_index], replace_original=False)
    assert any(col.endswith("_standardize") or col.endswith("_log") for col in out.columns)


def test_unfitted_transformer_refuses_to_transform() -> None:
    from okmich_quant_research.features.eda import _FittedRank
    with pytest.raises(RuntimeError, match="must be fitted"):
        _FittedRank().transform(pd.Series([1.0, 2.0]))


def test_apply_transformation_rejects_an_unknown_name() -> None:
    features, target = _panel()
    with pytest.raises(ValueError):
        _eda(features, target).apply_transformation("real", "not-a-transformation")


def test_apply_transformation_preserves_the_legacy_series_names() -> None:
    features, target = _panel()
    eda = _eda(features, target)
    assert eda.apply_transformation("bounded", Transformation.LOGIT).name == "bounded_logit"
    assert eda.apply_transformation("skewed", Transformation.LOG).name == "skewed_log"
    assert eda.apply_transformation("real", Transformation.YEO_JOHNSON).name == "real_yeojohnson"
    assert eda.apply_transformation("real", Transformation.STANDARDIZE).name == "real_std"


# ============================================================================
# HAC / EFFECTIVE SAMPLE SIZE
# ============================================================================


def test_hac_deflates_significance_under_overlapping_labels() -> None:
    """A persistent feature against an overlapping forward return: the iid p-value screams
    discovery, the HAC p-value does not."""
    rng = np.random.default_rng(3)
    n, horizon = 3000, 24
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    persistent = pd.Series(rng.normal(size=n), index=idx).rolling(50).mean().bfill()
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))), index=idx)
    forward = (price.shift(-horizon) / price - 1.0).rename("fwd")
    features = pd.DataFrame({"persistent": persistent, "noise": rng.normal(size=n)}, index=idx)

    eda = _eda(features, forward, horizon=horizon)
    row = eda.analyze_feature_relevance().set_index("feature").loc["persistent"]

    assert row["pearson_pval"] < 0.01
    assert row["hac_pval"] > row["pearson_pval"] * 100
    assert row["n_eff"] < len(eda.train_index) / 5


def test_effective_n_stays_near_n_for_an_iid_feature() -> None:
    features, target = _panel(n=3000)
    eda = _eda(features, target, horizon=1)
    row = eda.analyze_feature_relevance().set_index("feature").loc["fake"]
    assert row["n_eff"] > 0.5 * len(eda.train_index)


def test_multiclass_target_gets_no_hac_statistic_rather_than_a_wrong_one() -> None:
    features, _ = _panel(n=1500)
    rng = np.random.default_rng(5)
    labels = pd.Series(rng.integers(0, 3, len(features)), index=features.index)
    eda = _eda(features, labels, target_type=TargetType.CATEGORICAL)
    result = eda.analyze_feature_relevance()
    assert result["hac_tstat"].isna().all()
    assert "mutual_info" in result.columns


# ============================================================================
# MODEL-BASED IMPORTANCE
# ============================================================================


def test_model_importance_is_averaged_over_purged_folds() -> None:
    features, target = _panel(n=3000)
    eda = _eda(features, target, n_splits=4)
    result = eda.analyze_model_based_importance()
    assert list(result.columns) == ["feature", "importance", "importance_std", "n_folds"]
    assert (result["n_folds"] == 4).all()
    assert result.iloc[0]["feature"] == "real"


def test_linear_importance_is_scale_invariant() -> None:
    """Z-scoring inside each fold means a unit change cannot reorder the ranking."""
    features, target = _panel(n=3000)
    baseline = _eda(features, target).analyze_model_based_importance(model_type="linear")

    rescaled = features.copy()
    rescaled["fake"] = rescaled["fake"] * 1e6
    scaled = _eda(rescaled, target).analyze_model_based_importance(model_type="linear")

    assert baseline["feature"].tolist() == scaled["feature"].tolist()


# ============================================================================
# RECOMMENDATIONS
# ============================================================================


def test_tukey_outlier_rule_fires_on_a_heavy_tailed_feature() -> None:
    """Regression: the old rule counted points outside the 1st/99th percentiles, which is
    ~2% of any sample by construction, so it could never clear its own 5% trigger."""
    rng = np.random.default_rng(9)
    n = 2000
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    heavy = pd.Series(rng.standard_t(df=1.2, size=n), index=idx)
    features = pd.DataFrame({"heavy": heavy}, index=idx)
    target = pd.Series(rng.normal(size=n), index=idx)

    recommendations = _eda(features, target).recommend_transformations()
    assert "quantile/rank" in recommendations.set_index("feature").loc["heavy", "transformations"]


def test_bounded_feature_is_recommended_logit() -> None:
    features, target = _panel()
    recommendations = _eda(features, target).recommend_transformations()
    assert "logit" in recommendations.set_index("feature").loc["bounded", "transformations"]


# ============================================================================
# REPORT / CONVENIENCE / BACKWARD COMPATIBILITY
# ============================================================================


def test_report_carries_the_leakage_manifest() -> None:
    features, target = _panel()
    report = _eda(features, target).generate_comprehensive_report()
    assert isinstance(report["manifest"], LeakageManifest)
    assert report["manifest"].n_holdout > 0
    assert set(report) >= {"relevance", "distribution", "correlation_matrix", "vif", "transformations"}


def test_report_writes_html_when_given_a_path(tmp_path) -> None:
    features, target = _panel()
    destination = tmp_path / "report.html"
    _eda(features, target).generate_comprehensive_report(output_path=str(destination))
    assert destination.exists()
    assert "LEAKAGE MANIFEST" in destination.read_text(encoding="utf-8")


def test_legacy_positional_signature_still_constructs() -> None:
    """FeatureEDA(features, target, 'continuous', ['real']) is the pre-audit call shape."""
    features, target = _panel()
    with pytest.warns(UserWarning, match="horizon"):
        eda = FeatureEDA(features, target, "continuous", ["real", "fake"])
    eda.verbose = False
    assert eda.feature_names == ["real", "fake"]
    assert eda.target_type == TargetType.CONTINUOUS
    assert not eda.analyze_feature_relevance().empty


def test_numpy_target_is_accepted() -> None:
    features, target = _panel()
    eda = _eda(features, target.to_numpy())
    assert len(eda.train_index) > 0


def test_quick_eda_runs_end_to_end() -> None:
    features, target = _panel(n=1500)
    eda = quick_eda(features, target, horizon=12, verbose=False)
    assert isinstance(eda, FeatureEDA)
    assert not eda.relevance_results.empty


def test_verbose_false_suppresses_output(capsys) -> None:
    features, target = _panel(n=1000)
    _eda(features, target).generate_comprehensive_report()
    assert capsys.readouterr().out == ""


def test_verbose_true_emits_the_manifest(capsys) -> None:
    features, target = _panel(n=1000)
    _eda(features, target, verbose=True).generate_comprehensive_report()
    assert "LEAKAGE MANIFEST" in capsys.readouterr().out


# ============================================================================
# PLOTS
# ============================================================================


def test_plots_return_figures_and_stamp_the_scope() -> None:
    features, target = _panel(n=1000)
    eda = _eda(features, target)
    for figure in (eda.plot_distributions(n_features=4), eda.plot_qq_plots(n_features=4),
                   eda.plot_correlation_matrix(cluster=False)):
        assert figure is not None
    assert "train" in eda.plot_distributions(n_features=2)._suptitle.get_text()


# ============================================================================
# INPUT VALIDATION (regressions from the review pass)
# ============================================================================


def test_descending_index_is_rejected() -> None:
    """A descending index made the *oldest* bars the holdout and trained on the future,
    silently, because every partition operation is positional."""
    features, target = _panel(n=800)
    with pytest.raises(ValueError, match="sorted ascending"):
        _eda(features.iloc[::-1], target.iloc[::-1])


def test_shuffled_index_is_rejected() -> None:
    features, target = _panel(n=800)
    order = np.random.default_rng(0).permutation(len(features))
    with pytest.raises(ValueError, match="sorted ascending"):
        _eda(features.iloc[order], target.iloc[order])


def test_duplicate_index_labels_are_rejected() -> None:
    """`.loc` on a duplicated label multiplies rows, quietly changing the sample."""
    features, target = _panel(n=400)
    duplicated_features = pd.concat([features, features.iloc[[-1]]])
    duplicated_target = pd.concat([target, target.iloc[[-1]]])
    with pytest.raises(ValueError, match="must be unique"):
        _eda(duplicated_features, duplicated_target)


def test_non_numeric_columns_are_excluded_with_a_warning() -> None:
    features, target = _panel(n=800)
    features = features.copy()
    features["regime_label"] = "trending"
    with pytest.warns(UserWarning, match="non-numeric"):
        eda = _eda(features, target)
    assert "regime_label" not in eda.feature_names
    assert not eda.analyze_feature_relevance().empty


def test_all_non_numeric_features_is_an_error_not_an_empty_table() -> None:
    idx = pd.date_range("2023-01-01", periods=400, freq="5min")
    features = pd.DataFrame({"label": ["a"] * 400}, index=idx)
    target = pd.Series(np.random.default_rng(0).normal(size=400), index=idx)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no numeric features"):
            _eda(features, target)


def test_unknown_requested_feature_names_warn() -> None:
    features, target = _panel(n=800)
    with pytest.warns(UserWarning, match="not columns of the frame"):
        eda = _eda(features, target, feature_names=["real", "does_not_exist"])
    assert eda.feature_names == ["real"]


def test_infinite_targets_are_excluded_from_the_train_partition() -> None:
    features, target = _panel(n=800)
    poisoned = target.copy()
    poisoned.iloc[10:20] = np.inf
    eda = _eda(features, poisoned)
    assert len(eda.train_index) == len(_eda(features, target).train_index) - 10
    assert np.isfinite(eda.target.loc[eda.train_index].to_numpy()).all()


# ============================================================================
# STATISTICAL ROBUSTNESS (regressions from the review pass)
# ============================================================================


def test_string_labelled_categorical_target_still_produces_a_table() -> None:
    """spearmanr on string labels used to raise, and the per-feature except swallowed it --
    emptying the entire relevance table instead of one column."""
    features, _ = _panel(n=1500)
    rng = np.random.default_rng(4)
    labels = pd.Series(rng.choice(["bull", "bear"], size=len(features)), index=features.index)
    result = _eda(features, labels, target_type=TargetType.CATEGORICAL).analyze_feature_relevance()
    assert not result.empty
    assert result["spearman_corr"].notna().any()
    assert result["hac_tstat"].notna().any()  # binary -> point-biserial is well defined


def test_hac_lag_length_is_capped_below_the_sample_size() -> None:
    """A horizon approaching the window length makes the Newey-West kernel degenerate."""
    features, target = _panel(n=600)
    eda = _eda(features, target, horizon=200, train_threshold=0.9)
    result = eda.analyze_feature_relevance()
    n_train = len(eda.train_index)
    assert result["n_eff"].dropna().between(1.0, n_train).all()
    assert np.isfinite(result["hac_tstat"].dropna()).all()


def test_walk_forward_pairs_is_and_oos_over_the_same_folds() -> None:
    features, target = _panel(n=4000)
    eda = _eda(features, target, mode=EDAMode.WALK_FORWARD)
    result = eda.analyze_feature_relevance()
    assert (result["n_folds"] <= len(eda.wf_folds)).all()
    assert (result["n_folds"] >= 1).all()
    # decay is only meaningful when both sides came from the same folds
    assert result["decay"].notna().any()


def test_constant_feature_does_not_break_the_clustered_heatmap() -> None:
    features, target = _panel(n=800)
    features = features.copy()
    features["constant"] = 1.0
    eda = _eda(features, target)
    assert eda.plot_correlation_matrix(cluster=True) is not None


def test_constant_feature_is_skipped_rather_than_crashing_relevance() -> None:
    features, target = _panel(n=800)
    features = features.copy()
    features["constant"] = 1.0
    result = _eda(features, target).analyze_feature_relevance()
    assert "constant" not in result["feature"].tolist()


def test_duplicate_feature_columns_are_rejected_with_a_clear_message() -> None:
    features, target = _panel(n=800)
    duplicated = features[["real", "fake"]].copy()
    duplicated.columns = ["real", "real"]
    with pytest.raises(ValueError, match="duplicate column name"):
        _eda(duplicated, target)


def test_vif_uses_an_intercept() -> None:
    """Without an intercept the auxiliary regression is forced through the origin and its R^2
    absorbs the feature mean, so independent features with large means report VIF in the
    thousands and get dropped as collinear."""
    rng = np.random.default_rng(0)
    n = 2000
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    features = pd.DataFrame({
        "atr": 50.0 + rng.normal(0, 1, n),
        "price": 1800.0 + rng.normal(0, 1, n),
    }, index=idx)
    target = pd.Series(rng.normal(size=n), index=idx)

    vif = _eda(features, target).compute_vif().set_index("feature")["vif"]
    assert vif.max() < 2.0, f"independent features should have VIF near 1, got {vif.to_dict()}"
    assert not _eda(features, target).compute_vif()["high_multicollinearity"].any()


def test_vif_still_detects_genuine_collinearity() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    a = rng.normal(size=n)
    features = pd.DataFrame({"a": a, "a_copy": a + rng.normal(0, 1e-3, n), "c": rng.normal(size=n)}, index=idx)
    target = pd.Series(rng.normal(size=n), index=idx)

    vif = _eda(features, target).compute_vif().set_index("feature")["vif"]
    assert vif["a"] > 100 and vif["a_copy"] > 100
    assert vif["c"] < 2.0
