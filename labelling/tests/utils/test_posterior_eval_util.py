"""Tests for the posterior-native evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.posterior_eval_util import (
    all_posteriors_dynamics_statistics, axis_ranks, count_sign_divergence,
    discover_posterior_variants, evaluate_all_posteriors_returns_potentials,
    evaluate_posterior_returns_potentials, extract_posteriors, posterior_dynamics_statistics,
    posterior_weighted_mean, _backtest_position, _effective_sample_size, _gated_position,
    _posterior_weighted_std, _soft_position)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _persistent_states(T: int, K: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.zeros(T, dtype=int)
    stay = 0.97
    off = (1.0 - stay) / (K - 1)
    trans = np.full((K, K), off)
    np.fill_diagonal(trans, stay)
    for t in range(1, T):
        states[t] = rng.choice(K, p=trans[states[t - 1]])
    return states


def _onehot(states: np.ndarray, K: int) -> np.ndarray:
    probs = np.zeros((len(states), K), dtype=float)
    probs[np.arange(len(states)), states] = 1.0
    return probs


def _build_df(T: int = 600, K: int = 3, seed: int = 0, soft: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = _persistent_states(T, K, seed)
    mu = {0: -0.0008, 1: 0.0, 2: 0.0008}
    sd = {0: 0.001, 1: 0.0005, 2: 0.001}
    ret = np.array([rng.normal(mu[s % 3], sd[s % 3]) for s in states])
    close = 100.0 * np.exp(np.cumsum(ret))
    # axis feature monotonically tied to state index so axis ranks are deterministic
    axis = states.astype(float) + rng.normal(0, 0.01, size=T)
    df = pd.DataFrame({"close": close, "returns": ret, "state": states, "axis_feat": axis},
                      index=pd.date_range("2024-01-01", periods=T, freq="5min"))
    if soft:
        logits = np.full((T, K), 0.0)
        logits[np.arange(T), states] = 3.0
        logits += rng.normal(0, 0.5, size=(T, K))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    else:
        probs = _onehot(states, K)
    for k in range(K):
        df[f"post_v1_s{k}"] = probs[:, k]
    return df


# ---------------------------------------------------------------------------
# discover_posterior_variants
# ---------------------------------------------------------------------------

def test_discover_variants_orders_states_and_ignores_other_columns():
    df = pd.DataFrame({"close": [1.0], "post_a_s1": [0.5], "post_a_s0": [0.5],
                       "post_b_s0": [1.0], "lbl_x": [0]})
    variants = discover_posterior_variants(df)
    assert set(variants) == {"a", "b"}
    assert variants["a"] == ["post_a_s0", "post_a_s1"]  # ordered by state index, not column order


def test_discover_variants_handles_multi_underscore_names():
    df = pd.DataFrame({"post_hmm_mm_lambda_s0": [1.0], "post_hmm_mm_lambda_s1": [0.0]})
    variants = discover_posterior_variants(df)
    assert list(variants) == ["hmm_mm_lambda"]


def test_discover_variants_rejects_non_contiguous_states():
    df = pd.DataFrame({"post_a_s0": [1.0], "post_a_s2": [0.0]})
    with pytest.raises(ValueError, match="non-contiguous"):
        discover_posterior_variants(df)


# ---------------------------------------------------------------------------
# extract_posteriors validation
# ---------------------------------------------------------------------------

def test_extract_posteriors_returns_matrix():
    df = _build_df(T=50)
    probs = extract_posteriors(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"])
    assert probs.shape == (50, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_extract_posteriors_rejects_nan():
    df = _build_df(T=20)
    df.loc[df.index[0], "post_v1_s0"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        extract_posteriors(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"])


def test_extract_posteriors_rejects_non_simplex_rows():
    df = pd.DataFrame({"post_v1_s0": [0.3, 0.5], "post_v1_s1": [0.3, 0.5]})  # rows sum to 0.6, 1.0
    with pytest.raises(ValueError, match="sum to 1"):
        extract_posteriors(df, ["post_v1_s0", "post_v1_s1"])


def test_extract_posteriors_rejects_out_of_range():
    df = pd.DataFrame({"post_v1_s0": [1.3], "post_v1_s1": [-0.3]})
    with pytest.raises(ValueError, match="lie in"):
        extract_posteriors(df, ["post_v1_s0", "post_v1_s1"])


# ---------------------------------------------------------------------------
# weighting primitives
# ---------------------------------------------------------------------------

def test_posterior_weighted_mean_onehot_equals_group_mean():
    states = np.array([0, 0, 1, 1, 2])
    probs = _onehot(states, 3)
    values = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    got = posterior_weighted_mean(probs, values)
    np.testing.assert_allclose(got, [2.0, 15.0, 100.0])


def test_posterior_weighted_mean_masks_nan_values():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([2.0, np.nan, 5.0])
    got = posterior_weighted_mean(probs, values)
    np.testing.assert_allclose(got, [2.0, 5.0])  # NaN row dropped from state 0


def test_posterior_weighted_mean_soft_weights():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    values = np.array([0.0, 4.0])
    got = posterior_weighted_mean(probs, values)
    np.testing.assert_allclose(got, [2.0, 2.0])  # equal weights -> both states see mean 2


def test_effective_sample_size_onehot_equals_counts():
    states = np.array([0, 0, 0, 1, 2])
    probs = _onehot(states, 3)
    np.testing.assert_allclose(_effective_sample_size(probs), [3.0, 1.0, 1.0])


def test_effective_sample_size_uniform_is_T_over_K():
    probs = np.full((10, 2), 0.5)
    # (sum g)^2 / sum g^2 = 5^2 / (10 * 0.25) = 25 / 2.5 = 10
    np.testing.assert_allclose(_effective_sample_size(probs), [10.0, 10.0])


def test_posterior_weighted_std_onehot_equals_group_std():
    states = np.array([0, 0, 1, 1])
    probs = _onehot(states, 2)
    values = np.array([1.0, 3.0, 10.0, 10.0])
    means = posterior_weighted_mean(probs, values)
    got = _posterior_weighted_std(probs, values, means)
    np.testing.assert_allclose(got, [1.0, 0.0])  # std of {1,3}=1, std of {10,10}=0


def test_axis_ranks_ascending_and_descending():
    scores = np.array([0.5, 0.1, 0.9])
    np.testing.assert_array_equal(axis_ranks(scores, ascending=True), [1, 0, 2])
    np.testing.assert_array_equal(axis_ranks(scores, ascending=False), [1, 2, 0])


# ---------------------------------------------------------------------------
# count_sign_divergence
# ---------------------------------------------------------------------------

def test_count_sign_divergence_counts_only_genuine_disagreements():
    signs = {0: 1, 1: -1, 2: 0, 3: 1}
    pw = np.array([-0.001, -0.002, 0.5, np.nan])
    # state 0: mapped +1, pw -1 -> divergent. state 1: agree. state 2: flat -> skip.
    # state 3: pw nan -> skip.
    assert count_sign_divergence(signs, pw) == 1


def test_count_sign_divergence_zero_when_all_agree():
    signs = {0: 1, 1: -1}
    pw = np.array([0.01, -0.01])
    assert count_sign_divergence(signs, pw) == 0


# ---------------------------------------------------------------------------
# posterior_dynamics_statistics
# ---------------------------------------------------------------------------

def test_dynamics_onehot_is_maximally_decisive():
    df = _build_df(T=600, soft=False)
    stats = posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], returns_col="returns")
    # one-hot => zero entropy, perplexity 1, fully decisive, occupancy == argmax frequency
    assert stats["mean_entropy"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert stats["perplexity"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert stats["decisiveness"].iloc[0] == pytest.approx(1.0)
    # argmax_freq_pct is rounded to 4 dp (matching the hard-label table); occupancy is not
    np.testing.assert_allclose(stats["occupancy"], stats["argmax_freq_pct"] / 100.0, atol=1e-4)


def test_dynamics_pw_mean_return_matches_group_mean_onehot():
    df = _build_df(T=600, soft=False)
    stats = posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], returns_col="returns")
    for _, row in stats.iterrows():
        k = int(row["state"])
        expected = df.loc[df["state"] == k, "returns"].mean()
        assert row["pw_mean_return"] == pytest.approx(expected, rel=1e-6)


def test_dynamics_soft_posterior_is_less_decisive_than_onehot():
    soft = posterior_dynamics_statistics(_build_df(soft=True), ["post_v1_s0", "post_v1_s1", "post_v1_s2"],
                                         returns_col="returns")
    hard = posterior_dynamics_statistics(_build_df(soft=False), ["post_v1_s0", "post_v1_s1", "post_v1_s2"],
                                         returns_col="returns")
    assert soft["mean_entropy"].iloc[0] > hard["mean_entropy"].iloc[0]
    assert soft["perplexity"].iloc[0] > hard["perplexity"].iloc[0]


def test_dynamics_axis_orders_states_by_rank():
    df = _build_df(T=600, soft=True)
    stats = posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], axis_col="axis_feat",
                                          returns_col="returns", rank_labels=["low", "mid", "high"])
    # axis_feat ~ state index, so rank order should be the natural 0,1,2 and labels follow
    assert list(stats["axis_rank"]) == [0, 1, 2]
    assert list(stats["axis_label"]) == ["low", "mid", "high"]
    assert stats["axis_score"].is_monotonic_increasing


def test_dynamics_without_axis_sorts_by_occupancy_desc():
    df = _build_df(T=600, soft=True)
    stats = posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], returns_col="returns")
    assert "axis_rank" not in stats.columns
    assert stats["occupancy"].is_monotonic_decreasing


def test_dynamics_raises_on_missing_axis_col():
    df = _build_df(T=100)
    with pytest.raises(ValueError, match="axis_col"):
        posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], axis_col="nope")


# ---------------------------------------------------------------------------
# all_posteriors_dynamics_statistics
# ---------------------------------------------------------------------------

def _two_variant_df(T: int = 600) -> pd.DataFrame:
    df = _build_df(T=T, soft=True, seed=1)
    df2 = _build_df(T=T, soft=False, seed=1)
    for k in range(3):
        df[f"post_v2_s{k}"] = df2[f"post_v1_s{k}"].to_numpy()
    return df


def test_all_dynamics_concats_variants_with_provenance():
    df = _two_variant_df()
    out = all_posteriors_dynamics_statistics(df, axis_col="axis_feat", returns_col="returns",
                                             provenance={"v1": "smoothing", "v2": "fixed_lag"})
    assert set(out["algo"]) == {"v1", "v2"}
    assert set(out["provenance"]) == {"smoothing", "fixed_lag"}
    assert (out[out["algo"] == "v2"]["provenance"] == "fixed_lag").all()


def test_all_dynamics_variant_subset():
    df = _two_variant_df()
    out = all_posteriors_dynamics_statistics(df, variants=["v2"], returns_col="returns")
    assert set(out["algo"]) == {"v2"}


def test_all_dynamics_unknown_variant_raises():
    df = _two_variant_df()
    with pytest.raises(ValueError, match="no post_"):
        all_posteriors_dynamics_statistics(df, variants=["ghost"], returns_col="returns")


# ---------------------------------------------------------------------------
# position builders
# ---------------------------------------------------------------------------

def test_soft_position_is_probability_weighted_signs():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    signs = {0: -1, 1: 0, 2: 1}
    got = _soft_position(probs, signs)
    np.testing.assert_allclose(got, [-0.7 + 0.1, -0.1 + 0.8])  # [-0.6, 0.7]
    assert (np.abs(got) <= 1.0).all()


def test_gated_position_flat_below_tau():
    probs = np.array([[0.45, 0.45, 0.10], [0.10, 0.10, 0.80]])
    signs = {0: -1, 1: 0, 2: 1}
    got = _gated_position(probs, signs, tau=0.5)
    np.testing.assert_allclose(got, [0.0, 1.0])  # row0 max 0.45<=tau -> flat; row1 argmax=2 -> +1


# ---------------------------------------------------------------------------
# _backtest_position (hand-computed)
# ---------------------------------------------------------------------------

def test_backtest_position_hand_computed():
    position = np.array([0.0, 1.0, 1.0, -1.0])
    fwd = np.array([0.1, -0.2, 0.3, 0.05])
    res = _backtest_position(position, fwd, whipsaw_cost=0.1, progressive_skip=0)
    assert res["n_bars"] == 4
    assert res["gross_return"] == pytest.approx(0.05)        # 0 -0.2 +0.3 -0.05
    assert res["turnover"] == pytest.approx(3.0)             # |0|+|1|+|0|+|2|
    assert res["total_whipsaw_cost"] == pytest.approx(0.3)
    assert res["net_return"] == pytest.approx(-0.25)
    assert res["exposure"] == pytest.approx(0.75)
    assert res["avg_abs_position"] == pytest.approx(0.75)
    assert res["n_position_changes"] == 2
    assert res["win_rate"] == pytest.approx(1 / 3)
    assert res["profit_factor"] == pytest.approx(0.3 / 0.25)
    assert res["max_drawdown"] == pytest.approx(0.3)


def test_backtest_skip_lags_position():
    position = np.array([1.0, 1.0, 1.0])
    fwd = np.array([0.1, 0.1, 0.1])
    res = _backtest_position(position, fwd, whipsaw_cost=0.0, progressive_skip=1)
    # first bar zeroed by lag -> only 2 active bars
    assert res["exposure"] == pytest.approx(2 / 3)


def test_backtest_all_flat_has_undefined_profit_factor():
    res = _backtest_position(np.zeros(5), np.full(5, 0.01), whipsaw_cost=0.0, progressive_skip=0)
    assert res["exposure"] == 0.0
    assert res["win_rate"] == 0.0
    assert np.isnan(res["profit_factor"])


def test_backtest_rejects_negative_skip():
    with pytest.raises(ValueError, match="progressive_skip"):
        _backtest_position(np.ones(3), np.ones(3), whipsaw_cost=0.0, progressive_skip=-1)


# ---------------------------------------------------------------------------
# evaluate_posterior_returns_potentials
# ---------------------------------------------------------------------------

def test_returns_potentials_has_soft_and_gated_rows():
    df = _build_df(T=600, soft=True)
    out = evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], whipsaw_cost=0.0002)
    assert list(out["position"]) == ["soft", "gated"]
    assert (out["n_states"] == 3).all()
    assert out["n_sign_divergent"].dtype.kind in "iu"
    assert all(isinstance(m, dict) for m in out["sign_map"])


def test_returns_potentials_soft_churns_more_than_gated():
    df = _build_df(T=800, soft=True)
    out = evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], whipsaw_cost=0.0)
    soft_turnover = out.loc[out["position"] == "soft", "turnover"].iloc[0]
    gated_turnover = out.loc[out["position"] == "gated", "turnover"].iloc[0]
    assert soft_turnover > gated_turnover  # continuous position nudges every bar


def test_returns_potentials_calibration_with_truth_col():
    df = _build_df(T=600, soft=True)
    out = evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], truth_col="state")
    assert "ece" in out.columns and "brier_score" in out.columns and "nll" in out.columns
    assert (out["ece"] >= 0).all()


def test_returns_potentials_no_calibration_by_default():
    df = _build_df(T=600, soft=True)
    out = evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"])
    assert "ece" not in out.columns


# ---------------------------------------------------------------------------
# evaluate_all_posteriors_returns_potentials
# ---------------------------------------------------------------------------

def test_all_returns_potentials_two_rows_per_variant():
    df = _two_variant_df()
    out = evaluate_all_posteriors_returns_potentials(df, whipsaw_cost=0.0002,
                                                     provenance={"v1": "smoothing", "v2": "fixed_lag"})
    assert len(out) == 4  # 2 variants x {soft, gated}
    assert set(out["algo"]) == {"v1", "v2"}
    assert set(out["provenance"]) == {"smoothing", "fixed_lag"}


def test_all_returns_potentials_truth_cols_routing():
    df = _two_variant_df()
    out = evaluate_all_posteriors_returns_potentials(df, truth_cols={"v1": "state"})
    # calibration columns present (only v1 had truth, v2 rows get NaN on concat)
    assert "ece" in out.columns
    assert out.loc[out["algo"] == "v1", "ece"].notna().all()


# ---------------------------------------------------------------------------
# validation gaps (review point 5)
# ---------------------------------------------------------------------------

def test_returns_potentials_missing_price_col_raises():
    df = _build_df(T=100, soft=True).drop(columns=["close"])
    with pytest.raises(ValueError, match="price_col"):
        evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"])


def test_dynamics_missing_both_returns_and_price_raises():
    df = _build_df(T=100, soft=True).drop(columns=["close", "returns"])
    with pytest.raises(ValueError, match="Neither returns_col"):
        posterior_dynamics_statistics(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"])


def test_all_functions_return_none_when_no_variants():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    assert all_posteriors_dynamics_statistics(df) is None
    assert evaluate_all_posteriors_returns_potentials(df) is None


def test_single_state_posterior_is_rejected():
    # a posterior over a single state carries no regime information -> clean rejection
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "returns": [0.0, 0.1, 0.2], "post_v_s0": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="K >= 2"):
        posterior_dynamics_statistics(df, ["post_v_s0"], returns_col="returns")


def test_invalid_truth_col_single_class_raises():
    df = _build_df(T=300, soft=True)
    df["bad_truth"] = 0  # only one class -> calibration undefined
    with pytest.raises(ValueError):
        evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], truth_col="bad_truth")


def test_sign_map_override_is_used_verbatim():
    df = _build_df(T=400, soft=True)
    forced = {0: 1, 1: 1, 2: 1}  # force all-long regardless of in-sample returns
    out = evaluate_posterior_returns_potentials(df, ["post_v1_s0", "post_v1_s1", "post_v1_s2"], sign_map=forced)
    assert out["sign_map"].iloc[0] == forced
    # soft position is then a non-negative long-only exposure
    assert out.loc[out["position"] == "gated", "avg_abs_position"].iloc[0] >= 0.0