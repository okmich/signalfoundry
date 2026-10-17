"""
Regression tests for label_eval_util.py.

Covers:
  - evaluate_all_labels_regime_returns_potentials does NOT mutate the caller's DataFrame
  - No log_return column is leaked onto the caller's frame
  - Original column set is preserved after the call
  - label_path_structure_statistics measures path statistics inside episodes, never across the join between two
  - regime_discriminability is an absolute, nearest-neighbour separation score
  - evaluate_regime_returns_potentials: attribution, flat regimes, timing, penalty and sign-fit fixes
  - label_transition_statistics: transitions, censored episodes, NaN and segment breaks
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.label_eval_util import (
    all_labels_path_structure_statistics,
    evaluate_all_labels_regime_returns_potentials,
    evaluate_regime_returns_potentials,
    fit_label_sign_map,
    label_path_structure_statistics,
    label_transition_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """Minimal DataFrame with open/close + two label columns (open required by evaluate_regime_returns_potentials)."""
    n = 200
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = closes * rng.uniform(0.998, 1.002, n)
    labels_a = rng.choice([0, 1, 2], size=n)
    labels_b = rng.choice([0, 1, 2], size=n)
    df = pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "label_a": labels_a.astype(float),
            "label_b": labels_b.astype(float),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    return df


# ---------------------------------------------------------------------------
# No-mutation tests
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_caller_columns_unchanged(self, price_df):
        """Function must not add columns (e.g. log_return) to caller's DataFrame."""
        original_cols = set(price_df.columns)

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a", "label_b"]
        )

        assert set(price_df.columns) == original_cols, (
            f"Columns added to caller frame: {set(price_df.columns) - original_cols}"
        )

    def test_no_log_return_column_on_caller(self, price_df):
        """Specifically, log_return must not appear on caller frame."""
        assert "log_return" not in price_df.columns

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a"]
        )

        assert "log_return" not in price_df.columns

    def test_close_column_values_unchanged(self, price_df):
        """Values in existing columns must be identical before and after."""
        close_before = price_df["close"].copy()

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a", "label_b"]
        )

        pd.testing.assert_series_equal(price_df["close"], close_before)

    def test_label_column_values_unchanged(self, price_df):
        """Label column values must be unchanged after the call."""
        label_before = price_df["label_a"].copy()

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a"]
        )

        pd.testing.assert_series_equal(price_df["label_a"], label_before)

    def test_preexisting_log_return_not_overwritten(self):
        """If caller already has a log_return column with custom values, it is preserved."""
        n = 100
        rng = np.random.default_rng(1)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        opens = closes * rng.uniform(0.998, 1.002, n)
        sentinel = np.full(n, 999.0)  # sentinel value to detect overwrite
        df = pd.DataFrame(
            {
                "open": opens,
                "close": closes,
                "label_a": rng.choice([0, 1, 2], size=n).astype(float),
                "log_return": sentinel,
            },
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        log_ret_before = df["log_return"].copy()

        evaluate_all_labels_regime_returns_potentials(df, labels_cols=["label_a"])

        pd.testing.assert_series_equal(df["log_return"], log_ret_before)

    def test_empty_labels_cols_returns_none_without_mutation(self, price_df):
        """Passing an empty labels_cols list returns None and does not mutate frame."""
        original_cols = set(price_df.columns)

        result = evaluate_all_labels_regime_returns_potentials(price_df, labels_cols=[])

        assert result is None
        assert set(price_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Path-structure statistics
# ---------------------------------------------------------------------------

def _episodes_frame(episodes):
    """Frame of (label, returns) episodes laid end to end, with a matching close series."""
    labels = np.concatenate([np.full(len(rets), label, dtype=float) for label, rets in episodes])
    returns = np.concatenate([np.asarray(rets, dtype=float) for _, rets in episodes])
    return pd.DataFrame({"label": labels, "returns": returns, "close": 100 * np.exp(np.cumsum(returns))})


def _stats(df):
    return label_path_structure_statistics(df, state_col="label", returns_col="returns").set_index("regime")


def _alternating_episodes(rng, n_episodes, make_a, make_b, min_len=10, max_len=30):
    episodes = []
    for _ in range(n_episodes):
        episodes.append((0, make_a(rng.integers(min_len, max_len))))
        episodes.append((1, make_b(rng.integers(min_len, max_len))))
    return episodes


def _ar1(rng, n, phi, scale=0.001):
    shocks = rng.standard_t(3, n) * scale
    out = np.empty(n)
    out[0] = shocks[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


class TestEpisodeStitching:
    def test_efficiency_ratio_does_not_cancel_opposite_trends(self):
        """Straight-line up and down episodes of one regime are each perfectly efficient; pooling them cancels to 0."""
        trend_up, trend_down = np.full(10, 0.001), np.full(10, -0.001)
        noise = np.tile([0.001, -0.001], 5)
        episodes = []
        for i in range(20):
            episodes += [(0, trend_up if i % 2 == 0 else trend_down), (1, noise)]

        stats_df = _stats(_episodes_frame(episodes))

        assert stats_df.loc[0, "efficiency_ratio"] == pytest.approx(1.0)
        assert stats_df.loc[1, "efficiency_ratio"] == pytest.approx(0.0)
        assert stats_df.loc[0, "efficiency_ratio_rw"] > 2.0
        assert stats_df.loc[1, "efficiency_ratio_rw"] < 0.5

    def test_autocorrelation_ignores_pairs_across_episode_joins(self):
        """Within every episode returns alternate (acf -1); only the joins between episodes repeat a sign."""
        episodes = []
        for i in range(30):
            episodes += [(0, [0.001, -0.001] if i % 2 == 0 else [-0.001, 0.001]), (1, [0.002, 0.002, 0.002])]

        stats_df = _stats(_episodes_frame(episodes))

        assert stats_df.loc[0, "autocorrelation_lag1"] == pytest.approx(-1.0)

    def test_nan_label_ends_an_episode(self):
        """A NaN label between two runs of the same regime keeps them as two episodes."""
        episodes = [(0, np.full(3, 0.001)), (np.nan, [0.001]), (0, np.full(3, 0.001)), (1, np.full(5, 0.001))]
        df = _episodes_frame(episodes)

        stats_df = _stats(df)

        assert stats_df.loc[0, "max_duration"] == 3
        assert stats_df.loc[0, "n_observations"] == 6

    def test_drawdown_resets_at_each_episode(self):
        """Ten separate one-bar 1% losses are ten 1% drawdowns, not one compounded ~9.6% drawdown."""
        episodes = []
        for _ in range(10):
            episodes += [(0, [-0.01]), (1, [0.001, 0.001])]

        stats_df = _stats(_episodes_frame(episodes))

        assert stats_df.loc[0, "max_drawdown"] == pytest.approx(-0.01)

    def test_volatility_stability_needs_a_window_inside_one_episode(self):
        """Episodes shorter than the rolling window give no volatility-stability windows."""
        rng = np.random.default_rng(0)
        episodes = []
        for _ in range(10):
            episodes += [(0, rng.normal(0, 0.001, 3)), (1, rng.normal(0, 0.001, 8))]

        stats_df = _stats(_episodes_frame(episodes))

        assert np.isnan(stats_df.loc[0, "volatility_stability"])
        assert np.isfinite(stats_df.loc[1, "volatility_stability"])


class TestEfficiencyRatioRW:
    @pytest.mark.parametrize("phi, low, high", [(0.0, 0.9, 1.1), (0.3, 1.15, np.inf), (-0.3, 0.0, 0.85)])
    def test_random_walk_baseline_and_serial_dependence(self, phi, low, high):
        """Fat-tailed, vol-clustered random walk scores ~1; momentum scores above, mean reversion below."""
        rng = np.random.default_rng(7)

        def make(n):
            return _ar1(rng, n, phi, scale=0.001 * np.exp(rng.normal(0, 0.5)))

        stats_df = _stats(_episodes_frame(_alternating_episodes(rng, 300, make, make, min_len=3, max_len=60)))

        assert low < stats_df.loc[0, "efficiency_ratio_rw"] < high

    def test_is_deterministic(self):
        rng = np.random.default_rng(3)
        df = _episodes_frame(_alternating_episodes(rng, 50, lambda n: rng.normal(0, 1e-3, n),
                                                   lambda n: rng.normal(0, 1e-3, n)))

        assert _stats(df).loc[0, "efficiency_ratio_rw"] == _stats(df).loc[0, "efficiency_ratio_rw"]


class TestDiscriminability:
    def test_two_state_score_is_not_a_constant(self):
        """The old relative z-score gave every 2-state model 0.7071; separation must now track the data."""
        rng = np.random.default_rng(11)
        same = _episodes_frame(_alternating_episodes(rng, 60, lambda n: rng.normal(0, 1e-3, n),
                                                     lambda n: rng.normal(0, 1e-3, n)))
        apart = _episodes_frame(_alternating_episodes(rng, 60, lambda n: rng.normal(0, 1e-3, n),
                                                      lambda n: rng.normal(0, 1e-2, n)))

        assert (_stats(same)["regime_discriminability"] < 0.4).all()
        assert (_stats(apart)["regime_discriminability"] > 0.95).all()

    def test_names_the_separating_metric(self):
        rng = np.random.default_rng(12)
        df = _episodes_frame(_alternating_episodes(rng, 60, lambda n: rng.normal(0, 1e-3, n),
                                                   lambda n: rng.normal(0, 1e-2, n)))

        assert (_stats(df)["discriminability_metric"] == "volatility").all()

    def test_nearest_neighbour_sets_the_score(self):
        """Regimes 0 and 1 share a profile while 2 is far away: 0 and 1 score low, 2 scores high."""
        rng = np.random.default_rng(13)
        episodes = []
        for _ in range(60):
            for label, scale in ((0, 1e-3), (1, 1e-3), (2, 1e-2)):
                episodes.append((label, rng.normal(0, scale, rng.integers(10, 30))))

        scores = _stats(_episodes_frame(episodes))["regime_discriminability"]

        assert scores[0] < 0.4 and scores[1] < 0.4
        assert scores[2] > 0.95

    def test_too_few_episodes_gives_nan(self):
        rng = np.random.default_rng(14)
        df = _episodes_frame([(0, rng.normal(0, 1e-3, 50)), (1, rng.normal(0, 1e-2, 50))] * 3)

        assert _stats(df)["regime_discriminability"].isna().all()


class TestPathStructureCompatibility:
    def test_columns_and_constant_prices(self):
        """Existing columns are kept, new ones added, and zero returns (wfa_utils' synthetic prices) do not raise."""
        rng = np.random.default_rng(15)
        df = pd.DataFrame({"close": np.full(200, 100.0), "state": rng.choice([0, 1], 200)})
        df["returns"] = df["close"].pct_change()

        stats_df = label_path_structure_statistics(df, state_col="state", returns_col="returns")

        for col in ["mean_duration", "regime_stability", "autocorrelation_lag1", "trend_persistence",
                    "efficiency_ratio", "volatility_stability", "max_drawdown", "regime_discriminability",
                    "efficiency_ratio_rw", "discriminability_metric"]:
            assert col in stats_df.columns

    def test_all_labels_wrapper(self):
        rng = np.random.default_rng(16)
        df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 500))),
                           "label_a": rng.choice([0, 1], 500), "label_b": rng.choice([0, 1, 2], 500)})

        result = all_labels_path_structure_statistics(df, state_cols=["label_a", "label_b"])

        assert set(result["algo"]) == {"label_a", "label_b"}


# ---------------------------------------------------------------------------
# evaluate_regime_returns_potentials
# ---------------------------------------------------------------------------

def _bars(labels, returns=None, seed=0):
    """Bars whose open-to-close log return is ``returns`` (random when None); open = 100 on every bar."""
    labels = np.asarray(labels, dtype=float)
    if returns is None:
        returns = np.random.default_rng(seed).normal(0, 1e-3, labels.size)
    return pd.DataFrame({"open": np.full(labels.size, 100.0), "close": 100.0 * np.exp(returns), "label": labels})


class TestRegimeReturnsPotentials:
    def test_one_bar_run_keeps_its_own_label(self):
        """A 1-bar run of label 2 used to be credited to the next label."""
        labels = [0] * 10 + [2] + [0] * 10 + [1] * 10
        metrics, regimes = evaluate_regime_returns_potentials(_bars(labels), label_sign_map={0: -1, 1: 1, 2: 1})

        one_bar = regimes[regimes["label"] == 2]
        assert len(one_bar) == 1 and one_bar["regime_length"].iloc[0] == 1
        assert metrics.set_index("label").loc[2, "count"] == 1

    def test_flat_regimes_are_kept_and_no_trade_is_a_share_of_time(self):
        """Label 0 maps to 0: its regimes still count, and no_trade% is the share of bars out of the market."""
        labels = np.repeat([0, 1, 0, 1], 25)
        metrics, regimes = evaluate_regime_returns_potentials(_bars(labels), label_sign_map={0: 0, 1: 1})
        rows = metrics.set_index("label")

        assert rows.loc[0, "count"] == 2 and rows.loc[0, "n_trades"] == 0
        assert rows.loc["overall", "count"] == 4
        # bars 1..99 have a decision label (bar 0 has none); label 0 decides bars 1-25 and 51-75
        assert rows.loc["overall", "no_trade%"] == pytest.approx(100 * 50 / 99, abs=0.01)
        assert rows.loc[0, "avg_dwell"] > 0

    def test_progressive_skip_waits_extra_bars_and_zero_means_one(self):
        labels = np.repeat([0, 1] * 5, 6)
        df = _bars(labels)
        sign_map = {0: -1, 1: 1}
        traded = {k: evaluate_regime_returns_potentials(df, progressive_skip=k, label_sign_map=sign_map)[1]
                  ["total_trades"].sum() for k in (0, 1, 2)}
        assert traded[0] == traded[1]
        assert traded[2] < traded[1]

    def test_no_bar_trades_on_its_own_label(self):
        """A label that is the sign of its own bar's return: same-bar trading would win every bar."""
        returns = np.random.default_rng(3).normal(0, 1e-3, 2000)
        labels = (returns > 0).astype(float)
        _, regimes = evaluate_regime_returns_potentials(_bars(labels, returns), label_sign_map={0: -1, 1: 1})
        assert abs(regimes["correct_direction_count"].sum() / regimes["total_trades"].sum() - 0.5) < 0.05

    def test_wrong_direction_penalty_is_charged_once(self):
        df = _bars(np.repeat([0, 1] * 10, 8), seed=4)
        sign_map = {0: -1, 1: 1}
        plain = evaluate_regime_returns_potentials(df, label_sign_map=sign_map)[0].set_index("label").loc["overall"]
        penalised = evaluate_regime_returns_potentials(df, label_sign_map=sign_map, wrong_direction_penalty=0.5)[0]
        penalised = penalised.set_index("label").loc["overall"]

        assert penalised["total_return"] == pytest.approx(plain["total_return"])
        assert penalised["total_wrong_direction_penalty"] == pytest.approx(0.5 * plain["adv_ret_cum_raw"])
        assert penalised["net_return_after_costs"] == pytest.approx(plain["total_return"]
                                                                    - 0.5 * plain["adv_ret_cum_raw"])

    def test_nan_label_is_not_bridged(self):
        """The bar after a NaN label has no confirmed sign, so it is not traded."""
        labels = np.r_[np.ones(10), np.nan, np.ones(10)]
        _, regimes = evaluate_regime_returns_potentials(_bars(np.r_[labels, np.zeros(5)]), label_sign_map={0: -1, 1: 1})
        # decision regimes of label 1: bars 1..10 and bars 12..21 (bar 11's decision label is the NaN)
        ones = regimes[regimes["label"] == 1]
        assert ones["regime_length"].tolist() == [10, 10]
        assert ones["total_trades"].tolist() == [10, 10]

    def test_sign_map_is_fitted_on_the_next_bar_return(self):
        """Mean-reverting bars labelled by the sign of their own return: the next bar reverses, so the fitted map
        fades the label. A same-bar fit would have followed it."""
        rng = np.random.default_rng(5)
        returns = np.zeros(5000)
        for t in range(1, returns.size):
            returns[t] = -0.4 * returns[t - 1] + rng.normal(0, 1e-3)
        df = _bars((returns > 0).astype(float), returns)
        assert fit_label_sign_map(df, "label") == {0.0: 1, 1.0: -1}


# ---------------------------------------------------------------------------
# label_transition_statistics
# ---------------------------------------------------------------------------

class TestLabelTransitionStatistics:
    def test_transition_matrix_and_mean_self_transition(self):
        result = label_transition_statistics(pd.Series([0, 0, 1, 1, 1, 0]))
        np.testing.assert_allclose(result.transition_matrix.to_numpy(), [[0.5, 0.5], [1 / 3, 2 / 3]])
        assert result.mean_self_transition == pytest.approx((0.5 + 2 / 3) / 2)
        assert result.occupancy["bars"].tolist() == [3, 3]

    def test_episodes_cut_by_the_sample_ends_are_censored(self):
        dwell = label_transition_statistics(pd.Series([0, 0, 1, 1, 1, 0])).dwell
        assert dwell.loc[1, ["episodes", "censored", "mean"]].tolist() == [1, 0, 3.0]
        assert dwell.loc[0, "episodes"] == 2 and dwell.loc[0, "censored"] == 2 and np.isnan(dwell.loc[0, "mean"])

    def test_nan_and_segment_change_break_episodes(self):
        result = label_transition_statistics(pd.Series([0, 0, np.nan, 0, 0]))
        assert result.dwell.loc[0, "episodes"] == 2
        assert result.transition_matrix.loc[0, 0] == 1.0  # two 0->0 steps, none across the NaN

        segmented = label_transition_statistics(pd.Series([0, 0, 0, 0]), segment=pd.Series([1, 1, 2, 2]))
        assert segmented.dwell.loc[0, "episodes"] == 2

    def test_unused_category_does_not_break_the_matrix(self):
        """The retired LabelEvaluator sized the matrix after dropping rows and indexed it before: IndexError."""
        labels = pd.Series(pd.Categorical([0, 0, 2, 2], categories=[0, 1, 2]))
        matrix = label_transition_statistics(labels).transition_matrix
        assert matrix.shape == (3, 3)
        assert matrix.loc[1].isna().all()
