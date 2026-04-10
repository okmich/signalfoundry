"""
Tests for MTDC pipeline components:
  - generate_theta_pool
  - select_top_thresholds
  - train_ga_weights
  - run_mtdc_algorithm
  - MTDCPipeline
  - search_optimal_k

All GP/GA parameters are minimised for fast test execution.
The ITA HMM fixture reuses the same patterns as the C+GP+TS tests.
"""
import numpy as np
import pandas as pd
import pytest

try:
    from deap import base  # noqa: F401
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

from okmich_quant_features.directional_change import parse_dc_events, log_r
from okmich_quant_pipeline.directional_change import (
    MTDCPipeline,
    MTDC_THETA_MAX_DEFAULT,
    MTDC_THETA_MIN_DEFAULT,
    MTDC_THETA_STEP_DEFAULT,
    ConsensusMode,
    FitnessMode,
    fit_rcd_hmm,
    generate_theta_pool,
    run_mtdc_algorithm,
    s1_state_index,
    search_optimal_k,
    select_top_thresholds,
    train_ga_weights,
)

pytestmark = pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not installed")

# ── Reduced GP/GA parameters for fast tests ──────────────────────────────────
_GP_GENS = 3
_GP_POP = 20
_GA_GENS = 5
_GA_POP = 20
_THETA_POOL = [0.01, 0.015, 0.02, 0.025, 0.03]  # tiny pool


def _make_prices(n: int = 800, amplitude: float = 0.015, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 100.0 * np.exp(0.0001 * t)
    noise = rng.normal(0, 0.006, n)
    cycle1 = amplitude * np.sin(2 * np.pi * t / 30)
    cycle2 = (amplitude * 0.4) * np.sin(2 * np.pi * t / 7)
    return pd.Series(trend * (1 + cycle1 + cycle2 + noise))


@pytest.fixture(scope='module')
def prices():
    return _make_prices(800)


@pytest.fixture(scope='module')
def hmm_and_s1(prices):
    trends = parse_dc_events(prices.iloc[:600], theta=0.01)
    rcd = log_r(trends).dropna()
    hmm = fit_rcd_hmm(rcd.values.reshape(-1, 1))
    s1 = s1_state_index(hmm)
    return hmm, s1


@pytest.fixture(scope='module')
def top_pairs(prices):
    return select_top_thresholds(
        prices.iloc[:600], k=2, theta_pool=_THETA_POOL,
        gp_n_generations=_GP_GENS, gp_population_size=_GP_POP, random_seed=42,
    )


@pytest.fixture(scope='module')
def ga_weights(prices, top_pairs):
    thetas = [t for t, _ in top_pairs]
    models = [m for _, m in top_pairs]
    return train_ga_weights(
        prices.iloc[:600], models, thetas,
        population_size=_GA_POP, n_generations=_GA_GENS, random_seed=42,
    )


@pytest.fixture(scope='module')
def algo_result(prices, hmm_and_s1, top_pairs, ga_weights):
    hmm, s1 = hmm_and_s1
    thetas = [t for t, _ in top_pairs]
    models = [m for _, m in top_pairs]
    return run_mtdc_algorithm(
        prices.iloc[600:], hmm, s1, models, thetas, ga_weights,
        initial_capital=10_000.0,
    )


# ---------------------------------------------------------------------------
# generate_theta_pool tests
# ---------------------------------------------------------------------------

class TestGenerateThetaPool:
    def test_returns_list(self):
        pool = generate_theta_pool()
        assert isinstance(pool, list)

    def test_default_pool_length(self):
        pool = generate_theta_pool()
        expected = int(round((MTDC_THETA_MAX_DEFAULT - MTDC_THETA_MIN_DEFAULT) / MTDC_THETA_STEP_DEFAULT)) + 1
        assert len(pool) == expected

    def test_default_pool_starts_at_min(self):
        pool = generate_theta_pool()
        assert pool[0] == pytest.approx(MTDC_THETA_MIN_DEFAULT)

    def test_default_pool_ends_near_max(self):
        pool = generate_theta_pool()
        assert pool[-1] == pytest.approx(MTDC_THETA_MAX_DEFAULT, abs=MTDC_THETA_STEP_DEFAULT)

    def test_pool_is_sorted_ascending(self):
        pool = generate_theta_pool(0.01, 0.05, 0.01)
        assert pool == sorted(pool)

    def test_custom_pool(self):
        pool = generate_theta_pool(0.01, 0.03, 0.01)
        assert len(pool) == 3
        assert pool[0] == pytest.approx(0.01)
        assert pool[-1] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# select_top_thresholds tests
# ---------------------------------------------------------------------------

class TestSelectTopThresholds:
    def test_returns_list_of_tuples(self, top_pairs):
        assert isinstance(top_pairs, list)
        for item in top_pairs:
            assert len(item) == 2

    def test_length_equals_k(self, top_pairs):
        assert len(top_pairs) == 2

    def test_first_element_is_float(self, top_pairs):
        for theta, _ in top_pairs:
            assert isinstance(theta, float)

    def test_second_element_is_stdc_model(self, top_pairs):
        for _, model in top_pairs:
            for key in ('gp_func', 'gp_rmse', 'classifier', 'theta'):
                assert key in model

    def test_sorted_by_gp_rmse(self, top_pairs):
        rmses = [model['gp_rmse'] for _, model in top_pairs]
        assert rmses == sorted(rmses)

    def test_invalid_k_raises(self, prices):
        with pytest.raises(ValueError):
            select_top_thresholds(prices.iloc[:600], k=0, theta_pool=_THETA_POOL,
                                   gp_n_generations=_GP_GENS, gp_population_size=_GP_POP)

    def test_thetas_from_pool(self, top_pairs):
        thetas = [t for t, _ in top_pairs]
        for theta in thetas:
            assert theta in _THETA_POOL


# ---------------------------------------------------------------------------
# train_ga_weights tests
# ---------------------------------------------------------------------------

class TestTrainGAWeights:
    def test_returns_ndarray(self, ga_weights):
        assert isinstance(ga_weights, np.ndarray)

    def test_shape_matches_k(self, ga_weights, top_pairs):
        assert ga_weights.shape == (len(top_pairs),)

    def test_weights_in_zero_one(self, ga_weights):
        assert (ga_weights >= 0.0).all()
        assert (ga_weights <= 1.0).all()

    def test_mismatched_lengths_raises(self, prices, top_pairs):
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        with pytest.raises(ValueError, match='same length'):
            train_ga_weights(prices.iloc[:600], models, thetas[:-1],
                             population_size=_GA_POP, n_generations=_GA_GENS)

    def test_invalid_fitness_mode_raises(self, prices, top_pairs):
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        with pytest.raises(ValueError, match="FitnessMode"):
            train_ga_weights(prices.iloc[:600], models, thetas,
                             fitness_mode='invalid',
                             population_size=_GA_POP, n_generations=_GA_GENS)

    def test_fitness_return_mode_runs(self, prices, top_pairs):
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        w = train_ga_weights(prices.iloc[:600], models, thetas,
                             fitness_mode='return',
                             population_size=_GA_POP, n_generations=_GA_GENS, random_seed=7)
        assert w.shape == (len(top_pairs),)

    def test_reproducible_with_same_seed(self, prices, top_pairs):
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        w1 = train_ga_weights(prices.iloc[:600], models, thetas,
                              population_size=_GA_POP, n_generations=_GA_GENS, random_seed=99)
        w2 = train_ga_weights(prices.iloc[:600], models, thetas,
                              population_size=_GA_POP, n_generations=_GA_GENS, random_seed=99)
        np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# FitnessMode / ConsensusMode enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_fitness_mode_sharpe_value(self):
        assert FitnessMode.SHARPE == 'sharpe'

    def test_fitness_mode_return_value(self):
        assert FitnessMode.RETURN == 'return'

    def test_fitness_mode_from_str(self):
        assert FitnessMode('sharpe') is FitnessMode.SHARPE
        assert FitnessMode('return') is FitnessMode.RETURN

    def test_fitness_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="FitnessMode"):
            FitnessMode('invalid')

    def test_consensus_mode_weight_value(self):
        assert ConsensusMode.WEIGHT == 'weight'

    def test_consensus_mode_majority_value(self):
        assert ConsensusMode.MAJORITY == 'majority'

    def test_consensus_mode_from_str(self):
        assert ConsensusMode('weight') is ConsensusMode.WEIGHT
        assert ConsensusMode('majority') is ConsensusMode.MAJORITY

    def test_consensus_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="ConsensusMode"):
            ConsensusMode('invalid')

    def test_run_mtdc_accepts_consensus_mode_enum(self, prices, hmm_and_s1, top_pairs, ga_weights):
        hmm, s1 = hmm_and_s1
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        result = run_mtdc_algorithm(
            prices.iloc[600:], hmm, s1, models, thetas, ga_weights,
            consensus_mode=ConsensusMode.WEIGHT,
        )
        assert 'final_capital' in result

    def test_run_mtdc_invalid_consensus_raises(self, prices, hmm_and_s1, top_pairs, ga_weights):
        hmm, s1 = hmm_and_s1
        thetas = [t for t, _ in top_pairs]
        models = [m for _, m in top_pairs]
        with pytest.raises(ValueError, match="ConsensusMode"):
            run_mtdc_algorithm(
                prices.iloc[600:], hmm, s1, models, thetas, ga_weights,
                consensus_mode='invalid',
            )


# ---------------------------------------------------------------------------
# run_mtdc_algorithm tests
# ---------------------------------------------------------------------------

class TestRunMTDCAlgorithm:
    def test_returns_dict_with_required_keys(self, algo_result):
        for key in ('final_capital', 'cumulative_return', 'max_drawdown',
                    'n_trades', 'n_winners', 'win_ratio', 'profit_factor',
                    'sharpe', 'trade_log'):
            assert key in algo_result

    def test_final_capital_positive(self, algo_result):
        assert algo_result['final_capital'] > 0

    def test_n_winners_lte_n_trades(self, algo_result):
        assert algo_result['n_winners'] <= algo_result['n_trades']

    def test_win_ratio_in_range(self, algo_result):
        assert 0.0 <= algo_result['win_ratio'] <= 1.0

    def test_mdd_non_negative(self, algo_result):
        assert algo_result['max_drawdown'] >= 0.0

    def test_trade_log_required_fields(self, algo_result):
        for trade in algo_result['trade_log']:
            for field in ('entry_bar', 'entry_price', 'exit_bar', 'exit_price',
                          'exit_rule', 'pnl_pct', 'capital'):
                assert field in trade

    def test_exit_rules_are_known(self, algo_result):
        valid = {'MTDC_WeightedDCE', 'ConsensusExit_Sell', 'ConsensusExit_Buy', 'RegimeExit_S2', 'EndOfPeriod'}
        for trade in algo_result['trade_log']:
            assert trade['exit_rule'] in valid

    def test_mismatched_model_theta_raises(self, prices, hmm_and_s1, top_pairs, ga_weights):
        hmm, s1 = hmm_and_s1
        models = [m for _, m in top_pairs]
        thetas = [t for t, _ in top_pairs]
        with pytest.raises(ValueError):
            run_mtdc_algorithm(prices.iloc[600:], hmm, s1, models, thetas[:-1], ga_weights)

    def test_invalid_consensus_mode_raises(self, prices, hmm_and_s1, top_pairs, ga_weights):
        hmm, s1 = hmm_and_s1
        models = [m for _, m in top_pairs]
        thetas = [t for t, _ in top_pairs]
        with pytest.raises(ValueError, match='consensus_mode'):
            run_mtdc_algorithm(prices.iloc[600:], hmm, s1, models, thetas, ga_weights,
                               consensus_mode='invalid')

    def test_majority_consensus_mode_runs(self, prices, hmm_and_s1, top_pairs, ga_weights):
        hmm, s1 = hmm_and_s1
        models = [m for _, m in top_pairs]
        thetas = [t for t, _ in top_pairs]
        result = run_mtdc_algorithm(prices.iloc[600:], hmm, s1, models, thetas, ga_weights,
                                    consensus_mode='majority', majority_weight=0.5)
        assert 'final_capital' in result


# ---------------------------------------------------------------------------
# MTDCPipeline tests
# ---------------------------------------------------------------------------

class TestMTDCPipeline:
    def test_fit_returns_self(self, prices):
        pipeline = MTDCPipeline(k=2, theta_pool=_THETA_POOL,
                                 gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
                                 ga_n_generations=_GA_GENS, ga_population_size=_GA_POP)
        result = pipeline.fit(prices.iloc[:600])
        assert result is pipeline

    def test_stdc_models_fitted_after_fit(self, prices):
        pipeline = MTDCPipeline(k=2, theta_pool=_THETA_POOL,
                                 gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
                                 ga_n_generations=_GA_GENS, ga_population_size=_GA_POP)
        pipeline.fit(prices.iloc[:600])
        assert pipeline.stdc_models_ is not None
        assert len(pipeline.stdc_models_) == 2

    def test_weights_fitted_after_fit(self, prices):
        pipeline = MTDCPipeline(k=2, theta_pool=_THETA_POOL,
                                 gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
                                 ga_n_generations=_GA_GENS, ga_population_size=_GA_POP)
        pipeline.fit(prices.iloc[:600])
        assert pipeline.weights_ is not None
        assert len(pipeline.weights_) == 2

    def test_run_returns_required_keys(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        pipeline = MTDCPipeline(k=2, theta_pool=_THETA_POOL,
                                 gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
                                 ga_n_generations=_GA_GENS, ga_population_size=_GA_POP)
        pipeline.fit(prices.iloc[:600])
        result = pipeline.run(prices.iloc[600:], hmm, s1)
        for key in ('final_capital', 'cumulative_return', 'max_drawdown', 'n_trades'):
            assert key in result

    def test_run_without_fit_raises(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        pipeline = MTDCPipeline(k=2, theta_pool=_THETA_POOL)
        with pytest.raises(RuntimeError, match='fit()'):
            pipeline.run(prices.iloc[600:], hmm, s1)

    def test_cannot_specify_both_pool_and_explicit(self):
        with pytest.raises(ValueError):
            MTDCPipeline(theta_pool=[0.01, 0.02], thetas=[0.01, 0.02])

    def test_explicit_thetas(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        pipeline = MTDCPipeline(thetas=[0.01, 0.015],
                                 gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
                                 ga_n_generations=_GA_GENS, ga_population_size=_GA_POP)
        pipeline.fit(prices.iloc[:600])
        assert pipeline.thetas_ == [0.01, 0.015]
        result = pipeline.run(prices.iloc[600:], hmm, s1)
        assert result['final_capital'] > 0


# ---------------------------------------------------------------------------
# search_optimal_k tests
# ---------------------------------------------------------------------------

class TestSearchOptimalK:
    def test_returns_dataframe(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        df = search_optimal_k(
            prices.iloc[:600], prices.iloc[600:], hmm, s1,
            k_max=2, theta_pool=_THETA_POOL,
            gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
            ga_n_generations=_GA_GENS, ga_population_size=_GA_POP,
        )
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_k(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        df = search_optimal_k(
            prices.iloc[:600], prices.iloc[600:], hmm, s1,
            k_max=2, theta_pool=_THETA_POOL,
            gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
            ga_n_generations=_GA_GENS, ga_population_size=_GA_POP,
        )
        assert len(df) <= 2
        if len(df) > 0:
            assert set(df['k'].tolist()).issubset({1, 2})

    def test_required_columns_present(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        df = search_optimal_k(
            prices.iloc[:600], prices.iloc[600:], hmm, s1,
            k_max=2, theta_pool=_THETA_POOL,
            gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
            ga_n_generations=_GA_GENS, ga_population_size=_GA_POP,
        )
        if len(df) > 0:
            for col in ('k', 'thetas', 'cumulative_return', 'max_drawdown',
                        'n_trades', 'win_ratio', 'sharpe'):
                assert col in df.columns

    def test_thetas_column_grows_with_k(self, prices, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        df = search_optimal_k(
            prices.iloc[:600], prices.iloc[600:], hmm, s1,
            k_max=2, theta_pool=_THETA_POOL,
            gp_n_generations=_GP_GENS, gp_population_size=_GP_POP,
            ga_n_generations=_GA_GENS, ga_population_size=_GA_POP,
        )
        if len(df) == 2:
            assert len(df.iloc[0]['thetas']) < len(df.iloc[1]['thetas'])
