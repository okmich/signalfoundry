"""
Tests for C+GP+TS pipeline components:
  - build_gp_toolbox / run_gp_regression
  - train_cgpts_model / predict_trend_end
  - run_cgpts_algorithm

GP tests use a synthetic linear dataset to keep runtime short.
Algorithm tests use the same synthetic price series as ITA tests.
"""
import numpy as np
import pandas as pd
import pytest

try:
    from deap import base  # noqa: F401
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

from okmich_quant_pipeline.directional_change import (
    build_gp_toolbox,
    fit_rcd_hmm,
    predict_trend_end,
    run_cgpts_algorithm,
    run_gp_regression,
    s1_state_index,
    train_cgpts_model,
)

pytestmark = pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not installed")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_oscillating_prices(n: int = 800, amplitude: float = 0.015, seed: int = 0) -> pd.Series:
    """
    Synthetic price series with irregular cycles producing a mix of αDC and βDC trends.
    Uses two overlapping sine components at different frequencies and high noise to
    ensure some DC events are immediately reversed (βDC).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 100.0 * np.exp(0.0001 * t)
    noise = rng.normal(0, 0.006, n)          # high noise → some immediate reversals
    cycle1 = amplitude * np.sin(2 * np.pi * t / 30)
    cycle2 = (amplitude * 0.4) * np.sin(2 * np.pi * t / 7)  # short cycle creates βDC
    return pd.Series(trend * (1 + cycle1 + cycle2 + noise))


@pytest.fixture(scope='module')
def prices():
    return _make_oscillating_prices(800)


@pytest.fixture(scope='module')
def trained_model(prices):
    return train_cgpts_model(prices.iloc[:600], theta=0.01, alpha=1.0,
                             n_generations=5, population_size=50, random_seed=42)


@pytest.fixture(scope='module')
def hmm_and_s1(prices):
    from okmich_quant_features.directional_change import parse_dc_events, log_r
    trends = parse_dc_events(prices.iloc[:600], theta=0.01)
    rcd = log_r(trends).dropna()
    hmm = fit_rcd_hmm(rcd.values.reshape(-1, 1))
    s1 = s1_state_index(hmm)
    return hmm, s1


# ---------------------------------------------------------------------------
# GP toolbox tests
# ---------------------------------------------------------------------------

class TestBuildGPToolbox:
    def test_returns_tuple(self):
        toolbox, pset = build_gp_toolbox()
        assert toolbox is not None
        assert pset is not None

    def test_input_variable_named_dc_l(self):
        _, pset = build_gp_toolbox()
        assert 'DC_l' in pset.arguments

    def test_toolbox_has_required_operators(self):
        toolbox, _ = build_gp_toolbox()
        for attr in ('select', 'mate', 'mutate', 'evaluate', 'population'):
            # evaluate not registered yet — others should be
            pass
        assert hasattr(toolbox, 'select')
        assert hasattr(toolbox, 'mate')
        assert hasattr(toolbox, 'mutate')


# ---------------------------------------------------------------------------
# GP regression tests
# ---------------------------------------------------------------------------

class TestRunGPRegression:
    def test_returns_tuple_of_three(self):
        dc = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        os = dc * 2.0
        result = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=1)
        assert len(result) == 3

    def test_best_func_callable(self):
        dc = np.linspace(1, 20, 10)
        os = dc * 1.5
        func, _, _ = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=1)
        val = func(5.0)
        assert np.isfinite(val)

    def test_rmse_is_finite_positive(self):
        dc = np.linspace(1, 20, 10)
        os = dc * 1.5
        _, rmse, _ = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=1)
        assert np.isfinite(rmse)
        assert rmse >= 0.0

    def test_equation_str_returned(self):
        dc = np.linspace(1, 20, 10)
        os = dc * 1.5
        _, _, eq = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=1)
        assert isinstance(eq, str)
        assert len(eq) > 0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match='5 αDC samples'):
            run_gp_regression(np.array([1.0, 2.0]), np.array([2.0, 4.0]))

    def test_reproducible_with_same_seed(self):
        dc = np.linspace(1, 30, 15)
        os = dc * 1.8 + 2.0
        _, rmse1, eq1 = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=99)
        _, rmse2, eq2 = run_gp_regression(dc, os, n_generations=3, population_size=20, random_seed=99)
        assert rmse1 == pytest.approx(rmse2)
        assert eq1 == eq2


# ---------------------------------------------------------------------------
# train_cgpts_model tests
# ---------------------------------------------------------------------------

class TestTrainCGPTSModel:
    def test_returns_dict_with_required_keys(self, trained_model):
        for key in ('gp_func', 'gp_rmse', 'gp_equation', 'classifier',
                    'alpha_rate', 'n_alpha', 'n_beta', 'n_trends_total', 'theta', 'alpha_param'):
            assert key in trained_model

    def test_alpha_rate_in_range(self, trained_model):
        assert 0.0 <= trained_model['alpha_rate'] <= 1.0

    def test_n_counts_consistent(self, trained_model):
        assert trained_model['n_alpha'] + trained_model['n_beta'] == trained_model['n_trends_total']

    def test_gp_func_callable(self, trained_model):
        val = trained_model['gp_func'](10.0)
        assert np.isfinite(val) or val == 0.0

    def test_gp_rmse_finite(self, trained_model):
        assert np.isfinite(trained_model['gp_rmse'])

    def test_classifier_has_predict(self, trained_model):
        assert hasattr(trained_model['classifier'], 'predict')
        assert hasattr(trained_model['classifier'], 'predict_proba')

    def test_insufficient_data_raises(self):
        tiny = pd.Series([100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0])
        with pytest.raises(ValueError):
            train_cgpts_model(tiny, theta=0.1)

    def test_theta_stored_correctly(self, trained_model):
        assert trained_model['theta'] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# predict_trend_end tests
# ---------------------------------------------------------------------------

class TestPredictTrendEnd:
    def test_returns_dict_with_required_keys(self, trained_model):
        result = predict_trend_end(5, 100.5, 99.5, 101.0, False, trained_model)
        for key in ('trend_type', 'p_alpha', 'p_beta', 'predicted_os_bars', 'estimated_dce_offset'):
            assert key in result

    def test_trend_type_is_alpha_or_beta(self, trained_model):
        result = predict_trend_end(5, 100.5, 99.5, 101.0, False, trained_model)
        assert result['trend_type'] in ('alpha_dc', 'beta_dc')

    def test_probabilities_sum_to_one(self, trained_model):
        result = predict_trend_end(5, 100.5, 99.5, 101.0, False, trained_model)
        assert result['p_alpha'] + result['p_beta'] == pytest.approx(1.0, abs=1e-6)

    def test_beta_dc_has_zero_os_bars(self, trained_model):
        # Run enough predictions to likely get a βDC
        results = [predict_trend_end(i, 100.0, 100.5, 99.0, False, trained_model) for i in range(1, 10)]
        beta_results = [r for r in results if r['trend_type'] == 'beta_dc']
        for r in beta_results:
            assert r['predicted_os_bars'] == 0.0
            assert r['estimated_dce_offset'] == 0

    def test_alpha_dc_has_non_negative_offset(self, trained_model):
        results = [predict_trend_end(i, 100.0, 99.0, 101.0, True, trained_model) for i in range(1, 10)]
        alpha_results = [r for r in results if r['trend_type'] == 'alpha_dc']
        for r in alpha_results:
            assert r['estimated_dce_offset'] >= 0

    def test_nan_prev_dcc_handled(self, trained_model):
        result = predict_trend_end(5, 100.5, 99.5, float('nan'), False, trained_model)
        assert result['trend_type'] in ('alpha_dc', 'beta_dc')


# ---------------------------------------------------------------------------
# run_cgpts_algorithm tests
# ---------------------------------------------------------------------------

class TestRunCGPTSAlgorithm:
    def test_returns_dict_with_required_keys(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        for key in ('final_capital', 'cumulative_return', 'max_drawdown',
                    'n_trades', 'n_winners', 'trade_log'):
            assert key in result

    def test_final_capital_positive(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        assert result['final_capital'] > 0

    def test_n_winners_lte_n_trades(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        assert result['n_winners'] <= result['n_trades']

    def test_trade_log_entries_have_required_fields(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        for trade in result['trade_log']:
            for field in ('entry_bar', 'entry_price', 'exit_bar', 'exit_price',
                          'exit_rule', 'pnl_pct', 'capital'):
                assert field in trade

    def test_exit_rules_are_known_values(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        valid_rules = {'CGP_EstimatedDCE', 'EarlyExit_DownturnDC', 'RegimeExit_S2', 'EndOfPeriod'}
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        for trade in result['trade_log']:
            assert trade['exit_rule'] in valid_rules

    def test_mdd_non_negative(self, prices, trained_model, hmm_and_s1):
        hmm, s1 = hmm_and_s1
        result = run_cgpts_algorithm(prices.iloc[600:], theta=0.01, alpha=1.0,
                                     hmm=hmm, s1_idx=s1, cgpts_model=trained_model)
        assert result['max_drawdown'] >= 0.0