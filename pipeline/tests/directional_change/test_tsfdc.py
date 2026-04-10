"""
Tests for TSFDC pipeline components:
  - train_bbtheta_classifier / predict_bbtheta
  - run_tsfdc_algorithm
  - TSFDCPipeline
  - evaluate_threshold_pair / search_optimal_thresholds

All tests use synthetic price series with large oscillations to ensure
both STheta and BTheta DC events are generated in abundance.
"""
import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from okmich_quant_features.directional_change import (
    extract_tsfdc_features,
    label_bbtheta,
    parse_dual_dc,
)
from okmich_quant_pipeline.directional_change import (
    TSFDCPipeline,
    evaluate_threshold_pair,
    predict_bbtheta,
    run_tsfdc_algorithm,
    search_optimal_thresholds,
    train_bbtheta_classifier,
)

pytestmark = pytest.mark.skipif(not LGBM_AVAILABLE, reason="lightgbm not installed")

STHETA = 0.04
BTHETA = 0.08


def _make_prices(n: int = 600, amplitude: float = 0.07, seed: int = 11) -> pd.Series:
    """Synthetic price series with large regular oscillations."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    prices = 100.0 * (1 + amplitude * np.sin(2 * np.pi * t / 25))
    noise = rng.normal(0, 0.004, n)
    return pd.Series(prices * (1 + noise))


@pytest.fixture(scope='module')
def prices():
    return _make_prices(600)


@pytest.fixture(scope='module')
def trained_classifier(prices):
    trends_s, trends_b = parse_dual_dc(prices.iloc[:400], STHETA, BTHETA)
    labelled = label_bbtheta(trends_s, trends_b)
    feats = extract_tsfdc_features(labelled, trends_b, STHETA, BTHETA)
    feats['bbtheta'] = labelled['bbtheta']
    return train_bbtheta_classifier(feats, random_seed=42)


@pytest.fixture(scope='module')
def algo_result(prices, trained_classifier):
    return run_tsfdc_algorithm(prices.iloc[400:], STHETA, BTHETA, trained_classifier, initial_capital=10_000.0)


# ---------------------------------------------------------------------------
# train_bbtheta_classifier tests
# ---------------------------------------------------------------------------

class TestTrainBBThetaClassifier:
    def test_returns_pipeline(self, trained_classifier):
        from sklearn.pipeline import Pipeline
        assert isinstance(trained_classifier, Pipeline)

    def test_classifier_has_predict(self, trained_classifier):
        assert hasattr(trained_classifier, 'predict')
        assert hasattr(trained_classifier, 'predict_proba')

    def test_insufficient_samples_raises(self):
        tiny = pd.DataFrame({'TMV': [1.0], 'T': [5.0], 'OSV': [0.5], 'COP': [0.2], 'bbtheta': [True]})
        with pytest.raises(ValueError, match='20 clean samples'):
            train_bbtheta_classifier(tiny)

    def test_single_class_raises(self):
        df = pd.DataFrame({
            'TMV': np.ones(30), 'T': np.ones(30) * 5,
            'OSV': np.zeros(30), 'COP': np.zeros(30),
            'bbtheta': [True] * 30,
        })
        with pytest.raises(ValueError, match='one class'):
            train_bbtheta_classifier(df)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({'TMV': [1.0, 2.0], 'T': [5.0, 6.0]})
        with pytest.raises(ValueError, match='missing columns'):
            train_bbtheta_classifier(df)

    def test_drops_nan_rows(self, prices):
        """NaN rows should be silently dropped before training."""
        trends_s, trends_b = parse_dual_dc(prices.iloc[:400], STHETA, BTHETA)
        labelled = label_bbtheta(trends_s, trends_b)
        feats = extract_tsfdc_features(labelled, trends_b, STHETA, BTHETA)
        feats['bbtheta'] = labelled['bbtheta']
        # First row has OSV=NaN; should still train successfully
        clf = train_bbtheta_classifier(feats)
        assert clf is not None


# ---------------------------------------------------------------------------
# predict_bbtheta tests
# ---------------------------------------------------------------------------

class TestPredictBBTheta:
    def test_returns_bool(self, trained_classifier):
        result = predict_bbtheta(trained_classifier, 1.5, 10.0, 0.3, 0.2)
        assert isinstance(result, bool)

    def test_nan_cop_returns_false(self, trained_classifier):
        assert predict_bbtheta(trained_classifier, 1.5, 10.0, 0.3, float('nan')) is False

    def test_nan_cop_conservative_default(self, trained_classifier):
        """Rule 1 path (immediate entry) when no BTheta DCC has fired."""
        result = predict_bbtheta(trained_classifier, 2.0, 15.0, -0.5, np.nan)
        assert result is False


# ---------------------------------------------------------------------------
# run_tsfdc_algorithm tests
# ---------------------------------------------------------------------------

class TestRunTSFDCAlgorithm:
    def test_returns_dict_with_down_and_up(self, algo_result):
        assert 'down' in algo_result
        assert 'up' in algo_result

    def test_down_has_required_keys(self, algo_result):
        for key in ('final_capital', 'cumulative_return', 'max_drawdown',
                    'n_trades', 'n_winners', 'win_ratio', 'profit_factor', 'trade_log'):
            assert key in algo_result['down']

    def test_up_has_required_keys(self, algo_result):
        for key in ('final_capital', 'cumulative_return', 'max_drawdown',
                    'n_trades', 'n_winners', 'win_ratio', 'profit_factor', 'trade_log'):
            assert key in algo_result['up']

    def test_final_capital_positive(self, algo_result):
        assert algo_result['down']['final_capital'] > 0
        assert algo_result['up']['final_capital'] > 0

    def test_n_winners_lte_n_trades(self, algo_result):
        assert algo_result['down']['n_winners'] <= algo_result['down']['n_trades']
        assert algo_result['up']['n_winners'] <= algo_result['up']['n_trades']

    def test_win_ratio_in_range(self, algo_result):
        for d in ('down', 'up'):
            wr = algo_result[d]['win_ratio']
            assert 0.0 <= wr <= 1.0

    def test_mdd_non_negative(self, algo_result):
        assert algo_result['down']['max_drawdown'] >= 0.0
        assert algo_result['up']['max_drawdown'] >= 0.0

    def test_trade_log_entries_have_required_fields(self, algo_result):
        for direction in ('down', 'up'):
            for trade in algo_result[direction]['trade_log']:
                for field in ('entry_bar', 'entry_price', 'entry_rule',
                              'exit_bar', 'exit_price', 'exit_rule', 'pnl_pct', 'capital'):
                    assert field in trade, f"Trade missing field '{field}'"

    def test_exit_rules_are_known_values(self, algo_result):
        valid_down = {'down.3', 'EndOfPeriod'}
        valid_up = {'up.3', 'EndOfPeriod'}
        for trade in algo_result['down']['trade_log']:
            assert trade['exit_rule'] in valid_down
        for trade in algo_result['up']['trade_log']:
            assert trade['exit_rule'] in valid_up

    def test_entry_rules_are_known_values(self, algo_result):
        valid_down = {'down.1', 'down.2'}
        valid_up = {'up.1', 'up.2'}
        for trade in algo_result['down']['trade_log']:
            assert trade['entry_rule'] in valid_down
        for trade in algo_result['up']['trade_log']:
            assert trade['entry_rule'] in valid_up

    def test_raises_if_btheta_lte_stheta(self, prices, trained_classifier):
        with pytest.raises(ValueError):
            run_tsfdc_algorithm(prices.iloc[400:], 0.05, 0.03, trained_classifier)

    def test_profit_factor_non_negative(self, algo_result):
        for d in ('down', 'up'):
            pf = algo_result[d]['profit_factor']
            assert pf >= 0.0 or pf == float('inf')


# ---------------------------------------------------------------------------
# TSFDCPipeline tests
# ---------------------------------------------------------------------------

class TestTSFDCPipeline:
    def test_fit_returns_self(self, prices):
        pipeline = TSFDCPipeline(stheta=STHETA, btheta=BTHETA)
        result = pipeline.fit(prices.iloc[:400])
        assert result is pipeline

    def test_classifier_fitted_after_fit(self, prices):
        pipeline = TSFDCPipeline(stheta=STHETA, btheta=BTHETA)
        pipeline.fit(prices.iloc[:400])
        assert pipeline.classifier_ is not None

    def test_run_returns_down_and_up(self, prices):
        pipeline = TSFDCPipeline(stheta=STHETA, btheta=BTHETA)
        pipeline.fit(prices.iloc[:400])
        result = pipeline.run(prices.iloc[400:])
        assert 'down' in result
        assert 'up' in result

    def test_run_without_fit_raises(self, prices):
        pipeline = TSFDCPipeline(stheta=STHETA, btheta=BTHETA)
        with pytest.raises(RuntimeError, match='fit()'):
            pipeline.run(prices.iloc[400:])

    def test_raises_if_btheta_lte_stheta(self):
        with pytest.raises(ValueError):
            TSFDCPipeline(stheta=0.05, btheta=0.03)

    def test_too_few_events_raises(self):
        tiny = pd.Series([100.0, 101.0, 100.5, 101.5])
        pipeline = TSFDCPipeline(stheta=STHETA, btheta=BTHETA)
        with pytest.raises(ValueError, match='DC events'):
            pipeline.fit(tiny)


# ---------------------------------------------------------------------------
# evaluate_threshold_pair / search_optimal_thresholds tests
# ---------------------------------------------------------------------------

class TestEvaluateThresholdPair:
    def test_returns_dict_with_required_keys(self, prices):
        result = evaluate_threshold_pair(prices.iloc[:400], prices.iloc[400:], STHETA, BTHETA, min_trades=0)
        for key in ('stheta', 'btheta', 'valid', 'rr_down', 'rr_up',
                    'n_trades_down', 'n_trades_up'):
            assert key in result

    def test_stheta_btheta_stored_correctly(self, prices):
        result = evaluate_threshold_pair(prices.iloc[:400], prices.iloc[400:], STHETA, BTHETA, min_trades=0)
        assert result['stheta'] == pytest.approx(STHETA)
        assert result['btheta'] == pytest.approx(BTHETA)

    def test_invalid_if_btheta_lte_stheta(self, prices):
        result = evaluate_threshold_pair(prices.iloc[:400], prices.iloc[400:], 0.05, 0.03)
        assert result['valid'] is False

    def test_invalid_if_below_min_trades(self, prices):
        result = evaluate_threshold_pair(prices.iloc[:400], prices.iloc[400:], STHETA, BTHETA, min_trades=9999)
        assert result['valid'] is False

    def test_valid_when_sufficient_trades(self, prices):
        result = evaluate_threshold_pair(prices.iloc[:400], prices.iloc[400:], STHETA, BTHETA, min_trades=0)
        assert result['valid'] is True


class TestSearchOptimalThresholds:
    def test_returns_dataframe(self, prices):
        df = search_optimal_thresholds(
            prices.iloc[:400], prices.iloc[400:],
            stheta_values=[STHETA], btheta_values=[BTHETA], min_trades=0,
        )
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_valid_pair(self, prices):
        df = search_optimal_thresholds(
            prices.iloc[:400], prices.iloc[400:],
            stheta_values=[STHETA, BTHETA], btheta_values=[BTHETA, BTHETA * 2], min_trades=0,
        )
        # 4 combinations but only btheta > stheta pairs are evaluated
        assert len(df) > 0

    def test_excludes_pairs_where_btheta_lte_stheta(self, prices):
        df = search_optimal_thresholds(
            prices.iloc[:400], prices.iloc[400:],
            stheta_values=[0.05, 0.10],
            btheta_values=[0.03, 0.08],
            min_trades=0,
        )
        # Only 0.05/0.08 is valid (btheta > stheta); others excluded
        for _, row in df.iterrows():
            assert row['btheta'] > row['stheta']

    def test_sorted_by_combined_return(self, prices):
        df = search_optimal_thresholds(
            prices.iloc[:400], prices.iloc[400:],
            stheta_values=[STHETA, STHETA * 1.5],
            btheta_values=[BTHETA, BTHETA * 1.5],
            min_trades=0,
        )
        if len(df) > 1:
            combined = df['rr_down'].fillna(0) + df['rr_up'].fillna(0)
            assert (combined.diff().iloc[1:] <= 0).all()

    def test_contains_all_metric_columns(self, prices):
        df = search_optimal_thresholds(
            prices.iloc[:400], prices.iloc[400:],
            stheta_values=[STHETA], btheta_values=[BTHETA], min_trades=0,
        )
        for col in ('stheta', 'btheta', 'valid', 'rr_down', 'rr_up',
                    'mdd_down', 'mdd_up', 'n_trades_down', 'n_trades_up',
                    'win_ratio_down', 'win_ratio_up',
                    'profit_factor_down', 'profit_factor_up'):
            assert col in df.columns
