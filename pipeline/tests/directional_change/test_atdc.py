"""
Tests for ATDC: AdaptationMode, compute_metric, compute_adapted_theta,
run_atdc_algorithm, ATDCPipeline.

Fixtures use a short oscillating price series and minimised parameters for speed.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline.directional_change import (
    AdaptationMode,
    ATDCPipeline,
    compute_adapted_theta,
    compute_metric,
    run_atdc_algorithm,
)

# ── shared constants ──────────────────────────────────────────────────────────

_THETA_INIT = 0.02
_THETA_MIN = 0.005
_THETA_MAX = 0.10
_ALPHA = 1.0
_ADAPT_RATE = 0.5
_LOOKBACK = 50
_ADAPT_STEP = 20
_INITIAL_CAPITAL = 10_000.0


def _make_prices(n: int = 400, amplitude: float = 0.04, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = amplitude * np.sin(2 * np.pi * t / 40)
    noise = rng.normal(0, amplitude * 0.1, n)
    prices = 100.0 * np.exp(np.cumsum(signal + noise) * 0.1)
    idx = pd.date_range('2020-01-01', periods=n, freq='h')
    return pd.Series(prices, index=idx)


@pytest.fixture
def prices():
    return _make_prices()


@pytest.fixture
def prices_window(prices):
    return prices.iloc[:_LOOKBACK]


@pytest.fixture
def algo_result(prices):
    return run_atdc_algorithm(
        prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
        _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
        adaptation_mode='volatility', alpha=_ALPHA,
        hmm=None, initial_capital=_INITIAL_CAPITAL,
    )


# ── TestAdaptationMode ────────────────────────────────────────────────────────

class TestAdaptationMode:
    def test_volatility_is_valid(self):
        assert AdaptationMode('volatility') == AdaptationMode.VOLATILITY

    def test_rdc_is_valid(self):
        assert AdaptationMode('rdc') == AdaptationMode.RDC

    def test_tmv_is_valid(self):
        assert AdaptationMode('tmv') == AdaptationMode.TMV

    def test_custom_is_valid(self):
        assert AdaptationMode('custom') == AdaptationMode.CUSTOM

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AdaptationMode('unknown_mode')


# ── TestComputeMetric ─────────────────────────────────────────────────────────

class TestComputeMetric:
    def test_volatility_returns_float(self, prices_window):
        m = compute_metric(prices_window, _THETA_INIT, 'volatility')
        assert isinstance(m, float)

    def test_volatility_non_negative(self, prices_window):
        m = compute_metric(prices_window, _THETA_INIT, 'volatility')
        assert m >= 0.0

    def test_rdc_returns_float(self, prices_window):
        m = compute_metric(prices_window, _THETA_INIT, 'rdc', alpha=_ALPHA)
        assert isinstance(m, float)

    def test_rdc_non_negative(self, prices_window):
        m = compute_metric(prices_window, _THETA_INIT, 'rdc', alpha=_ALPHA)
        assert m >= 0.0

    def test_custom_calls_fn(self, prices_window):
        fn = lambda p, t: 42.0  # noqa: E731
        m = compute_metric(prices_window, _THETA_INIT, 'custom', custom_fn=fn)
        assert m == pytest.approx(42.0)

    def test_custom_without_fn_raises(self, prices_window):
        with pytest.raises(ValueError):
            compute_metric(prices_window, _THETA_INIT, 'custom', custom_fn=None)

    def test_short_series_returns_zero(self):
        tiny = pd.Series([100.0])
        m = compute_metric(tiny, _THETA_INIT, 'volatility')
        assert m == pytest.approx(0.0)


# ── TestComputeAdaptedTheta ───────────────────────────────────────────────────

class TestComputeAdaptedTheta:
    def test_returns_tuple(self, prices_window):
        result = compute_adapted_theta(
            prices_window, _THETA_INIT, 0.001,
            'volatility', _ADAPT_RATE, _THETA_MIN, _THETA_MAX,
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_theta_within_bounds(self, prices_window):
        theta_new, _ = compute_adapted_theta(
            prices_window, _THETA_INIT, 0.001,
            'volatility', _ADAPT_RATE, _THETA_MIN, _THETA_MAX,
        )
        assert _THETA_MIN <= theta_new <= _THETA_MAX

    def test_theta_within_bounds_rdc(self, prices_window):
        theta_new, _ = compute_adapted_theta(
            prices_window, _THETA_INIT, 0.001,
            'rdc', _ADAPT_RATE, _THETA_MIN, _THETA_MAX, alpha=_ALPHA,
        )
        assert _THETA_MIN <= theta_new <= _THETA_MAX

    def test_high_rate_still_bounded(self, prices_window):
        theta_new, _ = compute_adapted_theta(
            prices_window, _THETA_INIT, 0.0,
            'volatility', 100.0, _THETA_MIN, _THETA_MAX,
        )
        assert _THETA_MIN <= theta_new <= _THETA_MAX

    def test_metric_returned_is_float(self, prices_window):
        _, metric = compute_adapted_theta(
            prices_window, _THETA_INIT, 0.001,
            'volatility', _ADAPT_RATE, _THETA_MIN, _THETA_MAX,
        )
        assert isinstance(metric, float)


# ── TestRunATDCAlgorithm ──────────────────────────────────────────────────────

class TestRunATDCAlgorithm:
    def test_returns_dict(self, algo_result):
        assert isinstance(algo_result, dict)

    def test_required_keys(self, algo_result):
        required = {'final_capital', 'cumulative_return', 'max_drawdown',
                    'n_trades', 'n_winners', 'win_ratio', 'profit_factor',
                    'sharpe', 'theta_history', 'trade_log'}
        assert required <= set(algo_result.keys())

    def test_final_capital_positive(self, algo_result):
        assert algo_result['final_capital'] > 0.0

    def test_n_winners_lte_n_trades(self, algo_result):
        assert algo_result['n_winners'] <= algo_result['n_trades']

    def test_win_ratio_in_range(self, algo_result):
        assert 0.0 <= algo_result['win_ratio'] <= 1.0

    def test_mdd_non_negative(self, algo_result):
        assert algo_result['max_drawdown'] >= 0.0

    def test_theta_history_is_list(self, algo_result):
        assert isinstance(algo_result['theta_history'], list)

    def test_theta_history_first_entry(self, prices):
        result = run_atdc_algorithm(
            prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
            _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
            adaptation_mode='volatility', alpha=_ALPHA, hmm=None,
            initial_capital=_INITIAL_CAPITAL,
        )
        # First entry is always (bar=0, theta=theta_init)
        assert result['theta_history'][0] == (0, _THETA_INIT)

    def test_theta_history_values_within_bounds(self, algo_result):
        for _, theta in algo_result['theta_history']:
            assert _THETA_MIN <= theta <= _THETA_MAX

    def test_trade_log_is_list(self, algo_result):
        assert isinstance(algo_result['trade_log'], list)

    def test_trade_log_has_side_field(self, prices):
        result = run_atdc_algorithm(
            prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
            _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
            adaptation_mode='volatility', alpha=_ALPHA, hmm=None,
            initial_capital=_INITIAL_CAPITAL,
        )
        if result['trade_log']:
            assert 'side' in result['trade_log'][0]
            assert result['trade_log'][0]['side'] in ('long', 'short')

    def test_all_adaptation_modes_run(self, prices):
        for mode in ('volatility', 'rdc'):
            result = run_atdc_algorithm(
                prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
                _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
                adaptation_mode=mode, alpha=_ALPHA, hmm=None,
                initial_capital=_INITIAL_CAPITAL,
            )
            assert result['final_capital'] > 0.0

    def test_custom_mode_runs(self, prices):
        fn = lambda p, t: float(np.std(np.diff(np.log(p.values + 1e-10))))  # noqa: E731
        result = run_atdc_algorithm(
            prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
            _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
            adaptation_mode='custom', alpha=_ALPHA, hmm=None,
            initial_capital=_INITIAL_CAPITAL, custom_fn=fn,
        )
        assert result['final_capital'] > 0.0

    def test_no_hmm_runs(self, prices):
        result = run_atdc_algorithm(
            prices, _THETA_INIT, _THETA_MIN, _THETA_MAX,
            _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
            adaptation_mode='volatility', alpha=_ALPHA, hmm=None,
            initial_capital=_INITIAL_CAPITAL,
        )
        assert isinstance(result, dict)

    def test_flat_series_no_trades(self):
        flat = pd.Series([100.0] * 200)
        result = run_atdc_algorithm(
            flat, _THETA_INIT, _THETA_MIN, _THETA_MAX,
            _ADAPT_RATE, _LOOKBACK, _ADAPT_STEP,
            adaptation_mode='volatility', alpha=_ALPHA, hmm=None,
        )
        assert result['n_trades'] == 0

    def test_capital_consistent_with_crr(self, algo_result):
        expected = _INITIAL_CAPITAL * (1.0 + algo_result['cumulative_return'] / 100.0)
        assert algo_result['final_capital'] == pytest.approx(expected, rel=1e-9)


# ── TestATDCPipeline ──────────────────────────────────────────────────────────

class TestATDCPipeline:
    @pytest.fixture
    def prices_train_test(self):
        full = _make_prices(n=600)
        return full.iloc[:400], full.iloc[400:]

    @pytest.fixture
    def pipeline(self):
        return ATDCPipeline(
            adaptation_mode='volatility',
            adaptation_rate=_ADAPT_RATE,
            lookback_window=_LOOKBACK,
            adaptation_step=_ADAPT_STEP,
            theta_min=_THETA_MIN,
            theta_max=_THETA_MAX,
            use_hmm=False,
            use_gp=False,
            n_calls=5,
            n_initial=3,
            random_state=42,
        )

    def test_fit_returns_self(self, pipeline, prices_train_test):
        train, _ = prices_train_test
        result = pipeline.fit(train)
        assert result is pipeline

    def test_theta_init_fitted(self, pipeline, prices_train_test):
        train, _ = prices_train_test
        pipeline.fit(train)
        assert pipeline.theta_init_ is not None
        assert _THETA_MIN <= pipeline.theta_init_ <= _THETA_MAX

    def test_alpha_fitted(self, pipeline, prices_train_test):
        train, _ = prices_train_test
        pipeline.fit(train)
        assert pipeline.alpha_ is not None

    def test_run_returns_required_keys(self, pipeline, prices_train_test):
        train, test = prices_train_test
        pipeline.fit(train)
        result = pipeline.run(test, initial_capital=_INITIAL_CAPITAL)
        assert 'final_capital' in result
        assert 'optimal_theta_init' in result
        assert 'optimal_alpha' in result

    def test_run_without_fit_raises(self, pipeline, prices_train_test):
        _, test = prices_train_test
        with pytest.raises(RuntimeError):
            pipeline.run(test)

    def test_fixed_alpha(self, prices_train_test):
        train, test = prices_train_test
        pipeline = ATDCPipeline(
            adaptation_mode='volatility',
            alpha=0.8,
            use_hmm=False,
            n_calls=5, n_initial=3,
        )
        pipeline.fit(train)
        assert pipeline.alpha_ == pytest.approx(0.8)

    def test_no_hmm_pipeline(self, prices_train_test):
        train, test = prices_train_test
        pipeline = ATDCPipeline(use_hmm=False, n_calls=5, n_initial=3)
        pipeline.fit(train)
        assert pipeline.hmm_ is None
        result = pipeline.run(test)
        assert result['final_capital'] > 0.0
