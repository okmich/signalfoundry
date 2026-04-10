"""
Tests for optimise_idc_params.

Full optimisation (n_calls=100) is not run in the test suite — it is too slow
for CI and the quality of the result depends on data length and randomness.
Tests verify:
  - Return type and value ranges
  - Bounds are respected
  - Preset constants have correct values
  - Degenerate price series (no DC events) does not crash
  - n_calls / random_state are honoured
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline.directional_change import (
    ALPHA_BOUNDS,
    THETA_BOUNDS_5MIN,
    THETA_BOUNDS_TICK,
    optimise_idc_params,
)


@pytest.fixture
def random_prices():
    rng = np.random.default_rng(0)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 1000))))


class TestPresetConstants:
    def test_theta_bounds_tick(self):
        assert THETA_BOUNDS_TICK == (0.0003, 0.003)

    def test_theta_bounds_5min(self):
        assert THETA_BOUNDS_5MIN == (0.001, 0.005)

    def test_alpha_bounds(self):
        assert ALPHA_BOUNDS == (0.10, 1.00)

    def test_tick_lower_lt_upper(self):
        assert THETA_BOUNDS_TICK[0] < THETA_BOUNDS_TICK[1]

    def test_5min_lower_lt_upper(self):
        assert THETA_BOUNDS_5MIN[0] < THETA_BOUNDS_5MIN[1]

    def test_alpha_lower_lt_upper(self):
        assert ALPHA_BOUNDS[0] < ALPHA_BOUNDS[1]


class TestReturnContract:
    def test_returns_tuple_of_two_floats(self, random_prices):
        result = optimise_idc_params(random_prices, n_calls=5, n_initial=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_theta_within_bounds(self, random_prices):
        lo, hi = THETA_BOUNDS_TICK
        theta, _ = optimise_idc_params(random_prices, theta_bounds=(lo, hi), n_calls=5, n_initial=3)
        assert lo <= theta <= hi

    def test_alpha_within_bounds(self, random_prices):
        lo, hi = ALPHA_BOUNDS
        _, alpha = optimise_idc_params(random_prices, alpha_bounds=(lo, hi), n_calls=5, n_initial=3)
        assert lo <= alpha <= hi

    def test_custom_bounds_respected(self, random_prices):
        theta_b = (0.001, 0.002)
        alpha_b = (0.5, 0.9)
        theta, alpha = optimise_idc_params(random_prices, theta_bounds=theta_b, alpha_bounds=alpha_b, n_calls=5, n_initial=3)
        assert theta_b[0] <= theta <= theta_b[1]
        assert alpha_b[0] <= alpha <= alpha_b[1]


class TestReproducibility:
    def test_same_random_state_returns_same_result(self, random_prices):
        r1 = optimise_idc_params(random_prices, n_calls=5, n_initial=3, random_state=7)
        r2 = optimise_idc_params(random_prices, n_calls=5, n_initial=3, random_state=7)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])

    def test_different_random_state_may_differ(self, random_prices):
        r1 = optimise_idc_params(random_prices, n_calls=5, n_initial=3, random_state=1)
        r2 = optimise_idc_params(random_prices, n_calls=5, n_initial=3, random_state=99)
        # Results may coincidentally match, but typically differ
        # Just assert both are valid — no crash
        assert isinstance(r1, tuple) and isinstance(r2, tuple)


class TestDegenerateCases:
    def test_flat_series_does_not_crash(self):
        # Flat series produces no DC events — objective always returns 0.0
        prices = pd.Series([100.0] * 200)
        theta, alpha = optimise_idc_params(prices, n_calls=5, n_initial=3)
        assert THETA_BOUNDS_TICK[0] <= theta <= THETA_BOUNDS_TICK[1]
        assert ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]

    def test_5min_bounds_preset(self, random_prices):
        theta, alpha = optimise_idc_params(random_prices, theta_bounds=THETA_BOUNDS_5MIN, n_calls=5, n_initial=3)
        assert THETA_BOUNDS_5MIN[0] <= theta <= THETA_BOUNDS_5MIN[1]
        assert ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]
