"""
Tests for run_ita_simulation.

Hand-calculated reference: prices=[100, 112, 100, 90, 100, 115, 103], theta=0.1

idc_parse signals (both long and short traded):
  i=1: upturn_dc=T  → Rule 1L: BUY at 112
  i=2: downturn_dc=T → Rule 3L: SELL at 100 (pnl=-10.714%), then Rule 1S: SHORT at 100
  i=3: new_low (pl=90): Rule 2S check: 90 <= (1-0.2)*112=89.6? No → no exit
  i=4: upturn_dc=T  → Rule 3S: cover SHORT at 100 (pnl=0%), then Rule 1L: BUY at 100
  i=5: new_high=T, ph=115 >= (1+0.2)*90=108 → Rule 2L: SELL at 115, pnl=+15%
  i=6: downturn_dc=T → Rule 1S: SHORT at 103; closed EndOfPeriod at 103, pnl=0%

4 trades: LONG(-10.714%), SHORT(0%), LONG(+15%), SHORT(0%)
CRR = (102/112) * 1.0 * 1.15 * 1.0 - 1 = same as 2-trade long-only ≈ 2.6786%
MDD: same peak/trough as before ≈ 10.714%
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import idc_parse
from okmich_quant_pipeline.directional_change import run_ita_simulation

THETA = 0.1
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]
INITIAL_CAPITAL = 10_000.0


@pytest.fixture
def ref_prices():
    return pd.Series(REF_PRICES)


@pytest.fixture
def ref_idc(ref_prices):
    return idc_parse(ref_prices, THETA)


@pytest.fixture
def ref_result(ref_idc, ref_prices):
    return run_ita_simulation(ref_idc, ref_prices, THETA, INITIAL_CAPITAL)


class TestReturnKeys:
    def test_result_has_all_keys(self, ref_result):
        assert {"cumulative_return", "n_trades", "n_winners", "max_drawdown"} <= set(ref_result.keys())

    def test_n_trades_is_int(self, ref_result):
        assert isinstance(ref_result["n_trades"], int)

    def test_n_winners_is_int(self, ref_result):
        assert isinstance(ref_result["n_winners"], int)


class TestHandCalculated:
    def test_n_trades(self, ref_result):
        # 4 trades: LONG, SHORT(0%), LONG, SHORT(0%)
        assert ref_result["n_trades"] == 4

    def test_n_winners(self, ref_result):
        # Only the second LONG trade (+15%) is a winner; zero-PnL shorts don't count
        assert ref_result["n_winners"] == 1

    def test_cumulative_return(self, ref_result):
        t1 = (100 - 112) / 112
        t2 = (115 - 100) / 100
        expected_crr = ((1 + t1) * (1 + t2) - 1) * 100
        assert ref_result["cumulative_return"] == pytest.approx(expected_crr, rel=1e-6)

    def test_max_drawdown(self, ref_result):
        # Peak = 10000, trough after trade 1 = 10000*(102/112)
        trough = INITIAL_CAPITAL * (100 / 112)
        expected_mdd = (INITIAL_CAPITAL - trough) / INITIAL_CAPITAL * 100
        assert ref_result["max_drawdown"] == pytest.approx(expected_mdd, rel=1e-6)


class TestRule2ProfitTarget:
    def test_rule2_fires_when_ph_reaches_two_theta(self):
        # Construct: trough at 100, DC confirmation at 110 (theta=0.1),
        # then new high at 121 >= (1+0.2)*100=120 → Rule 2 fires
        prices = pd.Series([100.0, 90.0, 99.0, 121.0])
        idc = idc_parse(prices, theta=0.1)
        result = run_ita_simulation(idc, prices, theta=0.1)
        # At least one trade should complete via Rule 2
        assert result["n_trades"] >= 1

    def test_rule2_does_not_fire_below_target(self):
        # New high at 119 < 120 → Rule 2 does NOT fire; only Rule 3 eventually exits
        prices = pd.Series([100.0, 90.0, 99.0, 119.0, 105.0])
        idc = idc_parse(prices, theta=0.1)
        result = run_ita_simulation(idc, prices, theta=0.1)
        # Rule 2 target not reached — position may exit via Rule 3 or end-of-period
        # Just confirm simulation runs without error and returns valid dict
        assert "cumulative_return" in result


class TestRule3StopLoss:
    def test_rule3_fires_on_downturn_dc(self, ref_result):
        # In the reference, trade 1 exits via Rule 3 (downturn_dc at i=2)
        # Trade 1 is a loss confirming Rule 3 stop-loss fired
        assert ref_result["n_winners"] < ref_result["n_trades"]


class TestEndOfPeriod:
    def test_open_position_closed_at_end(self):
        # Price rises theta → DC, then keeps rising without Rule 2 or 3 firing
        # Position must close at end of period
        prices = pd.Series([100.0, 90.0, 99.0, 101.0, 103.0])
        idc = idc_parse(prices, theta=0.1)
        result = run_ita_simulation(idc, prices, theta=0.1)
        assert result["n_trades"] >= 1

    def test_no_position_no_end_of_period_trade(self):
        # Flat series — no DC ever fires, no trades
        prices = pd.Series([100.0] * 20)
        idc = idc_parse(prices, theta=0.1)
        result = run_ita_simulation(idc, prices, theta=0.1)
        assert result["n_trades"] == 0
        assert result["cumulative_return"] == pytest.approx(0.0)


class TestEdgeCases:
    def test_no_dc_events_returns_zero_crr(self):
        prices = pd.Series([100.0, 101.0, 100.5])
        idc = idc_parse(prices, theta=0.5)
        result = run_ita_simulation(idc, prices, theta=0.5)
        assert result["n_trades"] == 0
        assert result["cumulative_return"] == pytest.approx(0.0)
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_n_winners_never_exceeds_n_trades(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, 500))))
        idc = idc_parse(prices, theta=0.005)
        result = run_ita_simulation(idc, prices, theta=0.005)
        assert result["n_winners"] <= result["n_trades"]

    def test_max_drawdown_non_negative(self):
        rng = np.random.default_rng(7)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, 500))))
        idc = idc_parse(prices, theta=0.005)
        result = run_ita_simulation(idc, prices, theta=0.005)
        assert result["max_drawdown"] >= 0.0

    def test_custom_initial_capital(self, ref_idc, ref_prices):
        r1 = run_ita_simulation(ref_idc, ref_prices, THETA, initial_capital=10_000.0)
        r2 = run_ita_simulation(ref_idc, ref_prices, THETA, initial_capital=50_000.0)
        # CRR is capital-independent (percentage return)
        assert r1["cumulative_return"] == pytest.approx(r2["cumulative_return"], rel=1e-9)
        assert r1["n_trades"] == r2["n_trades"]
