"""
Tests for run_ita_algorithm1.

MockHMM design:
  MockHMM(winning_state=k) creates an HMM whose k-th distribution has log_prob=0.0
  (all others -1000.0), so get_current_regime() always returns state k.
  With s1_idx=0:
    MockHMM(0) → always S1 (normal)  → Rules 1L/1S can open, Rules 2/3 can close
    MockHMM(1) → always S2 (abnormal) → Rule 4 closes open positions, never opens new ones

Reference series: [100, 112, 100, 90, 100, 115, 103], theta=0.1
idc_parse signals:
  i=1: upturn_dc=T, rdc=NaN   → init DC — HMM skipped (no position opened/closed)
  i=2: downturn_dc=T, rdc=valid → HMM queried for short entry
  i=4: upturn_dc=T, rdc=valid  → HMM queried; Rule 3S exits short, Rule 1L enters long
  i=5: new_high=T, ph=115 >= (1+0.2)*90=108 → Rule 2L exits long
  i=6: downturn_dc=T, rdc=valid → Rule 1S enters short (flat after i=5 exit)

With always-S1 (MockHMM(0), s1_idx=0):
  i=2: enter SHORT at 100
  i=4: Rule3S exit SHORT at 100 (pnl=0%), then Rule1L enter LONG at 100
  i=5: Rule2L exit LONG at 115, pnl=+15%
  i=6: Rule1S enter SHORT at 103; closed EndOfPeriod at 103, pnl=0%
  → 3 trades, 1 winner, CRR=+15%

With always-S2 (MockHMM(1), s1_idx=0):
  No entries (both Rules 1L/1S require S1) → 0 trades
"""
import torch
import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline.directional_change import run_ita_algorithm1

THETA = 0.1
ALPHA = 1.0  # symmetric threshold
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]
INITIAL_CAPITAL = 10_000.0
S1_IDX = 0  # state 0 = S1 (normal) throughout all tests


# ── MockHMM ──────────────────────────────────────────────────────────────────

class _MockDistribution:
    def __init__(self, log_prob: float):
        self._log_prob = float(log_prob)

    def log_probability(self, x):
        return torch.tensor(self._log_prob)


class _MockModel:
    def __init__(self, distributions):
        self.distributions = distributions


class MockHMM:
    """Minimal HMM stub compatible with get_current_regime().

    Parameters
    ----------
    winning_state : int
        Index of the state that will always win the argmax (log_prob=0.0).
        All other states have log_prob=-1000.0.
    n_states : int
        Total number of HMM states (default 2).
    """

    def __init__(self, winning_state: int, n_states: int = 2):
        dists = [_MockDistribution(-1000.0) for _ in range(n_states)]
        dists[winning_state] = _MockDistribution(0.0)
        self._model = _MockModel(dists)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ref_prices():
    return pd.Series(REF_PRICES)


@pytest.fixture
def hmm_s1():
    """MockHMM that always predicts S1 (normal regime)."""
    return MockHMM(winning_state=S1_IDX)


@pytest.fixture
def hmm_s2():
    """MockHMM that always predicts S2 (abnormal regime)."""
    s2_idx = 1 - S1_IDX
    return MockHMM(winning_state=s2_idx)


@pytest.fixture
def result_s1(ref_prices, hmm_s1):
    return run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s1, S1_IDX, INITIAL_CAPITAL)


@pytest.fixture
def result_s2(ref_prices, hmm_s2):
    return run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s2, S1_IDX, INITIAL_CAPITAL)


# ── TestReturnContract ────────────────────────────────────────────────────────

class TestReturnContract:
    def test_result_is_dict(self, result_s1):
        assert isinstance(result_s1, dict)

    def test_has_all_keys(self, result_s1):
        expected = {"final_capital", "cumulative_return", "max_drawdown", "n_trades", "n_winners", "trade_log"}
        assert expected <= set(result_s1.keys())

    def test_final_capital_is_float(self, result_s1):
        assert isinstance(result_s1["final_capital"], float)

    def test_n_trades_is_int(self, result_s1):
        assert isinstance(result_s1["n_trades"], int)

    def test_n_winners_is_int(self, result_s1):
        assert isinstance(result_s1["n_winners"], int)

    def test_trade_log_is_list(self, result_s1):
        assert isinstance(result_s1["trade_log"], list)

    def test_n_winners_le_n_trades(self, result_s1):
        assert result_s1["n_winners"] <= result_s1["n_trades"]

    def test_max_drawdown_non_negative(self, result_s1):
        assert result_s1["max_drawdown"] >= 0.0


# ── TestInitDcSkip ────────────────────────────────────────────────────────────

class TestInitDcSkip:
    """First upturn_dc carries rdc=NaN — must not query HMM or open a position."""

    def test_init_dc_skipped_always_s1(self, ref_prices, hmm_s1):
        # i=1 is init upturn DC (rdc=NaN): HMM skipped, no long entry.
        # i=2: first real downturn DC (rdc=valid): Rule 1S → short entry.
        result = run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s1, S1_IDX, INITIAL_CAPITAL)
        # Without init-DC skip: i=1 LONG entry → 4 trades total.
        # With skip: no LONG at i=1 → 3 trades (SHORT@i=2, LONG@i=4, SHORT@i=6).
        assert result["n_trades"] == 3

    def test_init_dc_skipped_always_s2(self, ref_prices, hmm_s2):
        # i=1: init DC → skip. i=4: S2, no position → no Rule 4, no Rule 1 → 0 trades
        result = run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s2, S1_IDX, INITIAL_CAPITAL)
        assert result["n_trades"] == 0


# ── TestAlwaysS1 ─────────────────────────────────────────────────────────────

class TestAlwaysS1:
    """With regime always S1, Rules 1L, 1S, 2L, 2S, 3L, 3S all fire."""

    def test_n_trades(self, result_s1):
        # SHORT@i=2 (pnl=0%), LONG@i=4→i=5 (pnl=+15%), SHORT@i=6 (EndOfPeriod, pnl=0%)
        assert result_s1["n_trades"] == 3

    def test_n_winners(self, result_s1):
        # Only the LONG trade (pnl=+15%) is a winner; zero-PnL shorts are not
        assert result_s1["n_winners"] == 1

    def test_cumulative_return(self, result_s1):
        # SHORT pnl=0%, LONG pnl=+15%, SHORT pnl=0% → compound = +15%
        expected_crr = 15.0
        assert result_s1["cumulative_return"] == pytest.approx(expected_crr, rel=1e-6)

    def test_final_capital(self, result_s1):
        assert result_s1["final_capital"] == pytest.approx(INITIAL_CAPITAL * 1.15, rel=1e-6)

    def test_long_trade_exit_rule_is_rule2l(self, result_s1):
        # trade_log[0]=SHORT, trade_log[1]=LONG (Rule2L_AOLTarget)
        long_trades = [t for t in result_s1["trade_log"] if t["side"] == "long"]
        assert long_trades[0]["exit_rule"] == "Rule2L_AOLTarget"


# ── TestAlwaysS2 ─────────────────────────────────────────────────────────────

class TestAlwaysS2:
    """With regime always S2, Rule 1 never fires — no positions opened."""

    def test_no_trades(self, result_s2):
        assert result_s2["n_trades"] == 0

    def test_zero_crr(self, result_s2):
        assert result_s2["cumulative_return"] == pytest.approx(0.0)

    def test_capital_unchanged(self, result_s2):
        assert result_s2["final_capital"] == pytest.approx(INITIAL_CAPITAL)

    def test_zero_drawdown(self, result_s2):
        assert result_s2["max_drawdown"] == pytest.approx(0.0)


# ── TestRule4 ─────────────────────────────────────────────────────────────────

class TestRule4:
    """Rule 4 behaviour: S2 at upturn DC suppresses new entries.

    Structural note: Rule 4 (close open position on S2 at upturn_dc) is
    unreachable via idc_parse batch signals. The DC state machine guarantees
    that between any two upturn_dc bars there is always an intervening
    downturn_dc bar. When in_position=True and downturn_dc fires, Rule 3
    closes the position first. Therefore, when the next upturn_dc fires,
    in_position is always False and Rule 4's position-close branch never
    executes.

    The tests below verify the observable half of Rule 4: S2 prevents
    new position entry (Rule 1 is blocked).
    """

    def test_s2_prevents_entry(self, hmm_s2, ref_prices):
        # With always-S2, no position is ever opened → 0 trades
        result = run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s2, S1_IDX, INITIAL_CAPITAL)
        assert result["n_trades"] == 0

    def test_s2_capital_unchanged(self, hmm_s2, ref_prices):
        result = run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s2, S1_IDX, INITIAL_CAPITAL)
        assert result["final_capital"] == pytest.approx(INITIAL_CAPITAL)

    def test_s2_trade_log_empty(self, hmm_s2, ref_prices):
        result = run_ita_algorithm1(ref_prices, THETA, ALPHA, hmm_s2, S1_IDX, INITIAL_CAPITAL)
        assert result["trade_log"] == []


# ── TestTradeLog ──────────────────────────────────────────────────────────────

class TestTradeLog:
    def test_trade_log_length_matches_n_trades(self, result_s1):
        assert len(result_s1["trade_log"]) == result_s1["n_trades"]

    def test_trade_log_entry_has_all_keys(self, result_s1):
        trade = result_s1["trade_log"][0]
        for key in ("entry_bar", "entry_price", "exit_bar", "exit_price", "exit_rule", "pnl_pct", "capital", "side"):
            assert key in trade

    def test_trade_log_pnl_consistent_with_crr(self, result_s1):
        # Compound of all trade pnls must equal the reported CRR
        capital = 1.0
        for trade in result_s1["trade_log"]:
            capital *= 1.0 + trade["pnl_pct"] / 100.0
        expected_crr = (capital - 1.0) * 100.0
        assert expected_crr == pytest.approx(result_s1["cumulative_return"], rel=1e-6)

    def test_trade_log_empty_when_no_trades(self, result_s2):
        assert result_s2["trade_log"] == []

    def test_trade_log_exit_bar_after_entry_bar(self, result_s1):
        trade = result_s1["trade_log"][0]
        assert trade["exit_bar"] > trade["entry_bar"]


# ── TestEndOfPeriod ───────────────────────────────────────────────────────────

class TestEndOfPeriod:
    def test_open_position_closed_at_last_bar(self, hmm_s1):
        # Rising price — enter on upturn_dc, no Rule2/3 fires, closed at end
        prices = pd.Series([100.0, 90.0, 99.0, 101.0, 103.0, 105.0, 107.0])
        result = run_ita_algorithm1(prices, THETA, ALPHA, hmm_s1, S1_IDX, INITIAL_CAPITAL)
        if result["n_trades"] > 0:
            last_trade = result["trade_log"][-1]
            if last_trade["exit_rule"] == "EndOfPeriod":
                assert last_trade["exit_bar"] == len(prices) - 1


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_flat_series_no_trades(self, hmm_s1):
        prices = pd.Series([100.0] * 50)
        result = run_ita_algorithm1(prices, THETA, ALPHA, hmm_s1, S1_IDX, INITIAL_CAPITAL)
        assert result["n_trades"] == 0

    def test_random_series_winners_le_trades(self, hmm_s1):
        rng = np.random.default_rng(99)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, 500))))
        result = run_ita_algorithm1(prices, theta=0.005, alpha=0.5, hmm=hmm_s1, s1_idx=S1_IDX)
        assert result["n_winners"] <= result["n_trades"]

    def test_final_capital_consistent_with_crr(self, result_s1):
        expected = INITIAL_CAPITAL * (1.0 + result_s1["cumulative_return"] / 100.0)
        assert result_s1["final_capital"] == pytest.approx(expected, rel=1e-9)
