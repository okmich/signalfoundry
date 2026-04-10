"""
One-cycle correctness regression tests.

These tests verify that when a position has no initial SL (sl=0), the first
management cycle both sets the initial SL AND applies trailing/break-even
protection in the same cycle — without requiring a second cycle to activate.

This guards against the stale-snapshot bug where `sl = position["sl"]` (0)
was used for trailing/break-even comparisons after `_set_initial_sl_tp*` had
already sent the real SL to the broker.

Short-side tests are especially critical because the predicates
`new_sl < sl` and `sl > price_open` are unsatisfiable when sl=0 for any
positive-priced instrument.
"""

from unittest.mock import call, patch

import pytest

from okmich_quant_core.config import PositionManagerConfig, PositionManagerType
from okmich_quant_mt5.position_manager import get_position_manager
from . import with_strategy_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRICE_OPEN = 1.1000
POINT_SIZE = 0.0001


def make_position(ticket, price_open, price_current, sl=0.0, tp=0.0, is_long=True):
    return {
        "ticket": ticket,
        "type": 0 if is_long else 1,
        "profit": 0.0,
        "price_open": price_open,
        "price_current": price_current,
        "symbol": "EURUSD",
        "volume": 0.1,
        "sl": sl,
        "tp": tp,
    }


BASE_KWARGS = {"symbol": "EURUSD", "magic": 12345, "system_name": "TestSystem"}


# ---------------------------------------------------------------------------
# Point-based — trailing
# ---------------------------------------------------------------------------

class TestPointTrailingOneCycle:

    def test_short_trailing_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        Short position: sl=0, price_current already below price_open (profitable).
        Expected: two modify_position calls in ONE cycle:
          1. Set initial SL/TP.
          2. Trail SL downward (new_sl < effective_sl).
        """
        # sl_points=50 → initial short SL = 1.1050
        # trailing_points=20 → trail_amount = 0.0020
        # price_current = 1.0950 (50 pips below open → profitable)
        # new_sl_trail = 1.0950 + 0.0020 = 1.0970 < 1.1050 → should trail
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=POINT_SIZE,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=1,
            price_open=PRICE_OPEN,
            price_current=1.0950,  # 50 pips below → short in profit
            sl=0.0,
            is_long=False,
        )

        manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN + 50 * POINT_SIZE  # 1.1050
        expected_initial_tp = PRICE_OPEN - 100 * POINT_SIZE  # 1.0900
        expected_trail_sl = 1.0950 + 20 * POINT_SIZE           # 1.0970

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 modify_position calls, got {len(calls)}"
        # First call: set initial SL/TP
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[0][1]["tp"] - expected_initial_tp) < 1e-9
        # Second call: trail SL only (tp=0.0 because base always passes tp)
        assert abs(calls[1][1]["sl"] - expected_trail_sl) < 1e-9
        assert calls[1][1].get("tp", 0.0) == 0.0

    def test_long_trailing_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        Long position: sl=0, price_current already above price_open (profitable).
        Expected: two calls — set initial SL/TP, then trail SL upward.
        """
        # sl_points=50 → initial long SL = 1.0950
        # trailing_points=20 → trail_amount = 0.0020
        # price_current = 1.1060 (60 pips above → profitable)
        # new_sl_trail = 1.1060 - 0.0020 = 1.1040 > 1.0950 → should trail
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=POINT_SIZE,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=2,
            price_open=PRICE_OPEN,
            price_current=1.1060,
            sl=0.0,
        )

        manager.manage_long_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN - 50 * POINT_SIZE  # 1.0950
        expected_initial_tp = PRICE_OPEN + 100 * POINT_SIZE  # 1.1100
        expected_trail_sl = 1.1060 - 20 * POINT_SIZE          # 1.1040

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_trail_sl) < 1e-9

    def test_short_no_trail_when_price_not_moved(self, mock_mt5_functions):
        """
        Short position: sl=0, but price hasn't moved (no profit yet).
        Expected: one call only — initial SL/TP set; no trailing.
        """
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_TRAILING,
            sl=50, tp=100, trailing=20, point_size=POINT_SIZE,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=3,
            price_open=PRICE_OPEN,
            price_current=PRICE_OPEN,  # no movement
            sl=0.0,
            is_long=False,
        )

        manager.manage_short_position(position, flag=True)

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 1  # only initial SL/TP


# ---------------------------------------------------------------------------
# Point-based — break-even
# ---------------------------------------------------------------------------

class TestPointBreakEvenOneCycle:

    def test_short_break_even_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        Short position: sl=0, price_current past break-even threshold.
        Expected: two calls — set initial SL/TP, then move SL to break-even.
        """
        # sl=50 → initial short SL = 1.1050
        # break_even=30 → threshold = 0.0030; break_even_price = 1.0970
        # trailing=10 → trail_amount = 0.0010
        # price_current = 1.0960 (past break_even_price=1.0970 downward)
        # new_sl = 1.0960 + 0.0010 = 1.0970; min(1.0970, 1.1000)=1.0970 < 1.1050 → move
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
            sl=50, tp=100, break_even=30, trailing=10, point_size=POINT_SIZE,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=4,
            price_open=PRICE_OPEN,
            price_current=1.0960,  # 40 pips below open → past break-even (30 pips)
            sl=0.0,
            is_long=False,
        )

        manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN + 50 * POINT_SIZE   # 1.1050
        expected_initial_tp = PRICE_OPEN - 100 * POINT_SIZE  # 1.0900
        expected_be_sl = min(1.0960 + 10 * POINT_SIZE, PRICE_OPEN)  # 1.0970

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_be_sl) < 1e-9

    def test_short_no_break_even_when_sl_already_below_open(self, mock_mt5_functions):
        """
        Short position: both SL and TP already set; SL is at/below price_open
        (break-even already applied in a prior cycle).
        Expected: no modify_position calls — _set_initial_sl_tp skips (both set),
        and break-even guard `sl > price_open` blocks re-activation.
        """
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN,
            sl=50, tp=100, break_even=30, trailing=10, point_size=POINT_SIZE,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        # sl < price_open means break-even was already applied for this short
        position = {
            "ticket": 5,
            "type": 1,
            "profit": 80.0,
            "price_open": PRICE_OPEN,
            "price_current": 1.0950,
            "symbol": "EURUSD",
            "volume": 0.1,
            "sl": 1.0970,   # already at break-even — sl < price_open
            "tp": 1.0900,   # tp also set
        }

        manager.manage_short_position(position, flag=True)

        # sl=1.0970 < price_open=1.1000 → `sl > price_open` guard = False → no break-even
        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 0  # both SL/TP set and break-even guard blocks


# ---------------------------------------------------------------------------
# Percent-based — trailing
# ---------------------------------------------------------------------------

class TestPercentTrailingOneCycle:

    def test_short_trailing_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        Percent-based trailing: short, sl=0, price already moved favorably.
        Expected: two calls — initial SL/TP, then trail.
        """
        # sl_percent=1.0 → initial short SL = 1.1000 * 1.01 = 1.1110
        # tp_percent=2.0 → initial short TP = 1.1000 * 0.98 = 1.0780
        # trailing_percent=0.5 → trailing_amount = 1.1000 * 0.005 = 0.0055
        # price_current = 1.0940 (profitable for short)
        # new_sl = 1.0940 + 0.0055 = 1.0995 < 1.1110 → trail
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
            sl=1.0, tp=2.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=6,
            price_open=PRICE_OPEN,
            price_current=1.0940,
            sl=0.0,
            is_long=False,
        )

        manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN * 1.01   # 1.1110
        expected_initial_tp = PRICE_OPEN * 0.98   # 1.0780
        # Trailing is anchored to price_current, not price_open
        price_current = 1.0940
        trailing_amount = price_current * 0.005
        expected_trail_sl = price_current + trailing_amount

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_trail_sl) < 1e-9


# ---------------------------------------------------------------------------
# Percent-based — break-even
# ---------------------------------------------------------------------------

class TestPercentBreakEvenOneCycle:

    def test_short_break_even_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        Percent break-even: short, sl=0, price past threshold.
        Expected: two calls.
        """
        # sl_percent=1.0 → initial short SL = 1.1110
        # break_even_percent=1.0 → threshold = 1.1000 * 0.01 = 0.0110; be_price = 1.0890
        # trailing_percent=0.5 → trailing_amount = 0.0055
        # price_current = 1.0880 (past break-even price of 1.0890)
        # new_sl = 1.0880 + 0.0055 = 1.0935; min(1.0935, 1.1000) = 1.0935 < 1.1110 → move
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
            sl=1.0, tp=2.0, break_even=1.0, trailing=0.5,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=7,
            price_open=PRICE_OPEN,
            price_current=1.0880,
            sl=0.0,
            is_long=False,
        )

        manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN * 1.01  # 1.1110
        # Trailing is anchored to price_current, not price_open
        price_current = 1.0880
        trailing_amount = price_current * 0.005
        expected_be_sl = min(price_current + trailing_amount, PRICE_OPEN)

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_be_sl) < 1e-9


# ---------------------------------------------------------------------------
# ATR-based — trailing
# ---------------------------------------------------------------------------

class TestAtrTrailingOneCycle:

    def test_short_trailing_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        ATR trailing: short, sl=0, price already moved favorably.
        Expected: two calls — initial SL/TP, then trail.
        """
        # ATR = 0.0010; sl_mult=1.5; tp_mult=3.0; trailing_mult=2.0
        # initial short SL = 1.1000 + 0.0010*1.5 = 1.1015
        # initial short TP = 1.1000 - 0.0010*3.0 = 1.0970
        # trailing_amount = 0.0010 * 2.0 = 0.0020
        # price_current = 1.0970 (profitable for short)
        # new_sl = 1.0970 + 0.0020 = 1.0990 < 1.1015 → trail
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_TRAILING,
            sl=1.5, tp=3.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=8,
            price_open=PRICE_OPEN,
            price_current=1.0970,
            sl=0.0,
            is_long=False,
        )

        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN + atr * 1.5  # 1.1015
        expected_initial_tp = PRICE_OPEN - atr * 3.0  # 1.0970
        expected_trail_sl = 1.0970 + atr * 2.0         # 1.0990

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_trail_sl) < 1e-9


# ---------------------------------------------------------------------------
# ATR-based — break-even
# ---------------------------------------------------------------------------

class TestAtrBreakEvenOneCycle:

    def test_short_break_even_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        ATR break-even: short, sl=0, price past break-even threshold.
        Expected: two calls — initial SL/TP, then break-even SL.
        """
        # ATR=0.0010; sl_mult=1.5; tp_mult=3.0; be_mult=1.0; trailing_mult=2.0
        # initial short SL = 1.1000 + 0.0015 = 1.1015
        # be_threshold = 0.0010 * 1.0 = 0.0010; be_price = 1.1000 - 0.0010 = 1.0990
        # price_current = 1.0985 (past be_price downward)
        # trailing_amount = 0.0010 * 2.0 = 0.0020
        # new_sl = 1.0985 + 0.0020 = 1.1005; min(1.1005, 1.1000) = 1.1000
        # 1.1000 < 1.1015 → move
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=9,
            price_open=PRICE_OPEN,
            price_current=1.0985,
            sl=0.0,
            is_long=False,
        )

        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_short_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN + atr * 1.5  # 1.1015
        expected_initial_tp = PRICE_OPEN - atr * 3.0  # 1.0970
        trailing_amount = atr * 2.0                    # 0.0020
        expected_be_sl = min(1.0985 + trailing_amount, PRICE_OPEN)  # min(1.1005, 1.1000)=1.1000

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_be_sl) < 1e-9

    def test_long_break_even_activates_same_cycle_when_sl_zero(
        self, mock_mt5_functions
    ):
        """
        ATR break-even: long, sl=0, price past break-even threshold.
        Expected: two calls — initial SL/TP, then break-even SL.
        """
        # ATR=0.0010; sl_mult=1.5; tp_mult=3.0; be_mult=1.0; trailing_mult=2.0
        # initial long SL = 1.1000 - 0.0015 = 1.0985
        # be_threshold = 0.0010; be_price = 1.1000 + 0.0010 = 1.1010
        # price_current = 1.1020 (past be_price)
        # new_sl = 1.1020 - 0.0020 = 1.1000; max(1.1000, 1.1000) = 1.1000
        # 1.1000 > 1.0985 → move
        config = PositionManagerConfig(
            type=PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN,
            sl=1.5, tp=3.0, break_even=1.0, trailing=2.0, atr_period=14,
        )
        manager = get_position_manager(with_strategy_config(config), **BASE_KWARGS)

        position = make_position(
            ticket=10,
            price_open=PRICE_OPEN,
            price_current=1.1020,
            sl=0.0,
        )

        atr = 0.0010
        with patch.object(manager, "_get_current_atr", return_value=atr):
            manager.manage_long_position(position, flag=True)

        expected_initial_sl = PRICE_OPEN - atr * 1.5  # 1.0985
        trailing_amount = atr * 2.0
        expected_be_sl = max(1.1020 - trailing_amount, PRICE_OPEN)  # max(1.1000, 1.1000)=1.1000

        calls = mock_mt5_functions["modify_position"].call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        assert abs(calls[0][1]["sl"] - expected_initial_sl) < 1e-9
        assert abs(calls[1][1]["sl"] - expected_be_sl) < 1e-9