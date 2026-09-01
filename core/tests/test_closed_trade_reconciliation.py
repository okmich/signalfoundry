"""Closed-position reconciliation: the shared half of close detection (BaseStrategy.sync_positions).

The behaviours pinned here are the ones whose failure is SILENT in production — a book that was never observed
being read as an empty one, a close announced twice or not at all, a strategy's own close intent being lost to
the broker's coarser label. None of these raise; they just quietly corrupt the trade record.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.closed_trade import ClosedTrade, CloseReason
from okmich_quant_core.config import StrategyConfig
from okmich_quant_core.logging import BaseEventLogger, RunnerIdentity
from okmich_quant_core.notification.base import BaseNotifier
from okmich_quant_core.signal import BaseSignal


class _NullLogger(BaseEventLogger):
    def write(self, record):
        pass

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class _SpyNotifier(BaseNotifier):
    def __init__(self):
        self.closed = []

    def on_trade_opened(self, **kw):
        pass

    def on_trade_closed(self, symbol, ticket, profit, price=0.0, reason=""):
        self.closed.append({"symbol": symbol, "ticket": ticket, "profit": profit, "price": price, "reason": reason})

    def on_trade_modified(self, **kw):
        pass

    def on_error(self, strategy_name, error_message, context=None):
        pass

    def on_circuit_breaker_tripped(self, strategy_name, consecutive_errors):
        pass

    def on_connection_lost(self, strategy_name):
        pass

    def on_connection_restored(self, strategy_name):
        pass

    def close(self):
        pass


class _Strat(BaseStrategy):
    """Test double whose observation of the book and resolution of a close are both scripted."""

    def __init__(self, notifier=None):
        super().__init__(StrategyConfig(name="s", symbol="EURUSD", timeframe=5, magic=7),
                         BaseSignal(), notifier=notifier, inference_logger=_NullLogger())
        self.bind_runner_identity(RunnerIdentity(runner_id="r", runner_start_token="t", broker="b",
                                                 account_id="a", broker_session_id="s"))
        self.book = None                 # what manage_positions will report
        self.resolved = None             # what resolve_closed_trade will return
        self.resolve_calls = []
        self.resolve_raises = False

    def is_new_bar(self, run_dt):
        return False

    def on_new_bar(self):
        pass

    def manage_positions(self, run_dt, flag=False):
        return self.book

    def resolve_closed_trade(self, key, last_seen):
        self.resolve_calls.append((key, last_seen))
        if self.resolve_raises:
            raise RuntimeError("history unavailable")
        return self.resolved


def _pos(ticket, **kw):
    base = {"ticket": ticket, "volume": 0.1, "price_open": 1.1, "profit": 3.0}
    base.update(kw)
    return base


def _dt(second=0):
    return datetime(2026, 9, 1, 12, 0, second, tzinfo=timezone.utc)


def _closed(key, reason=CloseReason.TAKE_PROFIT, profit=12.0):
    return ClosedTrade(key=key, symbol="EURUSD", magic=7, reason=reason, volume=0.1,
                       entry_price=1.1, exit_price=1.2, profit=profit)


# --------------------------------------------------------------------------------------
# empty vs unknown — the fail-OPEN trap
# --------------------------------------------------------------------------------------

def test_none_means_unobserved_and_never_reports_a_close():
    """A book the broker did not return must NOT read as 'everything closed'."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1), _pos(2)]
    s.sync_positions(_dt(0))
    assert s.open_position_count == 2

    s.book = None                                   # debounced / disconnected
    assert s.sync_positions(_dt(5)) is None
    assert s.notifier.closed == []                  # no phantom exits
    assert s.open_position_count == 2               # last known count is retained, not zeroed
    assert set(s._open_trades) == {"1", "2"}        # still tracked


def test_empty_list_means_flat_and_does_report_closes():
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))

    s.resolved = _closed("1")
    s.book = []                                     # genuinely flat
    s.sync_positions(_dt(5))
    assert [c["ticket"] for c in s.notifier.closed] == ["1"]
    assert s.open_position_count == 0


def test_only_the_vanished_position_is_reported():
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1), _pos(2)]
    s.sync_positions(_dt(0))

    s.resolved = _closed("2")
    s.book = [_pos(1)]
    s.sync_positions(_dt(5))
    assert [c["ticket"] for c in s.notifier.closed] == ["2"]
    assert set(s._open_trades) == {"1"}


# --------------------------------------------------------------------------------------
# once-only emission
# --------------------------------------------------------------------------------------

def test_a_close_is_announced_exactly_once():
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.resolved = _closed("1")
    s.book = []
    s.sync_positions(_dt(5))
    s.sync_positions(_dt(10))
    s.sync_positions(_dt(15))
    assert len(s.notifier.closed) == 1


def test_reopened_ticket_is_tracked_again_after_closing():
    """Tracking must not be poisoned by a key that already closed once."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.resolved = _closed("1")
    s.book = []
    s.sync_positions(_dt(5))

    s.book = [_pos(1)]                              # broker reissued the id
    s.sync_positions(_dt(10))
    assert set(s._open_trades) == {"1"}
    s.book = []
    s.sync_positions(_dt(15))
    assert len(s.notifier.closed) == 2


# --------------------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------------------

def test_broker_take_profit_is_not_overwritten_by_strategy_intent():
    """A TP fill is a fact. Intent must not relabel it — that would hide which exits the targets earned."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.note_close_intent(1, "ctl_flip")
    s.resolved = _closed("1", reason=CloseReason.TAKE_PROFIT)
    s.book = []
    s.sync_positions(_dt(5))
    assert s.notifier.closed[0]["reason"] == CloseReason.TAKE_PROFIT.value


def test_strategy_intent_refines_a_coarse_broker_reason():
    """'closed by client' is all MT5 can say; only the strategy knows which rule asked for it."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.note_close_intent(1, "ctl_flip")
    s.resolved = _closed("1", reason=CloseReason.MANUAL)
    s.book = []
    closed = s.observe_open_positions([])
    assert closed[0].reason == CloseReason.STRATEGY
    assert closed[0].strategy_reason == "ctl_flip"


def test_unresolved_close_is_reported_and_flagged():
    """A close we could not resolve is still a close — reported, marked unresolved, described from last-seen."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1, volume=0.25, price_open=1.2345)]
    s.sync_positions(_dt(0))
    s.resolved = None                               # history had nothing
    s.book = []
    closed = s.observe_open_positions([])
    assert len(closed) == 1
    assert closed[0].resolved is False
    assert closed[0].reason == CloseReason.UNKNOWN
    assert closed[0].volume == 0.25                 # falls back to the last-seen snapshot
    assert closed[0].entry_price == 1.2345
    assert "UNRESOLVED" in closed[0].describe()


def test_resolution_failure_does_not_lose_the_close():
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.resolve_raises = True
    s.book = []
    s.sync_positions(_dt(5))
    assert len(s.notifier.closed) == 1
    assert s._open_trades == {}


def test_notifier_receives_net_profit_including_costs():
    """Swap and commission are booked apart from profit; reporting the gross figure overstates every trade."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.resolved = ClosedTrade(key="1", symbol="EURUSD", reason=CloseReason.STOP_LOSS, profit=10.0,
                             commission=-1.5, swap=-0.25, exit_price=1.2)
    s.book = []
    s.sync_positions(_dt(5))
    assert s.notifier.closed[0]["profit"] == pytest.approx(8.25)
    assert s.notifier.closed[0]["price"] == 1.2


# --------------------------------------------------------------------------------------
# tracking at entry — the fast open-and-close race
# --------------------------------------------------------------------------------------

def test_position_opened_and_closed_between_observations_is_still_reported():
    s = _Strat(notifier=_SpyNotifier())
    s.book = []
    s.sync_positions(_dt(0))
    s.register_open_position(_pos(99))              # entry: tracked before any sweep could see it
    s.resolved = _closed("99")
    s.book = []                                     # already gone by the next observation
    s.sync_positions(_dt(5))
    assert [c["ticket"] for c in s.notifier.closed] == ["99"]


def test_register_open_position_is_idempotent():
    s = _Strat()
    s.register_open_position(_pos(1))
    first_seen = s._open_trades["1"]["first_seen"]
    s.register_open_position(_pos(1))
    assert s._open_trades["1"]["first_seen"] == first_seen


# --------------------------------------------------------------------------------------
# identity + compatibility
# --------------------------------------------------------------------------------------

def test_position_key_falls_back_to_broker_specific_ids():
    s = _Strat()
    assert s._position_key({"ticket": 5}) == "5"
    assert s._position_key({"conId": 77}) == "77"
    assert s._position_key({"nothing": 1}) is None


def test_position_without_identity_is_skipped_not_fatal():
    s = _Strat(notifier=_SpyNotifier())
    s.book = [{"volume": 1.0}]
    s.sync_positions(_dt(0))
    assert s._open_trades == {}
    assert s.notifier.closed == []


def test_legacy_int_return_disables_reconciliation_without_crashing():
    """A pre-migration subclass still returning a count keeps working; it just never observes the book."""
    s = _Strat(notifier=_SpyNotifier())
    s.book = 0
    assert s.sync_positions(_dt(0)) is None
    assert s.sync_positions(_dt(5)) is None
    assert s._warned_legacy_manage_positions is True
    assert s.notifier.closed == []


def test_sync_positions_is_sealed():
    with pytest.raises(TypeError, match="sealed BaseStrategy.sync_positions"):
        class Bad(BaseStrategy):
            def sync_positions(self, run_dt, flag=False):
                return None

            def is_new_bar(self, run_dt):
                return False

            def on_new_bar(self):
                pass


def test_notifier_failure_does_not_break_position_management():
    class _Boom(_SpyNotifier):
        def on_trade_closed(self, *a, **kw):
            raise RuntimeError("telegram down")

    s = _Strat(notifier=_Boom())
    s.book = [_pos(1)]
    s.sync_positions(_dt(0))
    s.resolved = _closed("1")
    s.book = []
    s.sync_positions(_dt(5))                        # must not raise
    assert s._open_trades == {}


def test_reconciliation_fault_does_not_break_position_management():
    """Bookkeeping must never take down the sweep that protects still-open positions."""
    class _BrokenKey(_Strat):
        def _position_key(self, position):
            raise RuntimeError("bad key")

    s = _BrokenKey(notifier=_SpyNotifier())
    s.book = [_pos(1), _pos(2)]
    assert s.sync_positions(_dt(0)) == [_pos(1), _pos(2)]     # sweep still returns the book
    assert s.open_position_count == 2                          # and the count is still correct
