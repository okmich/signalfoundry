"""The account-directive guard in the base strategy and the dispatch layer (ACCOUNT_ADMIN_SPEC §10; §14.3, §14.4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from okmich_quant_core.account_admin import (AccountDirective, DirectiveAccount, DirectiveFile, directive_path,
                                             write_directive)
from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import StrategyConfig
from okmich_quant_core.directive_guard import PROCESS_READER, GuardActionStatus, GuardPending, GuardPosition
from okmich_quant_core.logging import BaseEventLogger, GuardedOp, GuardOutcome, LogEventType, RunnerIdentity
from okmich_quant_core.multi_trader import MultiTrader
from okmich_quant_core.signal import BaseSignal

LOGIN, SERVER = 51234567, "ICMarketsSC-Demo"


class _Rec(BaseEventLogger):
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class _Notifier:
    def __init__(self):
        self.events = []

    def on_account_event(self, title, message, level="info"):
        self.events.append((level, title, message))

    def titles(self):
        return [e[1] for e in self.events]

    def close(self):
        pass


class Guarded(BaseStrategy):
    """A strategy on a broker that implements the guard hooks, over an in-memory book."""

    _GUARD_SUPPORTED = True

    def __init__(self, name="sleeve", magic=7, notifier=None, fail_on_bar=False):
        self.rec = _Rec()
        super().__init__(StrategyConfig(name=name, symbol="EURUSD", timeframe=5, magic=magic), BaseSignal(),
                         notifier=notifier, inference_logger=self.rec)
        self.pending = [101]
        self.positions = {201: True}
        self.close_fails = False
        self.identity_calls = 0
        self.fail_on_bar = fail_on_bar
        self.entries_sent = []
        self.bind_runner_identity(RunnerIdentity.generate(name="r", broker=SERVER, account_id=str(LOGIN)))

    def is_new_bar(self, run_dt):
        return True

    def on_new_bar(self):
        if self.fail_on_bar:
            raise ValueError("boom")

    # a broker order method, as the MT5 base class writes it
    def open(self, long=True):
        op = GuardedOp.OPEN_LONG if long else GuardedOp.OPEN_SHORT
        if not self.guard_entry(op, detail="0.10 lots @ 1.1000"):
            return False
        self.entries_sent.append(op)
        return True

    def exit_by_own_rule(self, ticket):
        """The system's own exit: never goes through the guard (invariant 2)."""
        self.positions.pop(ticket, None)
        return True

    def _guard_terminal_identity(self):
        self.identity_calls += 1
        return LOGIN, SERVER

    def _guard_own_pending(self):
        return [GuardPending(t, "buy_limit 0.10 lots @ 1.0950") for t in self.pending]

    def _guard_cancel_pending(self, ticket):
        self.pending.remove(ticket)
        return GuardActionStatus.DONE, "cancelled"

    def _guard_own_positions(self):
        return [GuardPosition(t, long, "0.10 lots @ 1.1000, P&L -3.20") for t, long in self.positions.items()]

    def _guard_close_position(self, ticket, directive):
        if self.close_fails:
            return GuardActionStatus.FAILED, "market closed"
        if ticket not in self.positions:
            return GuardActionStatus.GONE, "already closed"
        self.positions.pop(ticket)
        return GuardActionStatus.DONE, "closed"

    def records(self, event):
        return [r for r in self.rec.records if r.envelope.event is event]


@pytest.fixture
def live(tmp_path, monkeypatch):
    base = tmp_path / "live"
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(base))
    return base


def publish(base, directive, *, age_s=0, episode=1, login=LOGIN):
    now = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    write_directive(directive_path(base, "icmarkets.demo"), DirectiveFile(
        DirectiveAccount("icmarkets", SERVER, login, "USD"), directive, f"test {directive}", ("test",), now, now, 240,
        episode * 10, episode))
    PROCESS_READER.invalidate()


def test_all_ops_lets_entries_through_and_records_the_applied_directive(live):
    publish(live, AccountDirective.ALL_OPS)
    s = Guarded()
    assert s.open() and s.open(long=False)
    applied = s.records(LogEventType.ACCOUNT_DIRECTIVE_APPLIED)
    assert len(applied) == 1 and applied[0].directive == "ALL_OPS" and applied[0].directive_source == "file"
    assert not s.records(LogEventType.ACCOUNT_DIRECTIVE_OP)


def test_no_entry_ops_suppresses_every_entry_kind_and_alerts_once_per_op(live):
    publish(live, AccountDirective.NO_ENTRY_OPS)
    n = _Notifier()
    s = Guarded(notifier=n)
    for _ in range(5):                                                     # a signal that stays on re-asks every bar
        assert s.open() is False
    assert s.open(long=False) is False
    assert s.guard_entry(GuardedOp.PLACE_PENDING, detail="buy_limit 0.10 @ 1.09") is False
    assert s.entries_sent == []
    ops = s.records(LogEventType.ACCOUNT_DIRECTIVE_OP)
    assert [o.op for o in ops].count(GuardedOp.OPEN_LONG) == 5 and all(o.outcome is GuardOutcome.SUPPRESSED for o in ops)
    record = ops[0].to_dict()
    assert not {"volume", "price", "profit", "lots"} & set(record)        # no size, price or P&L on the ops channel
    assert n.titles().count("ENTRY SUPPRESSED: open_long") == 1
    assert n.titles().count("ENTRY SUPPRESSED: open_short") == 1


def test_no_entry_ops_cancels_own_pending_orders_and_leaves_positions(live):
    publish(live, AccountDirective.NO_ENTRY_OPS)
    s = Guarded()
    s.enforce_account_directive(datetime.now())
    assert s.pending == [] and s.positions == {201: True}
    forced = s.records(LogEventType.ACCOUNT_DIRECTIVE_OP)
    assert [(o.op, o.outcome) for o in forced] == [(GuardedOp.CANCEL_PENDING, GuardOutcome.FORCED)]


def test_closing_is_never_blocked(live):
    publish(live, AccountDirective.NO_OPS)
    s = Guarded()
    assert s.exit_by_own_rule(201) and s.positions == {}


def test_no_ops_closes_a_circuit_broken_strategy_through_the_dispatch_layer(live):
    """§14.3: the breaker disabled the strategy; the dispatch-layer hook still closes its position."""
    publish(live, AccountDirective.ALL_OPS)
    broken, healthy = Guarded("broken", magic=1, fail_on_bar=True), Guarded("healthy", magic=2)
    mt = MultiTrader([broken, healthy], max_consecutive_errors=1)
    mt.run(datetime(2026, 9, 24, 13, 0))
    assert not mt.health_trackers["broken"].is_enabled
    publish(live, AccountDirective.NO_OPS)
    mt.check_positions(datetime(2026, 9, 24, 13, 0, 30))
    assert broken.positions == {} and healthy.positions == {} and broken.pending == []
    closes = [o for o in broken.records(LogEventType.ACCOUNT_DIRECTIVE_OP) if o.op is GuardedOp.CLOSE_LONG]
    assert closes and closes[0].outcome is GuardOutcome.FORCED and closes[0].directive == "NO_OPS"


def test_a_failing_forced_close_is_alerted_once_per_episode(live):
    publish(live, AccountDirective.NO_OPS)
    n = _Notifier()
    s = Guarded(notifier=n)
    s.close_fails = True
    for _ in range(3):
        s.enforce_account_directive(datetime.now())
    assert n.titles().count("FORCED close_long FAILED") == 1 and s.positions == {201: True}
    s.close_fails = False
    s.enforce_account_directive(datetime.now())
    assert s.positions == {}


@pytest.mark.parametrize("age_s,login,source", [(600, LOGIN, "stale"), (0, 1, "wrong_account")])
def test_fallbacks_apply_no_entry_and_alert_once_per_runner(live, monkeypatch, age_s, login, source):
    from okmich_quant_core import directive_guard
    monkeypatch.setattr(directive_guard._ProcessReader, "account", lambda self: "icmarkets.demo")   # in the folder
    publish(live, AccountDirective.ALL_OPS, age_s=age_s, login=login)
    n1, n2 = _Notifier(), _Notifier()
    a, b = Guarded("a", magic=1, notifier=n1), Guarded("b", magic=2, notifier=n2)
    assert a.open() is False and b.open() is False
    assert a.records(LogEventType.ACCOUNT_DIRECTIVE_APPLIED)[0].directive_source == source
    fallback_alerts = [t for t in n1.titles() + n2.titles() if t.startswith("DIRECTIVE FALLBACK:")]
    assert len(fallback_alerts) == 1                                       # one per runner, not one per sleeve
    publish(live, AccountDirective.ALL_OPS)
    assert a.open() and b.open()
    assert [t for t in n1.titles() + n2.titles() if t == "DIRECTIVE FALLBACK ENDED"] == ["DIRECTIVE FALLBACK ENDED"]


def test_episode_summary_counts_every_suppression(live):
    publish(live, AccountDirective.NO_ENTRY_OPS, episode=4)
    n = _Notifier()
    s = Guarded(notifier=n)
    for _ in range(3):
        s.open()
    s.open(long=False)
    publish(live, AccountDirective.ALL_OPS, episode=5)
    assert s.open()
    summary = [m for _, t, m in n.events if t == "DIRECTIVE EPISODE SUMMARY"]
    assert len(summary) == 1 and "3 x open_long" in summary[0] and "1 x open_short" in summary[0]
    applied = s.records(LogEventType.ACCOUNT_DIRECTIVE_APPLIED)
    assert [(r.previous_directive, r.directive) for r in applied] == [(None, "NO_ENTRY_OPS"), ("NO_ENTRY_OPS", "ALL_OPS")]


def test_outside_an_account_folder_another_logins_directive_does_not_apply(live):
    publish(live, AccountDirective.NO_OPS, login=1)
    s = Guarded()
    assert s.open()
    assert s.records(LogEventType.ACCOUNT_DIRECTIVE_APPLIED)[0].directive_source == "absent"


def test_an_ungoverned_account_costs_no_terminal_query(live, monkeypatch):
    from okmich_quant_core import directive_guard
    monkeypatch.setattr(directive_guard._ProcessReader, "account", lambda self: "icmarkets.demo")
    s = Guarded()
    assert s.open()
    assert s.identity_calls == 0
    assert s.records(LogEventType.ACCOUNT_DIRECTIVE_APPLIED)[0].directive_source == "absent"


def test_a_broker_without_the_hooks_is_inert(live):
    publish(live, AccountDirective.NO_OPS)

    class Plain(Guarded):
        _GUARD_SUPPORTED = False

    s = Plain()
    assert s.open() and s.positions == {201: True}
    s.enforce_account_directive(datetime.now())
    assert s.positions == {201: True} and not s.rec.records


def test_the_guard_cannot_be_overridden():
    with pytest.raises(TypeError, match="guard_entry"):
        class Sneaky(Guarded):
            def guard_entry(self, op, *, detail, signal_bar_utc=None):
                return True
    with pytest.raises(TypeError, match="enforce_account_directive"):
        class Sneakier(Guarded):
            def enforce_account_directive(self, run_dt):
                pass
