"""The MT5 side of the Account Admin: server clock, account source, book actions, and run_admin's two startup stages
against a fake terminal (ACCOUNT_ADMIN_SPEC §6.7, §7.5, §12.3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from okmich_quant_core.account_admin import (BookActionOutcome, DealEntry, DealKind, PendingOrderType,
                                             parse_admin_config)
from okmich_quant_mt5.account_admin import Mt5AccountSource, Mt5BrokerActions, ServerClock, parse_broker_section
from okmich_quant_mt5.account_admin import app as app_mod

UTC = timezone.utc
NY7 = ServerClock(ZoneInfo("America/New_York"), 7.0)
LOGIN, SERVER = 51234567, "ICMarketsSC-Demo"


def server_epoch(utc: datetime) -> int:
    return NY7.to_server_epoch(utc)


# ------------------------------------------------------------------------------------------------ server clock


def test_server_clock_follows_us_dst():
    summer = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)          # EDT: server UTC+3
    winter = datetime(2026, 12, 1, 13, 0, tzinfo=UTC)          # EST: server UTC+2
    assert NY7.offset_h_at(summer) == 3.0 and NY7.offset_h_at(winter) == 2.0
    for instant in (summer, winter):
        assert NY7.to_utc(server_epoch(instant)) == instant
    # server midnight is 17:00 New York: the NY-close day boundary
    assert NY7.to_utc(datetime(2026, 9, 25, 0, 0, tzinfo=UTC).timestamp()) == datetime(2026, 9, 24, 21, 0, tzinfo=UTC)


def test_a_tick_from_the_future_proves_the_clock_wrong():
    now = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    true_tick = server_epoch(now - timedelta(seconds=2))
    assert NY7.tick_contradicts(true_tick, now) is None
    assert NY7.tick_contradicts(server_epoch(now - timedelta(days=2)), now) is None      # stale: proves nothing
    naive_utc_clock = ServerClock(ZoneInfo("UTC"), 0.0)                                   # the "read it as UTC" mistake
    assert "future" in naive_utc_clock.tick_contradicts(true_tick, now)


@pytest.mark.parametrize("raw,needle", [(None, "missing"), ({"server_clock_tz": "Nowhere/Land", "server_clock_shift_h": 7},
                                                            "IANA"),
                                        ({"server_clock_tz": "America/New_York", "server_clock_shift_h": 7.3}, "quarter"),
                                        ({"server_clock_tz": "America/New_York", "server_clock_shift_h": 7, "x": 1},
                                         "unknown key")])
def test_broker_section_validation(raw, needle):
    clock, problems = parse_broker_section(raw)
    assert clock is None and any(needle in p for p in problems), problems


# ------------------------------------------------------------------------------------------------ fake terminal


class FakeMt5:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_STOP = 0, 1, 2, 5
    DEAL_TYPE_BUY, DEAL_TYPE_SELL, DEAL_TYPE_BALANCE = 0, 1, 2
    DEAL_ENTRY_IN, DEAL_ENTRY_OUT = 0, 1
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_REMOVE, TRADE_RETCODE_DONE = 8, 10009

    def __init__(self, login=LOGIN, server=SERVER):
        self.login, self.server = login, server
        self.calls: list[str] = []
        self.orders: list[SimpleNamespace] = []
        self.deals: list[SimpleNamespace] = []
        self.sent: list[dict] = []
        self.remove_retcode = 10009
        self.attached = False

    def _log(self, name):
        self.calls.append(name)

    def initialize(self, path=None, **kw):
        self._log("initialize")
        if kw:
            raise AssertionError("the Admin must attach only, never log in")
        self.attached = True
        return True

    def login(self, *a, **k):
        raise AssertionError("the Admin must never call login")

    def shutdown(self):
        self._log("shutdown")
        self.attached = False

    def terminal_info(self):
        return SimpleNamespace(name="MetaTrader 5", build=5000) if self.attached else None

    def last_error(self):
        return (1, "ok")

    def account_info(self):
        self._log("account_info")
        return SimpleNamespace(login=self.login, server=self.server, currency="USD", balance=100_000.0, equity=100_000.0)

    def positions_get(self, ticket=None):
        return ()

    def orders_get(self, ticket=None):
        if ticket is not None:
            return tuple(o for o in self.orders if o.ticket == ticket)
        return tuple(self.orders)

    def history_deals_get(self, start, end):
        return tuple(d for d in self.deals if start <= d.time <= end)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(time=server_epoch(datetime.now(UTC)), bid=1.1, ask=1.1001)

    def order_send(self, request):
        self.sent.append(request)
        if request["action"] == self.TRADE_ACTION_REMOVE and self.remove_retcode == 10009:
            self.orders = [o for o in self.orders if o.ticket != request["order"]]
        return SimpleNamespace(retcode=self.remove_retcode, comment="done" if self.remove_retcode == 10009 else "busy")


def _order(ticket, setup_utc, type_=2, magic=7):
    return SimpleNamespace(ticket=ticket, symbol="EURUSD", magic=magic, type=type_, volume_current=0.1,
                           volume_initial=0.1, price_open=1.05, time_setup=server_epoch(setup_utc))


def test_source_converts_server_times_and_maps_types():
    mt5 = FakeMt5()
    setup = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    mt5.orders = [_order(1, setup), _order(2, setup, type_=5), _order(3, setup, type_=0)]   # 3 is a market order
    src = Mt5AccountSource(mt5, NY7)
    orders = src.pending_orders()
    assert [(o.ticket, o.order_type) for o in orders] == [(1, PendingOrderType.BUY_LIMIT), (2, PendingOrderType.SELL_STOP)]
    assert orders[0].setup_utc == setup                        # not shifted by the server's +3h


def test_source_deals_are_windowed_in_utc():
    mt5 = FakeMt5()
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    for i, minutes in enumerate((-90, -30, 10, 90), start=1):
        mt5.deals.append(SimpleNamespace(ticket=i, order=i, position_id=i, symbol="EURUSD", magic=7, type=2 if i == 1 else 1,
                                         entry=1, volume=0.1, price=1.1, profit=10.0, commission=-1.0, swap=0.0, fee=0.0,
                                         time=server_epoch(t0 + timedelta(minutes=minutes))))
    deals = Mt5AccountSource(mt5, NY7).deals(t0 - timedelta(hours=1), t0 + timedelta(hours=1))
    assert [d.ticket for d in deals] == [2, 3] and deals[0].kind is DealKind.SELL and deals[0].entry is DealEntry.OUT
    assert deals[0].net == 9.0


def test_cancel_outcomes():
    mt5 = FakeMt5()
    setup = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    src, actions = Mt5AccountSource(mt5, NY7), Mt5BrokerActions(mt5)
    mt5.orders = [_order(1, setup), _order(2, setup)]
    o1, o2 = src.pending_orders()
    assert actions.cancel_pending(o1).outcome is BookActionOutcome.DONE
    assert mt5.sent[-1] == {"action": 8, "order": 1, "comment": "account_admin"}
    assert actions.cancel_pending(o1).outcome is BookActionOutcome.SKIPPED_FILLED     # already gone
    mt5.remove_retcode = 10018
    result = actions.cancel_pending(o2)
    assert result.outcome is BookActionOutcome.FAILED and result.retcode == 10018


# ------------------------------------------------------------------------------------------------ run_admin


def _deploy(tmp_path, monkeypatch, config=None, env_lines=None):
    live, logs, envs = tmp_path / "live", tmp_path / "logs", tmp_path / "env"
    admin = live / "icmarkets.demo" / "_account_admin"
    admin.mkdir(parents=True)
    envs.mkdir()
    run_py = admin / "run.py"
    run_py.write_text("")
    cfg = config if config is not None else {
        "kind": "account_admin", "runner": "_account_admin",
        "broker": {"server_clock_tz": "America/New_York", "server_clock_shift_h": 7},
        "clock": {"cycle_s": 20, "jitter_s": 5, "valid_for_s": 240,
                  "blackout": [{"every_s": 60, "offset_s": 0, "length_s": 3}]},
        "requests": {"request_ttl_s": 900},
        "tasks": [{"kind": "pending_order_cleanup", "max_age_s": 3600}]}
    (admin / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    lines = env_lines if env_lines is not None else [f"TERMINAL_PATH=C:/MT5/terminal64.exe", f"LOGIN_ID={LOGIN}",
                                                     f"LOGIN_SERVER={SERVER}", "BROKER_NAME=ICMarkets"]
    (envs / ".env.icmarkets.demo").write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(live))
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(logs))
    monkeypatch.setenv("OKMICH_QUANT_ENV_DIR", str(envs))
    # deployment_account() reads the running main script; point it at the Admin's run.py
    monkeypatch.setattr("okmich_quant_core.account._main_script", lambda: run_py)
    return run_py, admin, logs


def test_stage1_refuses_a_bad_config_without_touching_the_terminal(tmp_path, monkeypatch):
    bad = {"kind": "account_admin", "runner": "_account_admin", "clock": {}, "requests": {"request_ttl_s": 900},
           "tasks": [{"kind": "pending_order_cleanup", "max_age_s": 1}]}
    run_py, *_ = _deploy(tmp_path, monkeypatch, config=bad)
    mt5 = FakeMt5()
    assert app_mod.run_admin(run_py, mt5_module=mt5) == app_mod.EXIT_STAGE1
    assert mt5.calls == []


def test_stage1_refuses_a_missing_env_key(tmp_path, monkeypatch, capsys):
    run_py, *_ = _deploy(tmp_path, monkeypatch, env_lines=[f"LOGIN_ID={LOGIN}", f"LOGIN_SERVER={SERVER}"])
    mt5 = FakeMt5()
    assert app_mod.run_admin(run_py, mt5_module=mt5) == app_mod.EXIT_STAGE1
    assert mt5.calls == [] and "TERMINAL_PATH is missing" in capsys.readouterr().out


def test_stage2_refuses_the_wrong_account_and_detaches(tmp_path, monkeypatch):
    run_py, *_ = _deploy(tmp_path, monkeypatch)
    mt5 = FakeMt5(login=LOGIN + 1)
    assert app_mod.run_admin(run_py, mt5_module=mt5) == app_mod.EXIT_STAGE2
    assert mt5.calls[0] == "initialize" and mt5.calls[-1] == "shutdown"


def test_a_full_run_publishes_outputs_status_and_heartbeat(tmp_path, monkeypatch):
    run_py, admin, logs = _deploy(tmp_path, monkeypatch)
    mt5 = FakeMt5()
    mt5.orders = [_order(1, datetime.now(UTC) - timedelta(hours=2))]
    monkeypatch.setattr(app_mod, "AdminClock", _InstantClock)
    assert app_mod.run_admin(run_py, mt5_module=mt5, max_cycles=2) == app_mod.EXIT_OK
    report = json.loads((admin / "pending_order_cleanup.json").read_text())
    assert report["cancelled_total"] == 1 and not (admin / "directive.json").exists()
    status = json.loads((logs / "icmarkets.demo" / "_account_admin" / "status.json").read_text())
    assert status["state"] == "stopped" and status["account"] == "icmarkets.demo"
    assert status["account_id"] == str(LOGIN) and status["broker_disconnected"] is True
    beats = list((logs / "icmarkets.demo" / "_account_admin" / "account" / "1" / "inference").glob("*.jsonl"))
    assert beats and "\"event\": \"bar\"" in beats[0].read_text()


class _InstantClock:
    """A clock with no waiting, for a bounded run."""

    def __init__(self, config):
        self.config = config

    def release(self, t):
        return t

    def next_start(self, previous, now):
        return now


def test_an_unanswered_query_is_a_failed_cancel_not_a_fill():
    """R5: orders_get returning None (IPC failure) must not read as 'filled or removed'."""
    mt5 = FakeMt5()
    src, actions = Mt5AccountSource(mt5, NY7), Mt5BrokerActions(mt5)
    mt5.orders = [_order(1, datetime(2026, 9, 24, 12, 0, tzinfo=UTC))]
    (o1,) = src.pending_orders()
    mt5.orders_get = lambda ticket=None: None
    result = actions.cancel_pending(o1)
    assert result.outcome is BookActionOutcome.FAILED and mt5.sent == []


def test_an_unexpected_stage2_error_detaches_releases_and_refuses(tmp_path, monkeypatch):
    run_py, admin, _ = _deploy(tmp_path, monkeypatch)
    mt5 = FakeMt5()
    monkeypatch.setattr(app_mod, "AdminHost", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_mod.run_admin(run_py, mt5_module=mt5) == app_mod.EXIT_STAGE2
    assert mt5.calls[-1] == "shutdown"
    from okmich_quant_core.account_admin import WriterLock
    WriterLock(admin / "writer.lock").acquire()                        # released: a restart can take it
