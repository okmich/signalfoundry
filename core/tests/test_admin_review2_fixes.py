"""Regressions for the second Account Admin review (2026-09-27)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from okmich_quant_core.account_admin import (AccountIdentity, AdminHost, PropGuardTask, parse_admin_config,
                                             submit_request)
from okmich_quant_core.account_admin.request import main as request_main
from okmich_quant_core.logging import GuardedOp, LogEventType

from .admin_fakes import LOGIN, SERVER, FakeAccount, admin_config, make_ctx, prop_guard_entry, t
from .test_directive_guard import Guarded, live, publish  # noqa: F401  (live is a fixture)


def _run(task, acct, when, slot, **kw):
    r = task.on_cycle(make_ctx(acct, when, slot, **kw))
    return r, r.slot


@pytest.mark.parametrize("new_hour", [6, 16])     # +13 h (10:00 UTC) and -1 h (20:00 UTC) from 17:00 New York
def test_a_boundary_edit_of_any_size_keeps_today_until_its_old_end(new_hour):
    acct = FakeAccount()
    task = PropGuardTask(prop_guard_entry())
    r, slot = _run(task, acct, t("2026-09-24 21:30"), {})               # day starts 21:00 UTC (17:00 NY)
    acct.equity = 94_000
    r, slot = _run(task, acct, t("2026-09-24 22:00"), slot)
    assert "daily_loss" in slot["latches"]
    moved = PropGuardTask(prop_guard_entry(day_start_hour=new_hour))
    acct.equity = 99_000
    for when in ("2026-09-25 10:30", "2026-09-25 20:30", "2026-09-25 20:59"):
        r, slot = _run(moved, acct, t(when), slot)
        assert "daily_loss" in slot["latches"] and r.output["directive"] == "NO_OPS", when
        assert r.output["trading_day"]["start_utc"] == "2026-09-24T21:00:00Z"
    r, slot = _run(moved, acct, t("2026-09-25 21:00:30"), slot)          # the old day's natural end
    assert "daily_loss" not in slot["latches"] and r.output["directive"] == "ALL_OPS"
    assert r.output["trading_day"]["start_utc"] == "2026-09-25T21:00:00Z"


class _HistoryDown(FakeAccount):
    def deals(self, a, b):
        raise ConnectionError("history_deals_get returned None")


def test_an_unreadable_deal_history_degrades_the_cycle_and_keeps_no_ops():
    entry = prop_guard_entry()
    entry["policy"]["conditions"]["daily_loss"] = {"directive": "NO_OPS", "latch": "none"}
    task, acct = PropGuardTask(entry), FakeAccount()
    r, slot = _run(task, acct, t("2026-09-24 13:00"), {})
    acct.equity = 94_000
    r, slot = _run(task, acct, t("2026-09-24 13:01"), slot)
    assert r.output["directive"] == "NO_OPS"
    down = _HistoryDown(equity=94_000)
    r, slot = _run(task, down, t("2026-09-24 13:02"), slot)             # the task does not raise: it degrades
    assert r.output["directive"] == "NO_OPS" and "admin_degraded" in r.output["causes"]
    assert "history unreadable" in r.output["reason"] or "daily_loss" in r.output["causes"]


def _host(tmp_path, acct):
    admin = tmp_path / "live" / "icmarkets.demo" / "_account_admin"
    return AdminHost(parse_admin_config(admin_config()), identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER),
                     admin_dir=admin, log_dir=tmp_path / "logs", live_account_dir=admin.parent, log_account_dir=None,
                     source=acct, actions=acct)


def test_a_request_with_nan_is_rejected_and_the_admin_keeps_cycling(tmp_path):
    acct = FakeAccount()
    host = _host(tmp_path, acct)
    now = datetime.now(timezone.utc)
    host.start(now)
    inbox = host.inbox.directory
    (inbox / "nan.json").write_text('{"request_id": "x", "created_utc": "%s", "operator": "o", "task": "prop_guard", '
                                    '"kind": "override", "directive": "NO_OPS", "until_utc": NaN}'
                                    % now.strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="utf-8")
    host.run_cycle(now)
    host.run_cycle(now + timedelta(seconds=20))
    assert not list(inbox.glob("*.json"))
    done = [json.loads(p.read_text()) for p in (inbox / "done").glob("*.json")]
    assert done and done[0]["outcome"] == "rejected"
    assert json.loads((host.admin_dir / "directive.json").read_text())["sequence"] == 2


def test_a_consumed_request_is_not_replayed_after_a_restart(tmp_path, monkeypatch):
    acct = FakeAccount()
    host = _host(tmp_path, acct)
    now = datetime.now(timezone.utc)
    host.start(now)
    host.run_cycle(now)
    submit_request(host.inbox.directory, operator="ops", task="prop_guard", kind="override", now=now,
                   directive="NO_ENTRY_OPS", until_utc=(now + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    monkeypatch.setattr("okmich_quant_core.account_admin.requests.os.remove",
                        lambda p: (_ for _ in ()).throw(PermissionError(32, "in use")))
    host.run_cycle(now + timedelta(seconds=1))                          # applied; the move to done\\ fails
    submit_request(host.inbox.directory, operator="ops", task="prop_guard", kind="clear_override", now=now)
    monkeypatch.undo()
    again = _host(tmp_path, acct)                                       # restart before any further cycle
    again.start(now + timedelta(seconds=2))
    again.run_cycle(now + timedelta(seconds=2))
    state = json.loads((again.admin_dir / "state.json").read_text())["tasks"]["prop_guard"]
    assert state["override"] is None                                    # cleared, not re-applied by the replay
    events = [json.loads(l)["event"] for p in again.audit.directory.glob("*.jsonl") for l in p.read_text().splitlines()]
    assert "request_rejected" in events                                 # the duplicate is audited


def test_a_position_that_closed_on_its_own_is_not_a_forced_close(live):
    from okmich_quant_core.account_admin import AccountDirective
    publish(live, AccountDirective.NO_OPS)
    s = Guarded()
    original = s._guard_close_position

    def sl_hit_first(ticket, directive):
        s.positions.pop(ticket, None)                                    # its stop-loss filled just before the sweep
        return original(ticket, directive)

    s._guard_close_position = sl_hit_first
    s.enforce_account_directive(datetime.now())
    assert not [r for r in s.records(LogEventType.ACCOUNT_DIRECTIVE_OP) if r.op is GuardedOp.CLOSE_LONG]


def test_the_request_command_refuses_an_until_without_a_zone(tmp_path):
    (tmp_path / "icmarkets.demo" / "_account_admin").mkdir(parents=True)
    with pytest.raises(SystemExit, match="--until"):
        request_main(["icmarkets.demo", "--live-base", str(tmp_path), "override", "--directive", "NO_OPS",
                      "--until", "2026-09-27T19:00:00"])
