"""Regressions for the Account Admin review findings (2026-09-27)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from okmich_quant_core.account_admin import (AccountDirective, AccountIdentity, AdminHost, AdminRequest, AdminTaskKind,
                                             DirectiveAccount, DirectiveFile, PropGuardTask, RequestOutcome,
                                             directive_path, parse_admin_config, submit_request, write_directive)
from okmich_quant_core.account_admin.tasks.base import TASK_REGISTRY

from .admin_fakes import LOGIN, SERVER, FakeAccount, admin_config, make_ctx, prop_guard_entry, t
from .test_directive_guard import Guarded, live, publish  # noqa: F401  (live is a fixture)


def _run(task, acct, when, slot, **kw):
    r = task.on_cycle(make_ctx(acct, when, slot, **kw))
    return r, r.slot


def test_a_degraded_cycle_keeps_an_unlatched_no_ops_breach():
    """R1: daily_loss NO_OPS with latch none must not loosen to NO_ENTRY_OPS while the Admin is blind."""
    entry = prop_guard_entry()
    entry["policy"]["conditions"]["daily_loss"] = {"directive": "NO_OPS", "latch": "none"}
    task, acct = PropGuardTask(entry), FakeAccount()
    r, slot = _run(task, acct, t("2026-09-24 13:00"), {})
    acct.equity = 94_000
    r, slot = _run(task, acct, t("2026-09-24 13:01"), slot)
    assert r.output["directive"] == "NO_OPS"
    r, slot = _run(task, acct, t("2026-09-24 13:02"), slot, degraded=True)
    assert r.output["directive"] == "NO_OPS" and "daily_loss" in r.output["causes"]
    assert r.output["episode"] == 2                                # no episode churn


def test_a_policy_change_of_the_day_boundary_keeps_todays_latch_and_base():
    """R2: moving day_start_hour mid-day neither re-bases the day nor clears a trading_day latch (spec §8.1)."""
    acct = FakeAccount()
    task = PropGuardTask(prop_guard_entry())
    r, slot = _run(task, acct, t("2026-09-24 13:00"), {})
    acct.equity = 94_000
    r, slot = _run(task, acct, t("2026-09-24 22:30"), slot)        # after the 21:00 boundary: day base 100k, loss
    assert "daily_loss" in slot["latches"]
    moved = PropGuardTask(prop_guard_entry(day_start_hour=18))      # restart with the boundary at 22:00 UTC
    acct.equity = 99_000
    r, slot = _run(moved, acct, t("2026-09-24 22:31"), slot)
    assert "daily_loss" in slot["latches"] and r.output["directive"] == "NO_OPS"
    assert r.output["trading_day"]["start_utc"] == "2026-09-24T21:00:00Z"
    r, slot = _run(moved, acct, t("2026-09-25 22:00:30"), slot)    # the next genuine boundary, new policy
    assert r.output["trading_day"]["start_utc"] == "2026-09-25T22:00:00Z" and "daily_loss" not in slot["latches"]


def test_a_reset_is_refused_while_degraded():
    task, acct = PropGuardTask(prop_guard_entry()), FakeAccount()
    r, slot = _run(task, acct, t("2026-09-24 13:00"), {})
    acct.equity = 89_000
    r, slot = _run(task, acct, t("2026-09-24 13:01"), slot)
    req = AdminRequest("r", t("2026-09-24 13:02"), "ops", "prop_guard", "reset_latch", {"target": "max_loss"})
    r, slot = _run(task, acct, t("2026-09-24 13:02"), slot, degraded=True, requests=[req])
    assert r.request_outcomes["r"][0] is RequestOutcome.REJECTED and "max_loss" in slot["latches"]
    assert r.output["directive"] == "NO_OPS"


class _FailsOnce:
    """Wraps an account whose deal history fails on the first read only."""

    def __init__(self, acct):
        self.acct, self.calls = acct, 0

    def __getattr__(self, name):
        return getattr(self.acct, name)

    def deals(self, a, b):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("history timeout")
        return self.acct.deals(a, b)


def test_state_lost_still_trips_when_the_first_cycle_after_the_loss_fails(tmp_path):
    admin = tmp_path / "live" / "icmarkets.demo" / "_account_admin"
    now = t("2026-09-24 13:00")
    write_directive(admin / "directive.json", DirectiveFile(DirectiveAccount("icmarkets", SERVER, LOGIN, "USD"),
                                                            AccountDirective.ALL_OPS, "x", (), now, now, 240, 90, 9))
    source = _FailsOnce(FakeAccount())
    host = AdminHost(parse_admin_config(admin_config()), identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER),
                     admin_dir=admin, log_dir=tmp_path / "logs", live_account_dir=admin.parent, log_account_dir=None,
                     source=source, actions=source)
    host.start(now)
    assert host.run_cycle(now).failed_tasks == ["prop_guard"]
    host.run_cycle(now + timedelta(seconds=20))
    d = json.loads((admin / "directive.json").read_text())
    assert d["causes"] == ["state_lost"] and d["directive"] == "NO_ENTRY_OPS"


def test_a_request_is_never_applied_twice_when_its_move_to_done_fails(tmp_path, monkeypatch):
    admin = tmp_path / "live" / "icmarkets.demo" / "_account_admin"
    acct = FakeAccount()
    host = AdminHost(parse_admin_config(admin_config()), identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER),
                     admin_dir=admin, log_dir=tmp_path / "logs", live_account_dir=admin.parent, log_account_dir=None,
                     source=acct, actions=acct)
    now = datetime.now(timezone.utc)
    host.start(now)
    host.run_cycle(now)
    submit_request(admin / "requests", operator="ops", task="prop_guard", kind="override", now=now,
                   directive="NO_ENTRY_OPS", until_utc=(now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    real_remove = __import__("os").remove
    monkeypatch.setattr("okmich_quant_core.account_admin.requests.os.remove",
                        lambda p: (_ for _ in ()).throw(PermissionError(32, "in use")))
    host.run_cycle(now + timedelta(seconds=1))
    applied_once = json.loads((admin / "state.json").read_text())["tasks"]["prop_guard"]["override"]
    submit_request(admin / "requests", operator="ops", task="prop_guard", kind="clear_override", now=now)
    monkeypatch.setattr("okmich_quant_core.account_admin.requests.os.remove", real_remove)
    host.run_cycle(now + timedelta(seconds=2))                     # clear applies; the stuck override is not replayed
    state = json.loads((admin / "state.json").read_text())["tasks"]["prop_guard"]
    assert applied_once is not None and state["override"] is None
    assert not list((admin / "requests").glob("*.json"))


def test_an_unknown_terminal_identity_suppresses_entries_but_destroys_nothing(live, monkeypatch):
    """R3: an IPC hiccup is not evidence of a wrong account: no forced cancel or close."""
    from okmich_quant_core import directive_guard
    monkeypatch.setattr(directive_guard._ProcessReader, "account", lambda self: "icmarkets.demo")
    monkeypatch.setattr(directive_guard, "IDENTITY_RETRY_S", 0.0)
    publish(live, AccountDirective.ALL_OPS)
    s = Guarded()
    s._guard_terminal_identity = lambda: None
    assert s.open() is False                                       # entries suppressed
    s.enforce_account_directive(datetime.now())
    assert s.pending == [101] and s.positions == {201: True}       # nothing cancelled or closed
