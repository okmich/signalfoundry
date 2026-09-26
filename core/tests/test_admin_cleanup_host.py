"""``pending_order_cleanup`` (ACCOUNT_ADMIN_SPEC §12.3, §14.14) and the host (§7, §8.2, §12.1; §14.6, §14.9, §14.13)."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from okmich_quant_core.account_admin import (AccountDirective, AccountIdentity, AdminHost, BookActionOutcome,
                                             PendingOrderCleanupTask, PendingOrderType, parse_admin_config, submit_request)
from okmich_quant_core.account_admin.owners import MagicOwner
from okmich_quant_core.account_admin import AdminTaskKind
from okmich_quant_core.account_admin.tasks.base import TASK_REGISTRY, AdminTask, TaskResult

from .admin_fakes import (LOGIN, SERVER, FakeAccount, RecordingNotifier, RecordingPort, admin_config, make_ctx, order,
                          prop_guard_entry, t)

NOW = t("2026-09-24 13:00")


def cleanup(**entry):
    return PendingOrderCleanupTask({"kind": "pending_order_cleanup", "max_age_s": 3600, **entry})


# ------------------------------------------------------------------------------------------------ cleanup


def test_age_threshold_is_inclusive_and_measured_in_utc():
    acct = FakeAccount(orders=[order(1, NOW - timedelta(seconds=3599)), order(2, NOW - timedelta(seconds=3600))])
    result = cleanup().on_cycle(make_ctx(acct, NOW))
    assert acct.cancelled == [2]
    assert [c["ticket"] for c in result.output["cancelled_this_cycle"]] == [2]


def test_an_order_with_unknown_age_is_never_cancelled_and_alerted_once():
    acct = FakeAccount(orders=[order(1, None)])
    task = cleanup()
    r1 = task.on_cycle(make_ctx(acct, NOW))
    r2 = task.on_cycle(make_ctx(acct, NOW + timedelta(minutes=1), r1.slot))
    assert acct.cancelled == [] and [a.title for a in r1.alerts] == ["ORDER AGE UNKNOWN"] and not r2.alerts


def test_every_pending_type_is_covered():
    old = NOW - timedelta(hours=2)
    acct = FakeAccount(orders=[order(i, old, kind=k) for i, k in enumerate(PendingOrderType, start=1)])
    cleanup().on_cycle(make_ctx(acct, NOW))
    assert sorted(acct.cancelled) == list(range(1, len(PendingOrderType) + 1))


def test_scopes():
    old = NOW - timedelta(hours=2)
    owners = {7: MagicOwner(7, "a/live-multi", "live-multi", True), 8: MagicOwner(8, "a/stopped-multi", "stopped-multi", False)}

    def run(task):
        acct = FakeAccount(orders=[order(1, old, magic=0), order(2, old, magic=7), order(3, old, magic=8),
                                   order(4, old, magic=99)])
        task.on_cycle(make_ctx(acct, NOW, owners=owners))
        return sorted(acct.cancelled)

    assert run(cleanup()) == [1, 2, 3, 4]                                  # all: manual included
    assert run(cleanup(scope="orphaned_only")) == [1, 3, 4]               # spares the running system's order
    assert run(cleanup(scope="magics", magics=[8, 99])) == [3, 4]


def test_a_filled_order_is_skipped_and_a_failing_cancel_is_alerted_once():
    old = NOW - timedelta(hours=2)
    acct = FakeAccount(orders=[order(1, old), order(2, old)],
                       cancel_outcomes={1: BookActionOutcome.SKIPPED_FILLED, 2: BookActionOutcome.FAILED})
    task = cleanup()
    r1 = task.on_cycle(make_ctx(acct, NOW))
    assert [s["ticket"] for s in r1.output["skipped_filled_this_cycle"]] == [1]
    assert list(r1.output["failing"]) == ["2"] and [a.title for a in r1.alerts] == ["CANCEL FAILING"]
    r2 = task.on_cycle(make_ctx(acct, NOW + timedelta(seconds=30), r1.slot))
    assert not r2.alerts and "2" in r2.output["failing"]
    acct.orders = []                                                       # it left the book
    r3 = task.on_cycle(make_ctx(acct, NOW + timedelta(seconds=60), r2.slot))
    assert r3.output["failing"] == {}


def test_a_degraded_cycle_never_acts():
    acct = FakeAccount(orders=[order(1, NOW - timedelta(hours=2))])
    port = RecordingPort(acct, degraded=True)
    result = cleanup().on_cycle(make_ctx(acct, NOW, degraded=True, port=port))
    assert port.calls == [] and result.output["degraded"] is True


@pytest.mark.parametrize("entry,needle", [({"max_age_s": 59}, "max_age_s"), ({"scope": "magics"}, "magics"),
                                          ({"magics": [1]}, "only allowed"), ({"scope": "some"}, "unknown value"),
                                          ({"scope": "magics", "magics": [1, 1]}, "repeat")])
def test_cleanup_validation(entry, needle):
    problems = PendingOrderCleanupTask.validate({"kind": "pending_order_cleanup", "max_age_s": 3600, **entry}, "t")
    assert any(needle in p for p in problems), problems


# ------------------------------------------------------------------------------------------------ host


def make_host(tmp_path, acct, tasks=None, notifier=None, registry=None):
    cfg = parse_admin_config(admin_config(tasks), registry)
    admin_dir = tmp_path / "live" / "icmarkets.demo" / "_account_admin"
    return AdminHost(cfg, identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER), admin_dir=admin_dir,
                     log_dir=tmp_path / "logs" / "icmarkets.demo" / "_account_admin",
                     live_account_dir=tmp_path / "live" / "icmarkets.demo", log_account_dir=tmp_path / "logs" / "icmarkets.demo",
                     source=acct, actions=acct, notifier=notifier)


def directive_of(host):
    return json.loads((host.admin_dir / "directive.json").read_text(encoding="utf-8"))


def audit_events(host):
    lines = []
    for path in sorted(host.audit.directory.glob("*.jsonl")):
        lines += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lines


def test_cycle_publishes_every_task_output_and_state(tmp_path):
    acct = FakeAccount(orders=[order(1, NOW - timedelta(hours=2))])
    notifier = RecordingNotifier()
    host = make_host(tmp_path, acct, [prop_guard_entry(), {"kind": "pending_order_cleanup", "max_age_s": 3600}], notifier)
    host.start(NOW)
    report = host.run_cycle(NOW)
    assert not report.degraded and report.directive == "ALL_OPS"
    assert directive_of(host)["directive"] == "ALL_OPS"
    assert json.loads((host.admin_dir / "pending_order_cleanup.json").read_text())["cancelled_this_cycle"][0]["ticket"] == 1
    state = json.loads((host.admin_dir / "state.json").read_text())
    assert set(state["tasks"]) == {"prop_guard", "pending_order_cleanup", "_host"}
    events = [e["event"] for e in audit_events(host)]
    assert events[0] == "admin_started" and "book_action" in events and events[-1] == "cycle"
    assert any(title == "BOOK ACTIONS: pending_order_cleanup" for title in notifier.titles())


def test_no_prop_guard_means_no_directive(tmp_path):
    host = make_host(tmp_path, FakeAccount(), [{"kind": "pending_order_cleanup", "max_age_s": 3600}])
    host.start(NOW)
    host.run_cycle(NOW)
    assert not (host.admin_dir / "directive.json").exists()
    assert (host.admin_dir / "pending_order_cleanup.json").exists()


def test_wrong_account_degrades_the_cycle_and_the_port_refuses(tmp_path):
    acct = FakeAccount(login=LOGIN + 1, orders=[order(1, NOW - timedelta(hours=2))])
    host = make_host(tmp_path, acct, [prop_guard_entry(), {"kind": "pending_order_cleanup", "max_age_s": 3600}])
    host.start(NOW)
    report = host.run_cycle(NOW)
    assert report.degraded and "not the governed" in report.degraded_reason
    assert directive_of(host)["directive"] == "NO_ENTRY_OPS" and acct.cancelled == []
    assert directive_of(host)["account"]["login"] == LOGIN          # the directive names the governed account


class _Exploding(AdminTask):
    """Stands in for pending_order_cleanup in a private registry; raises on its second cycle."""

    kind = AdminTaskKind.PENDING_ORDER_CLEANUP
    calls = 0

    @classmethod
    def validate(cls, entry, prefix):
        return []

    def on_cycle(self, ctx):
        _Exploding.calls += 1
        if _Exploding.calls == 2:
            raise RuntimeError("boom")
        return TaskResult(slot={"n": _Exploding.calls}, output={"n": _Exploding.calls})


def test_a_failing_task_does_not_stop_the_others(tmp_path):
    registry = {**TASK_REGISTRY, AdminTaskKind.PENDING_ORDER_CLEANUP: _Exploding}
    notifier = RecordingNotifier()
    host = make_host(tmp_path, FakeAccount(), [prop_guard_entry(), {"kind": "pending_order_cleanup"}], notifier, registry)
    host.start(NOW)
    host.run_cycle(NOW)
    report = host.run_cycle(NOW + timedelta(seconds=20))
    assert report.failed_tasks == ["pending_order_cleanup"] and report.directive == "ALL_OPS"
    assert json.loads((host.admin_dir / "pending_order_cleanup.json").read_text()) == {"n": 1}   # last output kept
    assert directive_of(host)["sequence"] == 2                                                  # prop_guard ran
    host.run_cycle(NOW + timedelta(seconds=40))
    assert notifier.titles().count("TASK FAILED: pending_order_cleanup") == 1
    assert "task_failed" in [e["event"] for e in audit_events(host)]


def test_restart_reproduces_the_state_and_deleting_state_trips_state_lost(tmp_path):
    acct = FakeAccount()
    host = make_host(tmp_path, acct)
    host.start(NOW)
    host.run_cycle(NOW)
    acct.equity = 94_000
    host.run_cycle(NOW + timedelta(seconds=20))                     # NO_OPS, daily_loss latched
    before = directive_of(host)

    acct.equity = 99_000
    again = make_host(tmp_path, acct)                                # restart, same folders
    again.start(NOW + timedelta(minutes=2))
    again.run_cycle(NOW + timedelta(minutes=2))
    after = directive_of(again)
    assert after["directive"] == "NO_OPS" and after["episode"] == before["episode"]
    assert after["sequence"] == before["sequence"] + 1 and after["trading_day"] == before["trading_day"]

    (again.admin_dir / "state.json").unlink()
    third = make_host(tmp_path, acct)
    third.start(NOW + timedelta(minutes=3))
    third.run_cycle(NOW + timedelta(minutes=3))
    lost = directive_of(third)
    assert lost["directive"] == "NO_ENTRY_OPS" and lost["causes"] == ["state_lost"]
    assert lost["sequence"] == after["sequence"] + 1


def test_requests_are_routed_answered_and_archived(tmp_path):
    host = make_host(tmp_path, FakeAccount(), [prop_guard_entry(), {"kind": "pending_order_cleanup", "max_age_s": 3600}])
    host.start(NOW)
    inbox = host.inbox.directory
    submit_request(inbox, operator="ops", task="prop_guard", kind="override", now=NOW, directive="NO_ENTRY_OPS",
                   until_utc="2026-09-24T15:00:00Z", reason="FOMC")
    submit_request(inbox, operator="ops", task="prop_guard", kind="override", now=NOW - timedelta(hours=1),
                   directive="NO_OPS", until_utc="2026-09-24T15:00:00Z")                         # too old
    submit_request(inbox, operator="ops", task="coffee_maker", kind="brew", now=NOW)               # not configured
    submit_request(inbox, operator="ops", task="pending_order_cleanup", kind="purge", now=NOW)     # takes no requests
    (inbox / "garbage.json").write_text("{", encoding="utf-8")                                     # malformed
    host.run_cycle(NOW + timedelta(seconds=10))
    assert directive_of(host)["directive"] == "NO_ENTRY_OPS"
    assert not list(inbox.glob("*.json"))
    done = {p.name: json.loads(p.read_text()) for p in (inbox / "done").glob("*.json")}
    outcomes = sorted(d["outcome"] for d in done.values())
    assert outcomes == ["applied", "rejected", "rejected", "rejected", "rejected"]
    reasons = " | ".join(d["reason"] for d in done.values())
    for needle in ("request_ttl_s", "not configured", "takes no requests", "malformed"):
        assert needle in reasons


def test_config_validation_happens_before_the_account_is_touched(tmp_path):
    """§14.13: a bad config never reaches the terminal. Validation is a pure function of the config; the host is only
    built from a validated one."""
    acct = FakeAccount()
    with pytest.raises(Exception):
        make_host(tmp_path, acct, [{"kind": "pending_order_cleanup", "max_age_s": 1}])
    assert acct.touched == 0
