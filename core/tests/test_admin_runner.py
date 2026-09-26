"""The Admin as a Supervisor runner (ACCOUNT_ADMIN_SPEC §7.4, §7.6, §9): status file, heartbeat, stop, deployment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from okmich_quant_core.account_admin import (AccountIdentity, AdminClock, AdminConfigError, AdminHost, AdminRunLoop,
                                             WriterLock, WriterLockHeld, parse_admin_config, resolve_deployment)

from .admin_fakes import LOGIN, SERVER, FakeAccount, admin_config


class SimTime:
    def __init__(self, start: float):
        self.t = start

    def time(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeSession:
    broker, account_id, broker_session_id = "ICMarketsSC-Demo", str(LOGIN), "terminal:build5000"

    def __init__(self):
        self.disconnected = False

    def disconnect(self) -> bool:
        self.disconnected = True
        return True


def _loop(tmp_path, acct, stop_after=None):
    cfg = parse_admin_config(admin_config())
    admin_dir = tmp_path / "live" / "icmarkets.demo" / "_account_admin"
    host = AdminHost(cfg, identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER), admin_dir=admin_dir,
                     log_dir=tmp_path / "logs" / "_account_admin", live_account_dir=admin_dir.parent,
                     log_account_dir=tmp_path / "logs", source=acct, actions=acct)
    lock = WriterLock(admin_dir / "writer.lock")
    lock.acquire()
    sim = SimTime(1_790_000_007.0)
    session = FakeSession()
    loop = AdminRunLoop(host, AdminClock(cfg.clock), broker_session=session, writer_lock=lock,
                        log_base=tmp_path / "logs", time_fn=sim.time, sleep_fn=sim.sleep)
    return loop, lock, session, sim


def test_runner_writes_status_and_a_per_minute_heartbeat_where_the_supervisor_reads(tmp_path):
    loop, lock, session, sim = _loop(tmp_path, FakeAccount())
    reason = loop.run(max_cycles=12)                                   # ~4 minutes of 20 s cycles
    assert reason == "max_cycles"
    status = json.loads((tmp_path / "logs" / "_account_admin" / "status.json").read_text())
    assert status["state"] == "stopped" and status["broker_disconnected"] is True and status["clean"] is True
    assert status["logical_systems"] == [{"logical_system_id": "_account_admin/account/1", "symbol": "account",
                                          "timeframe": 1}]
    inference = tmp_path / "logs" / "_account_admin" / "account" / "1" / "inference"
    bars = [json.loads(l) for p in inference.glob("*.jsonl") for l in p.read_text().splitlines()]
    assert len(bars) >= 3 and all(b["event"] == "bar" and b["outcome"] == "ok" for b in bars)
    minutes = [b["asof_bar_ts"] for b in bars]
    assert minutes == sorted(set(minutes))                             # one per minute, advancing
    assert all(b["bar_close"] is None for b in bars)                   # no money on the inference channel
    assert session.disconnected and not lock.held


def test_a_degraded_admin_heartbeats_error(tmp_path):
    loop, *_ = _loop(tmp_path, FakeAccount(info_none=True))
    loop.run(max_cycles=4)
    inference = tmp_path / "logs" / "_account_admin" / "account" / "1" / "inference"
    bars = [json.loads(l) for p in inference.glob("*.jsonl") for l in p.read_text().splitlines()]
    assert bars and all(b["outcome"] == "error" for b in bars)


def test_stop_request_finishes_the_cycle_and_stops_cleanly(tmp_path):
    loop, lock, session, sim = _loop(tmp_path, FakeAccount())
    original = loop.host.run_cycle

    def cycle_then_stop(now):
        report = original(now)
        loop.request_stop()                                            # as a Ctrl+C would, mid-run
        return report

    loop.host.run_cycle = cycle_then_stop
    assert loop.run() == "operator_stop"
    assert (loop.host.admin_dir / "directive.json").exists()           # the directive stays for the systems


def test_writer_lock_refuses_a_second_admin(tmp_path):
    path = tmp_path / "writer.lock"
    with WriterLock(path):
        with pytest.raises(WriterLockHeld):
            WriterLock(path).acquire()
    WriterLock(path).acquire()                                         # released: a restart may take it


def test_resolve_deployment(tmp_path, monkeypatch):
    live = tmp_path / "live"
    run_py = live / "icmarkets.demo" / "_account_admin" / "run.py"
    run_py.parent.mkdir(parents=True)
    run_py.write_text("")
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(live))
    dep = resolve_deployment(run_py)
    assert dep.account == "icmarkets.demo" and dep.admin_dir == run_py.parent.resolve()

    flat = live / "_account_admin" / "run.py"
    flat.parent.mkdir(parents=True)
    with pytest.raises(AdminConfigError, match="not deployed in an account folder"):
        resolve_deployment(flat)
    wrong = live / "icmarkets.demo" / "admin" / "run.py"
    wrong.parent.mkdir(parents=True)
    with pytest.raises(AdminConfigError, match="must sit at"):
        resolve_deployment(wrong)
    monkeypatch.delenv("OKMICH_QUANT_LIVE_BASE")
    with pytest.raises(AdminConfigError, match="OKMICH_QUANT_LIVE_BASE"):
        resolve_deployment(run_py)
