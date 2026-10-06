"""ACCOUNT_ADMIN_SPEC §10.2 reading table (§14.1) and the atomic writer (§14.2)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import timedelta

import pytest

from okmich_quant_core.account_admin import (AccountDirective, DirectiveAccount, DirectiveFile, DirectiveSource,
                                             directive_path, read_directive, write_directive)
from okmich_quant_core.account_admin import atomic as atomic_mod

from .admin_fakes import LOGIN, SERVER, t

NOW = t("2026-09-24 13:15:22")
ACCOUNT = "icmarkets.demo"


def _file(directive=AccountDirective.NO_ENTRY_OPS, heartbeat=NOW, valid_for_s=240, login=LOGIN, server=SERVER):
    return DirectiveFile(account=DirectiveAccount("icmarkets", server, login, "USD"), directive=directive,
                         reason="daily_warn: test", causes=("daily_warn",), since_utc=heartbeat - timedelta(minutes=5),
                         heartbeat_utc=heartbeat, valid_for_s=valid_for_s, sequence=48211, episode=311,
                         evidence={"levels": {"daily_floor": 97_500.0}})


def _read(base, account=ACCOUNT, login=LOGIN, server=SERVER, now=NOW):
    return read_directive(account=account, login=login, server=server, live_base=base, now=now)


def test_round_trip_keeps_every_field(tmp_path):
    path = directive_path(tmp_path, ACCOUNT)
    write_directive(path, _file())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1 and payload["heartbeat_utc"] == "2026-09-24T13:15:22Z"
    parsed = DirectiveFile.from_dict(payload)
    assert parsed.directive is AccountDirective.NO_ENTRY_OPS and parsed.sequence == 48211
    assert parsed.evidence["levels"] == {"daily_floor": 97_500.0}


def test_absent_file_is_ungoverned(tmp_path):
    r = _read(tmp_path)
    assert (r.directive, r.source) == (AccountDirective.ALL_OPS, DirectiveSource.ABSENT)


def test_live_base_unset_is_absent(monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_LIVE_BASE", raising=False)
    r = read_directive(account=ACCOUNT, login=LOGIN, server=SERVER, now=NOW)
    assert (r.directive, r.source) == (AccountDirective.ALL_OPS, DirectiveSource.ABSENT)


@pytest.mark.parametrize("directive", list(AccountDirective))
def test_fresh_file_applies_its_directive(tmp_path, directive):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(directive))
    r = _read(tmp_path)
    assert (r.directive, r.source) == (directive, DirectiveSource.FILE)


def test_stale_file_falls_back_to_no_entry(tmp_path):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.ALL_OPS, heartbeat=NOW - timedelta(seconds=241)))
    r = _read(tmp_path)
    assert (r.directive, r.source) == (AccountDirective.NO_ENTRY_OPS, DirectiveSource.STALE)


def test_stale_no_ops_still_falls_back_to_no_entry(tmp_path):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.NO_OPS, heartbeat=NOW - timedelta(hours=1)))
    assert _read(tmp_path).directive is AccountDirective.NO_ENTRY_OPS


@pytest.mark.parametrize("content", ["{not json", "", "[]"])
def test_corrupt_file_is_invalid(tmp_path, content):
    path = directive_path(tmp_path, ACCOUNT)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    r = _read(tmp_path)
    assert (r.directive, r.source) == (AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("directive", "HALF_OPS"), ("valid_for_s", 0),
                                         ("heartbeat_utc", "2026-09-24 13:15:22")])
def test_unknown_or_malformed_fields_are_invalid(tmp_path, field, value):
    payload = _file().to_dict()
    payload[field] = value
    path = directive_path(tmp_path, ACCOUNT)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _read(tmp_path).source is DirectiveSource.INVALID


def test_future_heartbeat_is_invalid(tmp_path):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.ALL_OPS, heartbeat=NOW + timedelta(seconds=6)))
    assert _read(tmp_path).source is DirectiveSource.INVALID
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.ALL_OPS, heartbeat=NOW + timedelta(seconds=4)))
    assert _read(tmp_path).source is DirectiveSource.FILE


@pytest.mark.parametrize("login,server", [(LOGIN + 1, SERVER), (LOGIN, "ICMarketsSC-Live"), (None, SERVER)])
def test_wrong_account_is_no_entry(tmp_path, login, server):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.ALL_OPS))
    r = _read(tmp_path, login=login, server=server)
    assert (r.directive, r.source) == (AccountDirective.NO_ENTRY_OPS, DirectiveSource.WRONG_ACCOUNT)


def test_outside_an_account_folder_finds_its_terminals_directive(tmp_path):
    write_directive(directive_path(tmp_path, "icmarkets.demo"), _file(AccountDirective.NO_OPS))
    write_directive(directive_path(tmp_path, "fxify.demo"), _file(AccountDirective.ALL_OPS, login=999, server="FXIFY"))
    r = _read(tmp_path, account=None)
    assert (r.directive, r.source) == (AccountDirective.NO_OPS, DirectiveSource.FILE)
    assert r.path == directive_path(tmp_path, "icmarkets.demo")


def test_outside_an_account_folder_with_no_match_is_absent(tmp_path):
    write_directive(directive_path(tmp_path, "fxify.demo"), _file(login=999, server="FXIFY"))
    r = _read(tmp_path, account=None)
    assert (r.directive, r.source) == (AccountDirective.ALL_OPS, DirectiveSource.ABSENT)


def test_outside_an_account_folder_with_two_matches_is_invalid(tmp_path):
    write_directive(directive_path(tmp_path, "icmarkets.demo"), _file())
    write_directive(directive_path(tmp_path, "icmarkets.demo2"), _file())
    assert _read(tmp_path, account=None).source is DirectiveSource.INVALID


def test_outside_an_account_folder_an_unreadable_file_is_not_trade_freely(tmp_path):
    path = directive_path(tmp_path, "icmarkets.demo")
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    r = _read(tmp_path, account=None)
    assert (r.directive, r.source) == (AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID)


def test_a_transient_read_failure_is_retried_before_falling_back(tmp_path, monkeypatch):
    write_directive(directive_path(tmp_path, ACCOUNT), _file(AccountDirective.ALL_OPS))
    real_open = open
    calls = {"n": 0}

    def flaky_open(path, *args, **kwargs):
        if str(path).endswith("directive.json") and calls["n"] < 2:
            calls["n"] += 1
            raise PermissionError(32, "sharing violation")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)
    r = _read(tmp_path)
    assert (r.directive, r.source) == (AccountDirective.ALL_OPS, DirectiveSource.FILE)
    assert calls["n"] == 2


def test_a_read_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("okmich_quant_core.account_admin.directive._read_account",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = _read(tmp_path)
    assert (r.directive, r.source) == (AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID)


def test_writer_retries_a_sharing_violation(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    real_replace = atomic_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_mod.os, "replace", flaky_replace)
    atomic_mod.atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1} and calls["n"] == 3


def test_writer_gives_up_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic_mod.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError(5, "denied")))
    with pytest.raises(PermissionError):
        atomic_mod.atomic_write_json(tmp_path / "x.json", {"a": 1}, attempts=2, backoff_s=0)
    assert not list(tmp_path.glob("*.tmp"))


_WRITER = textwrap.dedent("""
    import sys, time
    from datetime import datetime, timezone, timedelta
    from okmich_quant_core.account_admin import (AccountDirective, DirectiveAccount, DirectiveFile, write_directive)
    path, seconds = sys.argv[1], float(sys.argv[2])
    end = time.time() + seconds
    seq = failures = 0
    while time.time() < end:
        seq += 1
        now = datetime.now(timezone.utc)
        f = DirectiveFile(DirectiveAccount("b", "S", 1, "USD"), AccountDirective.NO_ENTRY_OPS, "r" * (seq % 500), ("x",),
                          now, now, 240, seq, 1, {"levels": {"pad": "y" * 4000}})
        try:
            write_directive(path, f)
        except OSError:
            failures += 1
    print(seq, failures)
""")

_READER = textwrap.dedent("""
    import sys, time
    from datetime import datetime, timezone
    from okmich_quant_core.account_admin import read_directive, DirectiveSource
    base, seconds = sys.argv[1], float(sys.argv[2])
    end = time.time() + seconds
    reads = invalid = 0
    while time.time() < end:
        r = read_directive(account="acc.demo", login=1, server="S", live_base=base)
        reads += 1
        if r.source is DirectiveSource.INVALID:
            invalid += 1
        time.sleep(0.002)   # still hundreds of reads a second per reader; production reads once per position check
    print(reads, invalid)
""")


def test_concurrent_writer_and_readers_never_see_a_partial_file(tmp_path):
    """§14.2: a writer rewrites the file in a tight loop while readers in other processes read."""
    path = directive_path(tmp_path, "acc.demo")
    now = NOW
    write_directive(path, DirectiveFile(DirectiveAccount("b", "S", 1, "USD"), AccountDirective.NO_ENTRY_OPS, "r", ("x",),
                                        now, now, 240, 0, 1))
    seconds = "2.0"
    writer = subprocess.Popen([sys.executable, "-c", _WRITER, str(path), seconds], stdout=subprocess.PIPE, text=True)
    readers = [subprocess.Popen([sys.executable, "-c", _READER, str(tmp_path), seconds], stdout=subprocess.PIPE, text=True)
               for _ in range(3)]
    writes, write_failures = map(int, writer.communicate(timeout=60)[0].split())
    results = [tuple(map(int, r.communicate(timeout=60)[0].split())) for r in readers]
    total_reads = sum(r[0] for r in results)
    fell_back = sum(r[1] for r in results)
    print(f"writes={writes} write_failures={write_failures} reads={total_reads} fell_back={fell_back}")
    assert writes >= 10 and total_reads > 100
    assert fell_back == 0, f"{fell_back} reads fell back to invalid after retries"
    assert write_failures == 0, f"{write_failures} writes gave up after their sharing-violation retries"
