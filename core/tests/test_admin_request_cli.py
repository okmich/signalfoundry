"""The operator request command (ACCOUNT_ADMIN_SPEC §8.2) and its round trip through the host."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from okmich_quant_core.account_admin import AccountIdentity, AdminHost, parse_admin_config
from okmich_quant_core.account_admin.request import main

from .admin_fakes import LOGIN, SERVER, FakeAccount, admin_config, t


def test_override_request_round_trip(tmp_path, capsys):
    live = tmp_path / "live"
    admin = live / "icmarkets.demo" / "_account_admin"
    host = AdminHost(parse_admin_config(admin_config()), identity=AccountIdentity("icmarkets.demo", "icmarkets", LOGIN, SERVER),
                     admin_dir=admin, log_dir=tmp_path / "logs", live_account_dir=admin.parent, log_account_dir=None,
                     source=FakeAccount(), actions=FakeAccount())
    now = t("2026-09-24 13:00")
    host.start(now)
    host.run_cycle(now)
    assert main(["icmarkets.demo", "--live-base", str(live), "--operator", "ops", "override", "--directive",
                 "NO_ENTRY_OPS", "--for", "2h", "--reason", "FOMC"]) == 0
    request = json.loads(next((admin / "requests").glob("*.json")).read_text())
    assert request["task"] == "prop_guard" and request["kind"] == "override" and request["operator"] == "ops"

    from datetime import datetime, timezone
    host.run_cycle(datetime.now(timezone.utc))
    directive = json.loads((admin / "directive.json").read_text())
    assert directive["directive"] == "NO_ENTRY_OPS" and directive["override"]["operator"] == "ops"
    main(["icmarkets.demo", "--live-base", str(live), "show"])
    assert "applied" in capsys.readouterr().out


def test_the_command_refuses_an_unknown_account_or_directive(tmp_path):
    with pytest.raises(SystemExit):
        main(["icmarkets.demo", "--live-base", str(tmp_path), "clear-override"])      # no Admin deployed there
    with pytest.raises(SystemExit):
        main(["icmarkets.demo", "--live-base", str(tmp_path), "override", "--directive", "ALL_OPS", "--for", "1h"])
