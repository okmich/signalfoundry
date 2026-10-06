"""The operator's Account Admin command (ACCOUNT_ADMIN_SPEC §8.2).

It writes ONE request file atomically into ``<live_base>\\<account>\\_account_admin\\requests\\`` and never touches the
directive, a task's output or the state file. The Admin consumes the request at its next cycle and moves it to
``done\\`` with its outcome. ``show`` prints what the Admin currently publishes.

Examples::

    python -m okmich_quant_core.account_admin.request icmarkets.demo reset-latch --target max_loss
    python -m okmich_quant_core.account_admin.request icmarkets.demo override --directive NO_ENTRY_OPS --for 2h --reason FOMC
    python -m okmich_quant_core.account_admin.request icmarkets.demo clear-override
    python -m okmich_quant_core.account_admin.request icmarkets.demo show
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

from ..account import LIVE_BASE_ENV_VAR, is_account
from .directive import ADMIN_FOLDER, DIRECTIVE_FILE
from .enums import AccountDirective, AdminTaskKind, RequestKind
from .requests import DONE_DIR, REQUESTS_DIR, submit_request
from .timeutil import iso_z, parse_utc, utc_now

_DURATION = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def _admin_dir(account: str, live_base: str | None) -> Path:
    base = live_base or os.environ.get(LIVE_BASE_ENV_VAR)
    if not base:
        raise SystemExit(f"{LIVE_BASE_ENV_VAR} is not set (or pass --live-base)")
    if not is_account(account):
        raise SystemExit(f"{account!r} is not an account name (<broker>.<env>)")
    path = Path(base) / account / ADMIN_FOLDER
    if not path.is_dir():
        raise SystemExit(f"{path} does not exist: no Account Admin is deployed for {account}")
    return path


def _until(args) -> str:
    if args.until:
        try:
            return iso_z(parse_utc(args.until))
        except ValueError as exc:
            raise SystemExit(f"--until: {exc}") from exc
    m = _DURATION.match(args.for_ or "")
    if not m:
        raise SystemExit("give --until <ISO-8601 UTC> or --for <N>[s|m|h|d]")
    return iso_z(utc_now() + timedelta(**{_UNITS[m.group(2)]: int(m.group(1))}))


def _show(admin: Path) -> None:
    for path in sorted(admin.glob("*.json")):
        if path.name in ("config.json", "state.json"):
            continue
        print(f"== {path.name}")
        print(path.read_text(encoding="utf-8"))
    inbox = admin / REQUESTS_DIR
    waiting = sorted(p.name for p in inbox.glob("*.json")) if inbox.is_dir() else []
    print(f"== requests waiting: {waiting or 'none'}")
    done = sorted((inbox / DONE_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)[-5:] if inbox.is_dir() else []
    for path in done:
        record = json.loads(path.read_text(encoding="utf-8"))
        print(f"   done {path.name}: {record.get('outcome')} - {record.get('reason')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="account-admin-request", description=__doc__.split("\n\n")[0])
    ap.add_argument("account", help="the account folder name, e.g. icmarkets.demo")
    ap.add_argument("--live-base", default=None, help=f"defaults to {LIVE_BASE_ENV_VAR}")
    ap.add_argument("--operator", default=None, help="who is asking (defaults to the OS user)")
    sub = ap.add_subparsers(dest="command", required=True)
    reset = sub.add_parser("reset-latch", help="clear a tripped latch; it re-trips at once if its condition still holds")
    reset.add_argument("--target", required=True, help="a condition id (e.g. max_loss, state_lost) or 'all'")
    override = sub.add_parser("override", help="tighten the directive until a time (never loosens)")
    override.add_argument("--directive", required=True, choices=[d.value for d in AccountDirective if d is not AccountDirective.ALL_OPS])
    override.add_argument("--until", default=None, help="ISO-8601 UTC, e.g. 2026-09-24T19:00:00Z")
    override.add_argument("--for", dest="for_", default=None, help="a duration instead of --until, e.g. 90m, 2h")
    override.add_argument("--reason", default="")
    sub.add_parser("clear-override", help="end the active override")
    sub.add_parser("show", help="print what the Admin currently publishes")
    args = ap.parse_args(argv)

    admin = _admin_dir(args.account, args.live_base)
    if args.command == "show":
        _show(admin)
        return 0
    operator = args.operator or getpass.getuser()
    task = AdminTaskKind.PROP_GUARD.value
    if args.command == "reset-latch":
        path = submit_request(admin / REQUESTS_DIR, operator=operator, task=task, kind=RequestKind.RESET_LATCH.value,
                              target=args.target)
    elif args.command == "override":
        path = submit_request(admin / REQUESTS_DIR, operator=operator, task=task, kind=RequestKind.OVERRIDE.value,
                              directive=args.directive, until_utc=_until(args), reason=args.reason)
    else:
        path = submit_request(admin / REQUESTS_DIR, operator=operator, task=task, kind=RequestKind.CLEAR_OVERRIDE.value)
    print(f"request written: {path}\nThe Admin applies or rejects it at its next cycle; see {admin / REQUESTS_DIR / DONE_DIR}")
    if not (admin / DIRECTIVE_FILE).exists():
        print("note: this Admin publishes no directive.json (is prop_guard configured and the Admin running?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
