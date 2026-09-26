"""The directive file (ACCOUNT_ADMIN_SPEC §4, §10.2): its ONE schema, its writer and its reader.

The Admin's ``prop_guard`` task writes it; every trading system's guard reads it. Both import this module, so the
writer and the reader can never disagree about the format (spec §10.4).

Reading never raises. Every failure maps to a row of the reading table, and a file that cannot be trusted applies
``NO_ENTRY_OPS``: a failed Admin must never read as "trade freely" (invariant 5).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..account import LIVE_BASE_ENV_VAR
from .atomic import atomic_write_json, read_json_retry
from .enums import AccountDirective, DirectiveSource
from .timeutil import iso_z, parse_utc, utc_now

logger = logging.getLogger(__name__)

#: The Admin's folder inside an account folder, ``<live_base>\<account>\_account_admin`` (spec §3.1).
ADMIN_FOLDER = "_account_admin"
DIRECTIVE_FILE = "directive.json"
DIRECTIVE_SCHEMA_VERSION = 1

#: A heartbeat this far ahead of the reader's clock is a broken file, not a fresh one (spec §10.2).
FUTURE_TOLERANCE_S = 5.0

#: The fields a reader acts on or checks; everything else in the file is evidence (spec §4.1).
_EVIDENCE_KEYS = ("trading_day", "levels", "metrics", "override")


class DirectiveFormatError(ValueError):
    """A directive file that does not follow the schema this reader knows."""


@dataclass(frozen=True)
class DirectiveAccount:
    """The account the Admin evaluated: what the terminal reported, checked by every reader (spec §3.3)."""

    broker: str
    server: str
    login: int
    currency: str

    def to_dict(self) -> dict[str, Any]:
        return {"broker": self.broker, "server": self.server, "login": self.login, "currency": self.currency}


@dataclass(frozen=True)
class DirectiveFile:
    """One directive, as written by ``prop_guard`` (spec §4.1). ``evidence`` carries ``trading_day``, ``levels``,
    ``metrics`` and ``override``: the Admin's evidence for logs and alerts, never acted on by a reader."""

    account: DirectiveAccount
    directive: AccountDirective
    reason: str
    causes: tuple[str, ...]
    since_utc: datetime
    heartbeat_utc: datetime
    valid_for_s: int
    sequence: int
    episode: int
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": DIRECTIVE_SCHEMA_VERSION,
            "account": self.account.to_dict(),
            "directive": str(self.directive),
            "reason": self.reason,
            "causes": list(self.causes),
            "since_utc": iso_z(self.since_utc),
            "heartbeat_utc": iso_z(self.heartbeat_utc),
            "valid_for_s": self.valid_for_s,
            "sequence": self.sequence,
            "episode": self.episode,
        }
        for key in _EVIDENCE_KEYS:
            out[key] = self.evidence.get(key)
        return out

    @classmethod
    def from_dict(cls, payload: Any) -> "DirectiveFile":
        """Parse a directive file. Raises :class:`DirectiveFormatError` for anything a reader must not trust; unknown
        extra fields are ignored (spec §4.2: evidence may grow without a schema bump)."""
        if not isinstance(payload, Mapping):
            raise DirectiveFormatError("directive file is not a JSON object")
        version = payload.get("schema_version")
        if version != DIRECTIVE_SCHEMA_VERSION:
            raise DirectiveFormatError(f"unknown schema_version {version!r} (this reader knows {DIRECTIVE_SCHEMA_VERSION})")
        try:
            directive = AccountDirective(payload["directive"])
        except (KeyError, ValueError) as exc:
            raise DirectiveFormatError(f"unknown directive {payload.get('directive')!r}") from exc
        try:
            acc = payload["account"]
            account = DirectiveAccount(broker=str(acc["broker"]), server=str(acc["server"]), login=int(acc["login"]),
                                       currency=str(acc["currency"]))
            valid_for_s = int(payload["valid_for_s"])
            if valid_for_s <= 0:
                raise ValueError("valid_for_s must be positive")
            return cls(account=account, directive=directive, reason=str(payload.get("reason") or ""),
                       causes=tuple(str(c) for c in payload.get("causes") or ()),
                       since_utc=parse_utc(payload["since_utc"]), heartbeat_utc=parse_utc(payload["heartbeat_utc"]),
                       valid_for_s=valid_for_s, sequence=int(payload["sequence"]), episode=int(payload["episode"]),
                       evidence={k: payload.get(k) for k in _EVIDENCE_KEYS})
        except DirectiveFormatError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise DirectiveFormatError(f"malformed directive file: {exc}") from exc


def admin_dir(live_base: str | Path, account: str) -> Path:
    """``<live_base>\\<account>\\_account_admin``."""
    return Path(live_base) / account / ADMIN_FOLDER


def directive_path(live_base: str | Path, account: str) -> Path:
    """``<live_base>\\<account>\\_account_admin\\directive.json``: the fixed path a system in that account reads."""
    return admin_dir(live_base, account) / DIRECTIVE_FILE


def write_directive(path: str | Path, directive: DirectiveFile) -> None:
    """Atomically replace the directive file (spec §7.4). Raises ``OSError`` when the replace kept failing."""
    atomic_write_json(path, directive.to_dict())


@dataclass(frozen=True)
class DirectiveReading:
    """What a reader applies, and why (spec §10.2)."""

    directive: AccountDirective
    source: DirectiveSource
    file: DirectiveFile | None = None
    path: Path | None = None
    detail: str = ""


def _resolve_live_base(live_base: str | Path | None) -> Path | None:
    raw = live_base if live_base is not None else os.environ.get(LIVE_BASE_ENV_VAR)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(os.path.expandvars(str(raw).strip())))


def _judge(file: DirectiveFile, path: Path, *, login: int | None, server: str | None, now: datetime) -> DirectiveReading:
    """Apply the reading table's freshness and identity rows to a parsed file."""
    fallback = AccountDirective.NO_ENTRY_OPS
    if (file.heartbeat_utc - now).total_seconds() > FUTURE_TOLERANCE_S:
        return DirectiveReading(fallback, DirectiveSource.INVALID, file, path,
                                f"heartbeat {iso_z(file.heartbeat_utc)} is in the future (reader clock {iso_z(now)})")
    if login is None or server is None:
        return DirectiveReading(fallback, DirectiveSource.WRONG_ACCOUNT, file, path,
                                "the live terminal did not report its login/server; the directive's account cannot be verified")
    if int(file.account.login) != int(login) or str(file.account.server) != str(server):
        return DirectiveReading(fallback, DirectiveSource.WRONG_ACCOUNT, file, path,
                                f"directive is for {file.account.login}@{file.account.server}, terminal is on {login}@{server}")
    age = (now - file.heartbeat_utc).total_seconds()
    if age > file.valid_for_s:
        return DirectiveReading(fallback, DirectiveSource.STALE, file, path,
                                f"heartbeat {age:.0f}s old > valid_for_s {file.valid_for_s}")
    return DirectiveReading(file.directive, DirectiveSource.FILE, file, path, file.reason)


def _load(path: Path) -> DirectiveFile:
    """Read and parse one file. ``FileNotFoundError`` passes through; everything else is a ``DirectiveFormatError``."""
    try:
        payload = read_json_retry(path)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise DirectiveFormatError(f"unreadable: {exc}") from exc
    return DirectiveFile.from_dict(payload)


def read_directive(*, account: str | None, login: int | None, server: str | None, live_base: str | Path | None = None,
                   now: datetime | None = None) -> DirectiveReading:
    """The directive a system applies right now (spec §10.2). Never raises.

    ``account`` is the account folder the system is deployed in (``deployment_account()``), or ``None`` for a system
    outside one; ``login``/``server`` are its live terminal's. A system in an account folder reads that account's file.
    A system outside one looks for the directive whose account matches its terminal (spec §3.1).
    """
    now = now if now is not None else utc_now()
    try:
        base = _resolve_live_base(live_base)
        if base is None:
            return DirectiveReading(AccountDirective.ALL_OPS, DirectiveSource.ABSENT,
                                    detail=f"{LIVE_BASE_ENV_VAR} is not set: no directive can be located")
        if account is not None:
            return _read_account(directive_path(base, account), login=login, server=server, now=now)
        return _read_by_terminal(base, login=login, server=server, now=now)
    except Exception as exc:  # the table is exhaustive; this only guards against a defect here
        logger.exception("directive read failed unexpectedly")
        return DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID, detail=f"reader error: {exc!r}")


def _read_account(path: Path, *, login: int | None, server: str | None, now: datetime) -> DirectiveReading:
    try:
        file = _load(path)
    except FileNotFoundError:
        return DirectiveReading(AccountDirective.ALL_OPS, DirectiveSource.ABSENT, path=path,
                                detail="no directive file: this account is not governed")
    except DirectiveFormatError as exc:
        return DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID, path=path, detail=str(exc))
    return _judge(file, path, login=login, server=server, now=now)


def _read_by_terminal(base: Path, *, login: int | None, server: str | None, now: datetime) -> DirectiveReading:
    """Outside an account folder: the one directive whose account matches this terminal (spec §3.1, §10.1).

    None matching reads as absent. Two matching is a misconfiguration and reads as invalid. A file that cannot be
    parsed might be this terminal's own, so when nothing matches but such a file exists the read is invalid, not
    absent: an untrustworthy file must never read as "trade freely".
    """
    if login is None or server is None:
        return DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.WRONG_ACCOUNT,
                                detail="outside an account folder and the terminal did not report its login/server")
    matches: list[tuple[Path, DirectiveFile]] = []
    unreadable: list[str] = []
    for path in sorted(base.glob(f"*/{ADMIN_FOLDER}/{DIRECTIVE_FILE}")):
        try:
            file = _load(path)
        except FileNotFoundError:
            continue
        except DirectiveFormatError as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        if int(file.account.login) == int(login) and str(file.account.server) == str(server):
            matches.append((path, file))
    if len(matches) > 1:
        return DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID,
                                detail="several account folders govern this terminal: " + ", ".join(str(p) for p, _ in matches))
    if len(matches) == 1:
        path, file = matches[0]
        return _judge(file, path, login=login, server=server, now=now)
    if unreadable:
        return DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID,
                                detail="no directive matches this terminal, and some cannot be read: " + "; ".join(unreadable))
    return DirectiveReading(AccountDirective.ALL_OPS, DirectiveSource.ABSENT,
                            detail=f"no directive under {base} governs {login}@{server}")
