"""The MT5 Account Admin entry point: what the Admin's ``run.py`` calls (ACCOUNT_ADMIN_SPEC §6.7, §7.5, §9).

``run_admin(__file__)`` validates everything it can before touching the terminal (stage 1), then takes the writer lock,
attaches to the account's terminal WITHOUT logging in and checks the account (stage 2), then runs the core
:class:`~okmich_quant_core.account_admin.AdminRunLoop` until the Supervisor stops it.

Exit codes: 0 stopped cleanly; 2 refused in stage 1 (config, deployment, env file); 3 refused in stage 2 (lock,
terminal, account, server clock).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from okmich_quant_core import TelegramNotifier, setup_text_logger
from okmich_quant_core.account_admin import (AccountIdentity, AdminClock, AdminConfigError, AdminHost, AdminRunLoop,
                                             AlertLevel, NotifierAlerts, WriterLock, WriterLockHeld, load_admin_config,
                                             resolve_deployment)

from .actions import Mt5BrokerActions
from .server_clock import ServerClock, parse_broker_section
from .source import Mt5AccountSource, Mt5ReadError

logger = logging.getLogger(__name__)

ENV_DIR_VAR = "OKMICH_QUANT_ENV_DIR"
REQUIRED_ENV_KEYS = ("TERMINAL_PATH", "LOGIN_ID", "LOGIN_SERVER")
EXIT_OK, EXIT_STAGE1, EXIT_STAGE2 = 0, 2, 3

#: A stated server clock is re-checked against live ticks this often while running (spec §12.3).
CLOCK_RECHECK_S = 3600.0


class Mt5AdminSession:
    """The runner's broker session over an injected ``MetaTrader5`` module: identity for ``status.json`` and a proven
    disconnect (LOGGING_CONTRACT §7.4). The Admin attached, so it detaches; it never logged in, so it never logs out."""

    def __init__(self, mt5: Any, broker: str):
        self.mt5 = mt5
        self._broker = broker
        info = mt5.account_info()
        self._account_id = str(getattr(info, "login", "")) if info is not None else ""
        term = mt5.terminal_info()
        self._session_id = f"{getattr(term, 'name', 'mt5')}:build{getattr(term, 'build', '?')}" if term is not None else None
        self._disconnected: bool | None = None

    @property
    def broker(self) -> str:
        return self._broker

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def broker_session_id(self) -> str | None:
        return self._session_id

    def disconnect(self) -> bool:
        if self._disconnected is not None:
            return self._disconnected
        try:
            self.mt5.shutdown()
        except Exception:
            logger.exception("mt5.shutdown failed")
        try:
            self._disconnected = self.mt5.terminal_info() is None
        except Exception:
            self._disconnected = True
        return self._disconnected


class ClockCheckedSource(Mt5AccountSource):
    """An account source that re-checks the stated server clock against live ticks every ``CLOCK_RECHECK_S``. A tick
    that proves the clock wrong makes the read fail, so the cycle is degraded (``NO_ENTRY_OPS``) and no age is judged
    on a wrong clock."""

    def __init__(self, mt5: Any, clock: ServerClock, recheck_s: float = CLOCK_RECHECK_S):
        super().__init__(mt5, clock)
        self._recheck_s = recheck_s
        self._last_check: datetime | None = None

    def positions(self):
        rows = super().positions()
        now = datetime.now(timezone.utc)
        if self._last_check is None or (now - self._last_check).total_seconds() >= self._recheck_s:
            symbols = sorted({p.symbol for p in rows})
            problem = self.freshest_tick_problem(symbols, now) if symbols else None
            if problem:
                raise Mt5ReadError(f"server clock contradicted by a live tick: {problem}")
            self._last_check = now
        return rows


def _notifier_from(env: dict[str, Any]) -> NotifierAlerts | None:
    token = str(env.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = str(env.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        return None
    return NotifierAlerts(TelegramNotifier(bot_token=token, chat_id=chat, strategy_name="_account_admin",
                                           broker=str(env.get("LOGIN_SERVER") or "")))


def _refuse(stage: int, problems: list[str], notifier: NotifierAlerts | None, account: str | None) -> int:
    text = "\n".join(f"- {p}" for p in problems)
    logger.critical("Account Admin refuses to start (stage %d):\n%s", stage, text)
    print(f"Account Admin refuses to start (stage {stage}):\n{text}", flush=True)   # the Supervisor's console capture
    if notifier is not None:
        try:
            notifier.send_alert(AlertLevel.CRITICAL, f"ACCOUNT ADMIN REFUSED TO START [{account or '?'}]", text)
            notifier.close()
        except Exception:
            logger.exception("could not alert the refusal")
    return EXIT_STAGE1 if stage == 1 else EXIT_STAGE2


def run_admin(script: str | Path, *, mt5_module: Any = None, max_cycles: int | None = None) -> int:
    """Run the Account Admin whose ``run.py`` is ``script``. Returns the process exit code."""
    script = Path(script).resolve()
    config_path = script.parent / "config.json"
    setup_text_logger(config_path)

    # ---------------------------------------------------------------- stage 1: nothing touches the terminal
    problems: list[str] = []
    config = None
    try:
        config = load_admin_config(config_path)
    except AdminConfigError as exc:
        problems.extend(exc.problems)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        broker_raw = raw.get("broker") if isinstance(raw, dict) else None
    except (OSError, ValueError):
        broker_raw = None
    server_clock, clock_problems = parse_broker_section(broker_raw)
    problems.extend(clock_problems)
    deployment = None
    try:
        deployment = resolve_deployment(script)
    except AdminConfigError as exc:
        problems.extend(exc.problems)
    env: dict[str, Any] = {}
    env_dir = os.environ.get(ENV_DIR_VAR, "").strip()
    if not env_dir:
        problems.append(f"{ENV_DIR_VAR} is not set: the Admin cannot find its account's broker env file")
    elif deployment is not None:
        env_file = Path(env_dir) / f".env.{deployment.account}"
        if not env_file.is_file():
            problems.append(f"{env_file}: the account's broker env file is missing")
        else:
            env = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
            problems.extend(f"{env_file}: {key} is missing" for key in REQUIRED_ENV_KEYS if not str(env.get(key) or "").strip())
            if str(env.get("LOGIN_ID") or "").strip() and not str(env["LOGIN_ID"]).strip().isdigit():
                problems.append(f"{env_file}: LOGIN_ID must be the numeric account login")
    notifier = _notifier_from(env)
    account = deployment.account if deployment is not None else None
    if problems or config is None or deployment is None or server_clock is None:
        return _refuse(1, problems or ["invalid configuration"], notifier, account)
    load_dotenv(Path(env_dir) / f".env.{deployment.account}", override=True)
    expected_login, expected_server = int(env["LOGIN_ID"]), str(env["LOGIN_SERVER"]).strip()

    # ---------------------------------------------------------------- stage 2: lock, attach, verify
    lock = WriterLock(deployment.writer_lock_path)
    try:
        lock.acquire()
    except WriterLockHeld as exc:
        return _refuse(2, [str(exc)], notifier, account)
    if mt5_module is None:
        import MetaTrader5 as mt5_module
    mt5 = mt5_module
    if not mt5.initialize(path=str(env["TERMINAL_PATH"])):   # attach only: the Admin never calls login (invariant 6)
        lock.release()
        return _refuse(2, [f"could not attach to the terminal {env['TERMINAL_PATH']}: {mt5.last_error()}"], notifier, account)
    info = mt5.account_info()
    stage2: list[str] = []
    if info is None:
        stage2.append("the terminal did not report an account: bring its session up before starting the Admin")
    elif int(info.login) != expected_login or str(info.server) != expected_server:
        stage2.append(f"the terminal is logged into {info.login}@{info.server}, but .env.{account} governs "
                      f"{expected_login}@{expected_server}")
    source = ClockCheckedSource(mt5, server_clock)
    if not stage2:
        try:
            symbols = sorted({p.symbol for p in source.positions()} | {o.symbol for o in source.pending_orders()})
            problem = source.freshest_tick_problem(symbols, datetime.now(timezone.utc))
            if problem:
                stage2.append(f"the stated server clock is wrong: {problem}")
        except Mt5ReadError as exc:
            stage2.append(str(exc))
    if stage2:
        try:
            mt5.shutdown()
        finally:
            lock.release()
        return _refuse(2, stage2, notifier, account)

    identity = AccountIdentity(account=deployment.account, broker_label=str(env.get("BROKER_NAME") or "").strip().lower()
                               or deployment.account.split(".")[0], login=expected_login, server=expected_server)
    host = AdminHost(config, identity=identity, admin_dir=deployment.admin_dir, log_dir=deployment.log_dir,
                     live_account_dir=deployment.live_account_dir, log_account_dir=deployment.log_account_dir,
                     source=source, actions=Mt5BrokerActions(mt5), notifier=notifier)
    logger.info("Account Admin for %s (%s@%s): tasks %s; server clock %s %+gh", deployment.account, expected_login,
                expected_server, [t.kind.value for t in config.tasks], server_clock.zone.key, server_clock.shift_h)
    loop = AdminRunLoop(host, AdminClock(config.clock), broker_session=Mt5AdminSession(mt5, expected_server),
                        writer_lock=lock)
    try:
        reason = loop.run(max_cycles=max_cycles)
    finally:
        if notifier is not None:
            notifier.close()
    logger.info("Account Admin stopped: %s", reason)
    return EXIT_OK
