"""The Account Admin as a Supervisor runner (ACCOUNT_ADMIN_SPEC §3.1, §7.2, §7.6, §9).

Broker-neutral: a broker library's entry point validates the config (stage 1), resolves where the Admin is deployed,
takes the writer lock, attaches to its terminal and checks the account (stage 2), then hands a built
:class:`~okmich_quant_core.account_admin.host.AdminHost` to :class:`AdminRunLoop`.

To the Supervisor the Admin is an ordinary runner in its account folder: it writes the core ``status.json`` under the
runner root ``_account_admin`` and one ``bar`` heartbeat per minute for the logical system ``_account_admin/account/1``.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

from ..account import LIVE_BASE_ENV_VAR, deployment_account
from ..broker_session import BrokerSession
from ..process_control import reenable_ctrl_c
from ..logging import BarOutcome, JsonlEventLogger, LogicalSystemIdentity, RunnerIdentity, RunnerStatus, SystemRecordFactory
from ..logging.identity import LogRootConfigError, _resolve_log_base, runner_log_dir
from .clock import AdminClock
from .config import AdminConfigError
from .directive import ADMIN_FOLDER
from .host import AdminHost, CycleReport
from .lock import WRITER_LOCK_FILE, WriterLock

logger = logging.getLogger(__name__)

#: The Admin's liveness leg: one logical system, symbol ``account``, timeframe 1 minute (spec §9).
HEARTBEAT_SYMBOL = "account"
HEARTBEAT_TIMEFRAME_MIN = 1


@dataclass(frozen=True)
class AdminDeployment:
    """Where a deployed Admin lives and logs, derived from its own location (spec §3.1, §3.3)."""

    live_base: Path
    account: str
    admin_dir: Path          # <live_base>\<account>\_account_admin
    live_account_dir: Path   # <live_base>\<account>
    log_dir: Path            # <log_base>\<account>\_account_admin
    log_account_dir: Path    # <log_base>\<account>

    @property
    def writer_lock_path(self) -> Path:
        return self.admin_dir / WRITER_LOCK_FILE


def resolve_deployment(script: str | Path, log_base: str | Path | None = None) -> AdminDeployment:
    """Where the Admin whose ``run.py`` is ``script`` is deployed. Raises :class:`AdminConfigError` (stage 1) when it is
    not deployed as ``<live_base>\\<account>\\_account_admin\\run.py`` or the log root is not configured."""
    problems: list[str] = []
    raw_base = os.environ.get(LIVE_BASE_ENV_VAR)
    if not raw_base or not raw_base.strip():
        raise AdminConfigError([f"{LIVE_BASE_ENV_VAR} is not set: the Admin cannot locate its account folder"])
    live_base = Path(os.path.expanduser(os.path.expandvars(raw_base.strip()))).resolve()
    script_path = Path(script).resolve()
    account = deployment_account(script_path, live_base)
    if account is None:
        problems.append(f"{script_path} is not deployed in an account folder <live_base>\\<broker>.<env>\\")
    elif script_path.parent != live_base / account / ADMIN_FOLDER:
        problems.append(f"{script_path} must sit at {live_base / account / ADMIN_FOLDER / 'run.py'}")
    try:
        log_dir = runner_log_dir(ADMIN_FOLDER, log_base)
        log_account_dir = _resolve_log_base(log_base)
    except LogRootConfigError as exc:
        problems.append(str(exc))
        log_dir = log_account_dir = None
    if problems:
        raise AdminConfigError(problems)
    return AdminDeployment(live_base=live_base, account=account, admin_dir=live_base / account / ADMIN_FOLDER,
                           live_account_dir=live_base / account, log_dir=log_dir, log_account_dir=log_account_dir)


def _library_versions() -> dict:
    out: dict = {}
    for name in ("okmich-quant-core", "okmich-quant-mt5"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


class AdminRunLoop:
    """Owns the process lifecycle: status file, heartbeat, clock, graceful stop.

    A stop request (Ctrl+C from the Supervisor, SIGTERM) lets the current cycle finish, then the loop releases the
    writer lock, disconnects, and writes the stopped status (spec §7.6, §9). The directive and state stay in place, so a
    restart within ``valid_for_s`` is invisible to the trading systems.
    """

    def __init__(self, host: AdminHost, clock: AdminClock, *, broker_session: BrokerSession, writer_lock: WriterLock,
                 log_base: str | Path | None = None, poll_s: float = 0.25, time_fn: Callable[[], float] = time.time,
                 sleep_fn: Callable[[float], None] = time.sleep):
        self.host = host
        self.clock = clock
        self.broker_session = broker_session
        self.writer_lock = writer_lock
        self.log_base = log_base
        self.poll_s = poll_s
        self._time = time_fn
        self._sleep = sleep_fn
        self._stop_requested = False
        self._logical = LogicalSystemIdentity(strategy=ADMIN_FOLDER, symbol=HEARTBEAT_SYMBOL,
                                              timeframe_minutes=HEARTBEAT_TIMEFRAME_MIN)

    def request_stop(self, *_args) -> None:
        self._stop_requested = True

    def _install_signal_handlers(self) -> None:
        reenable_ctrl_c()   # a parent that ignores Ctrl+C must not disable the Supervisor's graceful stop
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):   # not the main thread
                logger.debug("could not install %s handler", name)

    @staticmethod
    def heartbeat_asof(t: float) -> datetime:
        """The open of the last completed minute before ``t`` (spec §9)."""
        minute_open = datetime.fromtimestamp(int(t // 60) * 60, tz=timezone.utc)
        return minute_open - timedelta(minutes=1)

    def run(self, max_cycles: int | None = None) -> str:
        """Run until stopped; returns the stop reason. ``max_cycles`` bounds the loop for tests and drills."""
        bs = self.broker_session
        runner = RunnerIdentity.generate(name=ADMIN_FOLDER, broker=bs.broker, account_id=bs.account_id,
                                         broker_session_id=bs.broker_session_id)
        status = RunnerStatus(runner, [self._logical], log_base=self.log_base, library_versions=_library_versions())
        heartbeat = JsonlEventLogger(self._logical, log_base=self.log_base)
        factory = SystemRecordFactory(runner, self._logical)
        reason = "operator_stop"
        close_ok = True
        now = datetime.now(timezone.utc)
        status.mark_started()
        self._install_signal_handlers()
        try:
            self.host.start(now)
            next_start = self.clock.release(self._time())
            last_heartbeat_minute: int | None = None
            cycles = 0
            while not self._stop_requested:
                t = self._time()
                if t < next_start:
                    self._sleep(min(self.poll_s, next_start - t))
                    continue
                now = datetime.fromtimestamp(t, tz=timezone.utc)
                try:
                    report = self.host.run_cycle(now)
                except Exception as exc:   # the host isolates tasks; this guards the host itself
                    logger.exception("Admin cycle raised")
                    report = CycleReport(degraded=True, degraded_reason=f"cycle raised: {exc!r}")
                minute = int(t // 60)
                if minute != last_heartbeat_minute:
                    last_heartbeat_minute = minute
                    outcome = BarOutcome.ERROR if report.degraded or report.failed_tasks else BarOutcome.OK
                    try:
                        heartbeat.write(factory.bar(asof_bar_ts=self.heartbeat_asof(t), outcome=outcome))
                    except Exception:
                        logger.exception("heartbeat write failed")
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    reason = "max_cycles"
                    break
                next_start = self.clock.next_start(t, self._time())
        except BaseException as exc:
            reason = f"crash: {type(exc).__name__}: {exc}"
            close_ok = False
            raise
        finally:
            try:
                self.host.stop(datetime.now(timezone.utc), reason)
            except Exception:
                close_ok = False
                logger.exception("host stop failed")
            try:
                heartbeat.close()
            except Exception:
                close_ok = False
                logger.exception("heartbeat logger close failed")
            self.writer_lock.release()
            disconnected = bs.disconnect()
            try:
                status.mark_stopped(broker_disconnected=disconnected, clean=close_ok and disconnected, reason=reason)
            except Exception:
                logger.exception("could not write the stopped status")
        return reason
