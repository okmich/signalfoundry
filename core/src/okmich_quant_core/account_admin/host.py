"""The Account Admin host: one cycle across every configured task (ACCOUNT_ADMIN_SPEC §7.1, §12.1).

The host owns what the tasks share: the account check, the one snapshot per cycle, the request inbox, the state file,
the output files, the audit log and alerts. It is broker-neutral: a broker library supplies the account through an
:class:`AccountSource` and the two book actions through a :class:`BrokerActions`.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..notification.base import BaseNotifier
from .atomic import atomic_write_json
from .audit import AuditLog
from .config import AdminConfig
from .enums import AdminEvent, AlertLevel, BookActionKind, BookActionOutcome, RequestOutcome
from .owners import MagicOwner, magic_owners
from .requests import REQUESTS_DIR, AdminRequest, PendingRequest, RequestInbox
from .snapshot import AccountSnapshot, AccountSource, BookActionResult, BrokerActions, PendingOrder, Position
from .state import STATE_FILE, StateLoad, load_state, save_state
from .tasks.base import AdminTask, Alert, TaskContext, TaskResult
from .timeutil import iso_z

logger = logging.getLogger(__name__)

#: The host's own slot in state.json (not a task kind: task kinds are enum values without a leading underscore).
HOST_SLOT = "_host"
#: How many consumed request ids the host remembers, so a request whose move to done\ failed is never applied twice.
MAX_CONSUMED_IDS = 500


class AdminNotifier(Protocol):
    def send_alert(self, level: AlertLevel, title: str, body: str) -> None:
        ...


class NotifierAlerts:
    """:class:`AdminNotifier` over any core notifier (Telegram in production)."""

    def __init__(self, notifier: BaseNotifier):
        self.notifier = notifier

    def send_alert(self, level: AlertLevel, title: str, body: str) -> None:
        self.notifier.on_account_event(title, body, level.value)

    def close(self) -> None:
        self.notifier.close()


@dataclass(frozen=True)
class AccountIdentity:
    """The governed account: its folder name, and what its env file says the terminal must be logged into."""

    account: str
    broker_label: str
    login: int
    server: str


@dataclass
class CycleReport:
    """What one cycle did, for the runner's heartbeat and logs."""

    degraded: bool
    degraded_reason: str | None = None
    failed_tasks: list[str] = field(default_factory=list)
    directive: str | None = None


class _TaskActionPort:
    """The action port handed to one task for one cycle: refuses in a degraded cycle, audits every call."""

    def __init__(self, host: "AdminHost", task: AdminTask, degraded: bool, now: datetime):
        self._host, self._task, self._degraded, self._now = host, task, degraded, now
        self.records: list[dict[str, Any]] = []

    def _do(self, kind: BookActionKind, target: PendingOrder | Position, reason: str, call) -> BookActionResult:
        if self._degraded:
            result = BookActionResult(BookActionOutcome.REFUSED, "degraded cycle: the account is not verified")
        else:
            try:
                result = call(target)
            except Exception as exc:
                logger.exception("%s %s raised", kind, target.ticket)
                result = BookActionResult(BookActionOutcome.FAILED, f"{type(exc).__name__}: {exc}")
        record = {"action": str(kind), "reason": reason, "ticket": target.ticket, "magic": target.magic,
                  "symbol": target.symbol, "volume": target.volume, "outcome": str(result.outcome), "error": result.error,
                  "retcode": result.retcode}
        if isinstance(target, PendingOrder):
            record.update({"type": str(target.order_type), "price": target.price, "setup_utc": iso_z(target.setup_utc)})
        else:
            record.update({"side": str(target.side), "price_open": target.price_open, "profit": target.profit})
        self.records.append(record)
        self._host._pending_events.append((self._task.kind.value, AdminEvent.BOOK_ACTION, record))
        (logger.warning if result.outcome is BookActionOutcome.FAILED else logger.info)(
            "%s: %s #%s %s magic %s -> %s %s", self._task.kind.value, kind, target.ticket, target.symbol, target.magic,
            result.outcome, result.error or "")
        return result

    def cancel_pending(self, order: PendingOrder, *, reason: str) -> BookActionResult:
        return self._do(BookActionKind.CANCEL_PENDING, order, reason, self._host.actions.cancel_pending)

    def close_position(self, position: Position, *, reason: str) -> BookActionResult:
        return self._do(BookActionKind.CLOSE_POSITION, position, reason, self._host.actions.close_position)


class AdminHost:
    """Runs cycles. Construct after stage-1 validation and the stage-2 account check; call :meth:`start` once, then
    :meth:`run_cycle` on each clock tick, then :meth:`stop`."""

    def __init__(self, config: AdminConfig, *, identity: AccountIdentity, admin_dir: str | Path, log_dir: str | Path,
                 live_account_dir: str | Path, log_account_dir: str | Path | None, source: AccountSource,
                 actions: BrokerActions, notifier: AdminNotifier | None = None):
        self.config = config
        self.identity = identity
        self.admin_dir = Path(admin_dir)
        self.live_account_dir = Path(live_account_dir)
        self.log_account_dir = Path(log_account_dir) if log_account_dir is not None else None
        self.source = source
        self.actions = actions
        self.notifier = notifier
        self.audit = AuditLog(Path(log_dir) / "audit", identity.account)
        self.inbox = RequestInbox(self.admin_dir / REQUESTS_DIR)
        self.state_path = self.admin_dir / STATE_FILE
        self._slots: dict[str, dict[str, Any]] = {}
        self._state_load = StateLoad.MISSING
        self._previous_outputs: dict[str, dict[str, Any] | None] = {}
        self._first_cycle = True
        self._failing_tasks: set[str] = set()
        self._last_write_alert: datetime | None = None
        self._audit_ids: tuple[int | None, int | None] = (None, None)
        self._consumed_ids: list[str] = []
        self._pending_events: list[tuple[str | None, AdminEvent, dict[str, Any]]] = []

    # ------------------------------------------------------------------------------------------ lifecycle
    def start(self, now: datetime) -> None:
        self.admin_dir.mkdir(parents=True, exist_ok=True)
        self.inbox.directory.mkdir(parents=True, exist_ok=True)
        self._slots, self._state_load = load_state(self.state_path)
        self._consumed_ids = list((self._slots.get(HOST_SLOT) or {}).get("consumed_request_ids") or [])
        for task in self.config.tasks:
            self._previous_outputs[task.kind.value] = self._read_output(task)
        self.audit.write(AdminEvent.ADMIN_STARTED, now=now, tasks=[t.kind.value for t in self.config.tasks],
                         state_load=str(self._state_load), login=self.identity.login, server=self.identity.server)
        self._alert(AlertLevel.INFO, "ACCOUNT ADMIN STARTED",
                    f"tasks: {', '.join(t.kind.value for t in self.config.tasks)}; state {self._state_load}")

    def stop(self, now: datetime, reason: str) -> None:
        self.audit.write(AdminEvent.ADMIN_STOPPED, now=now, reason=reason, sequence=self._audit_ids[0],
                         episode=self._audit_ids[1])
        self._alert(AlertLevel.WARNING, "ACCOUNT ADMIN STOPPED", reason)

    def _read_output(self, task: AdminTask) -> dict[str, Any] | None:
        try:
            with open(self.admin_dir / task.output_name, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload if isinstance(payload, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {}   # present but unreadable: still "an output existed" for a task judging lost history

    # ------------------------------------------------------------------------------------------ the cycle
    def run_cycle(self, now: datetime) -> CycleReport:
        self._pending_events = []
        snapshot, degraded_reason = self._read_account(now)
        degraded = snapshot is None
        report = CycleReport(degraded=degraded, degraded_reason=degraded_reason)

        routed, consumed = self._route_requests(now)
        owners_cache: dict[int, MagicOwner] | None = None

        def owners() -> dict[int, MagicOwner]:
            nonlocal owners_cache
            if owners_cache is None:
                owners_cache = magic_owners(self.live_account_dir, self.log_account_dir, self.identity.account)
            return owners_cache

        results: dict[str, TaskResult] = {}
        ports: dict[str, _TaskActionPort] = {}
        alerts: list[Alert] = []
        for task in self.config.tasks:
            kind = task.kind.value
            port = _TaskActionPort(self, task, degraded, now)
            ports[kind] = port
            ctx = TaskContext(now=now, account=self.identity.account, degraded=degraded, degraded_reason=degraded_reason,
                              snapshot=snapshot, slot=self._slots.get(kind) or {}, requests=routed.get(kind, []),
                              actions=port, deals=self.source.deals, owners=owners,
                              valid_for_s=self.config.clock.valid_for_s, broker_label=self.identity.broker_label,
                              expected_login=self.identity.login, expected_server=self.identity.server,
                              first_cycle=self._first_cycle, state_load=str(self._state_load),
                              # Until the task has a slot it may still be judging lost history, even if its first
                              # cycle failed: keep telling it what output it found at start.
                              previous_output=self._previous_outputs.get(kind) if kind not in self._slots else None)
            try:
                result = task.on_cycle(ctx)
            except Exception as exc:
                report.failed_tasks.append(kind)
                logger.exception("task %s failed", kind)
                self._pending_events.append((kind, AdminEvent.TASK_FAILED,
                                             {"error": f"{type(exc).__name__}: {exc}",
                                              "traceback": traceback.format_exc(limit=8)}))
                if kind not in self._failing_tasks:
                    self._failing_tasks.add(kind)
                    alerts.append(Alert(AlertLevel.CRITICAL, f"TASK FAILED: {kind}",
                                        f"{type(exc).__name__}: {exc}. Its state and output keep their last values; "
                                        f"the other tasks still run. Alerted once until it succeeds."))
                continue
            self._failing_tasks.discard(kind)
            results[kind] = result
            self._slots[kind] = result.slot
            if result.audit_ids is not None and task.governs_directive:
                self._audit_ids = result.audit_ids
                report.directive = result.output.get("directive")
            alerts.extend(result.alerts)
            for event in result.events:
                self._pending_events.append((kind, event.event, dict(event.fields)))

        self._persist(now, results)
        self._complete_requests(now, consumed, results)
        self._audit_cycle(now, report, results)
        self._report_book_actions(ports, alerts)
        for alert in alerts:
            self._alert(alert.level, alert.title, alert.body)
        self._first_cycle = False
        return report

    def _read_account(self, now: datetime) -> tuple[AccountSnapshot | None, str | None]:
        """Verify the terminal is on the governed account, then read it once (spec §7.1 steps 1-2, §7.5)."""
        want = f"{self.identity.login}@{self.identity.server}"
        try:
            info = self.source.account_info()
        except Exception as exc:
            return None, f"account_info failed: {type(exc).__name__}: {exc}"
        if info is None:
            return None, "the terminal did not report its account"
        if int(info.login) != int(self.identity.login) or str(info.server) != str(self.identity.server):
            return None, f"the terminal is on {info.login}@{info.server}, not the governed {want}"
        try:
            positions = tuple(self.source.positions())
            orders = tuple(self.source.pending_orders())
        except Exception as exc:
            return None, f"reading the book failed: {type(exc).__name__}: {exc}"
        return AccountSnapshot(taken_utc=now, info=info, positions=positions, orders=orders), None

    def _route_requests(self, now: datetime) -> tuple[dict[str, list[AdminRequest]], list[PendingRequest]]:
        """Check each waiting request's envelope and route it to its task (spec §8.2). Requests the host rejects are
        completed at once; routed ones are completed after the tasks ran."""
        configured = {t.kind.value for t in self.config.tasks}
        routed: dict[str, list[AdminRequest]] = {}
        consumed: list[PendingRequest] = []
        for item in self.inbox.pending():
            req = item.request
            if req is not None and req.request_id in self._consumed_ids:
                # Already applied in an earlier cycle; only its move to done\ failed. Retry the move, never the request.
                self._finish_request(item, RequestOutcome.REJECTED, "duplicate: this request was already consumed", now,
                                     req.task, alert=False)
                continue
            if req is None:
                self._finish_request(item, RequestOutcome.REJECTED, f"malformed request: {item.error}", now, None)
                continue
            age = (now - req.created_utc).total_seconds()
            if age > self.config.request_ttl_s:
                self._finish_request(item, RequestOutcome.REJECTED,
                                     f"older than request_ttl_s ({age:.0f}s > {self.config.request_ttl_s}s): not replayed blind",
                                     now, req.task)
                continue
            if age < -5:
                self._finish_request(item, RequestOutcome.REJECTED, "created_utc is in the future", now, req.task)
                continue
            if req.task not in configured:
                self._finish_request(item, RequestOutcome.REJECTED, f"task {req.task!r} is not configured on this Admin",
                                     now, req.task)
                continue
            routed.setdefault(req.task, []).append(req)
            consumed.append(item)
        return routed, consumed

    def _complete_requests(self, now: datetime, consumed: list[PendingRequest], results: dict[str, TaskResult]) -> None:
        for item in consumed:
            req = item.request
            result = results.get(req.task)
            if result is None:
                continue   # its task failed this cycle: the request stays in the inbox for the next one
            outcome, why = result.request_outcomes.get(req.request_id, (RequestOutcome.REJECTED, "the task did not answer"))
            self._finish_request(item, outcome, why, now, req.task)

    def _finish_request(self, item: PendingRequest, outcome: RequestOutcome, why: str, now: datetime,
                        task: str | None, alert: bool = True) -> None:
        req = item.request
        if req is not None and req.request_id not in self._consumed_ids:
            self._consumed_ids = (self._consumed_ids + [req.request_id])[-MAX_CONSUMED_IDS:]
        try:
            self.inbox.complete(item, outcome, why, now)
        except OSError:
            logger.exception("could not move request %s to done", item.path.name)
        if not alert:
            return
        event = AdminEvent.REQUEST_APPLIED if outcome is RequestOutcome.APPLIED else AdminEvent.REQUEST_REJECTED
        self._pending_events.append((task, event, {"request": item.raw, "outcome": str(outcome), "reason": why}))
        who = f"{req.operator}: {req.kind}" if req is not None else item.path.name
        self._alert(AlertLevel.INFO if outcome is RequestOutcome.APPLIED else AlertLevel.WARNING,
                    f"REQUEST {outcome.value.upper()}", f"{who} ({task or 'n/a'}): {why}")

    def _persist(self, now: datetime, results: dict[str, TaskResult]) -> None:
        """State first, then each task's output (spec §7.1 step 5). A failed output write keeps the old file, which
        ages toward stale: for the directive that falls back to NO_ENTRY_OPS at the readers, never to trade freely."""
        self._slots[HOST_SLOT] = {"consumed_request_ids": list(self._consumed_ids)}
        try:
            save_state(self.state_path, self._slots, now)
        except OSError as exc:
            self._write_failed(now, "state.json", exc)
        for task in self.config.tasks:
            result = results.get(task.kind.value)
            if result is None:
                continue
            try:
                atomic_write_json(self.admin_dir / task.output_name, result.output)
            except (OSError, ValueError) as exc:
                self._write_failed(now, task.output_name, exc)

    def _write_failed(self, now: datetime, name: str, exc: Exception) -> None:
        logger.error("write of %s failed: %s", name, exc)
        self._pending_events.append((None, AdminEvent.WRITE_FAILED, {"file": name, "error": str(exc)}))
        last = self._last_write_alert
        if last is None or (now - last).total_seconds() >= self.config.clock.valid_for_s:
            self._last_write_alert = now
            self._alert(AlertLevel.CRITICAL, "WRITE FAILED", f"{name}: {exc} (alerted at most once per valid_for_s)")

    def _audit_cycle(self, now: datetime, report: CycleReport, results: dict[str, TaskResult]) -> None:
        sequence, episode = self._audit_ids
        for task, event, fields in self._pending_events:
            self.audit.write(event, task=task, sequence=sequence, episode=episode, now=now, **fields)
        self.audit.write(AdminEvent.CYCLE, sequence=sequence, episode=episode, now=now, degraded=report.degraded,
                         degraded_reason=report.degraded_reason, failed_tasks=report.failed_tasks,
                         tasks={k: r.summary for k, r in results.items()})

    def _report_book_actions(self, ports: dict[str, _TaskActionPort], alerts: list[Alert]) -> None:
        """One alert per task per cycle listing the actions that happened. Failures are the task's to alert, under
        its own throttle (e.g. §12.3: a failing cancel is alerted once)."""
        for kind, port in ports.items():
            done = [r for r in port.records if r["outcome"] in (BookActionOutcome.DONE, BookActionOutcome.SKIPPED_FILLED,
                                                                 BookActionOutcome.SKIPPED_CLOSED)]
            if not done:
                continue
            lines = [f"{r['outcome']}: {r['action']} #{r['ticket']} {r['symbol']} {r['volume']}L magic {r['magic']} "
                     f"({r['reason']})" for r in done]
            alerts.append(Alert(AlertLevel.WARNING, f"BOOK ACTIONS: {kind}", "\n".join(lines)))

    def _alert(self, level: AlertLevel, title: str, body: str) -> None:
        log = logger.warning if level is not AlertLevel.INFO else logger.info
        log("[%s] %s: %s", self.identity.account, title, body)
        if self.notifier is None:
            return
        try:
            self.notifier.send_alert(level, f"{title} [{self.identity.account}]", body)
        except Exception:
            logger.exception("alert delivery failed")
