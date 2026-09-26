"""The account-directive guard every trading system carries (ACCOUNT_ADMIN_SPEC §10), broker-neutral.

:class:`DirectiveGuard` is composed into :class:`~okmich_quant_core.base_strategy.BaseStrategy`; the sealed
``BaseStrategy.guard_entry`` and ``BaseStrategy.enforce_account_directive`` delegate to it, and a broker base class
supplies the book through five hooks (``_guard_terminal_identity``, ``_guard_own_pending``, ``_guard_cancel_pending``,
``_guard_own_positions``, ``_guard_close_position``). A broker base that does not implement them has no guard, and
must not run on a governed account (spec §10.1).

Reading: at each entry check and each sweep. Strategies of one runner share a read made within the same second (they
are dispatched on the same tick; the Admin never writes on those ticks). On an ungoverned account the cost is one
failed file lookup. Nothing here raises into the strategy: closing is never blocked (invariant 2).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .account import deployment_account
from .account_admin.directive import DirectiveReading, directive_path, read_directive, _resolve_live_base
from .account_admin.enums import AccountDirective, DirectiveSource
from .logging import GuardedOp, GuardOutcome

if TYPE_CHECKING:
    from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

SHARED_READ_S = 1.0
FALLBACK_SOURCES = frozenset({DirectiveSource.STALE, DirectiveSource.INVALID, DirectiveSource.WRONG_ACCOUNT})
_UNSET = object()


@dataclass(frozen=True)
class GuardPending:
    """One of the strategy's own resting pending orders, as the broker hook reports it."""

    ticket: int
    detail: str   # size/price/type for the text log and the alert (never the structured record)


@dataclass(frozen=True)
class GuardPosition:
    ticket: int
    long: bool
    detail: str   # size/price/P&L for the text log and the alert


class _ProcessReader:
    """One per process: the shared one-second read, and the runner-level fallback alert state (spec §10.3: a dead Admin
    produces one alert per runner, not one per sleeve)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cached: tuple[float, Any, DirectiveReading] | None = None
        self._account: Any = _UNSET
        self.fallback_alerted: DirectiveSource | None = None

    def account(self) -> str | None:
        if self._account is _UNSET:
            self._account = deployment_account()
        return self._account

    def read(self, identity: Callable[[], tuple[int, str] | None], clock: Callable[[], datetime]) -> DirectiveReading:
        with self._lock:
            now_mono = time.monotonic()
            if self._cached is not None and now_mono - self._cached[0] < SHARED_READ_S:
                return self._cached[2]
            reading = self._fresh(identity, clock())
            self._cached = (now_mono, None, reading)
            return reading

    def _fresh(self, identity, now: datetime) -> DirectiveReading:
        account = self.account()
        base = _resolve_live_base(None)
        if base is None:
            return read_directive(account=account, login=None, server=None, now=now)   # absent; no terminal query
        if account is not None and not directive_path(base, account).exists():
            # Ungoverned: one failed lookup, no terminal query.
            return DirectiveReading(AccountDirective.ALL_OPS, DirectiveSource.ABSENT, path=directive_path(base, account),
                                    detail="no directive file: this account is not governed")
        try:
            ident = identity()
        except Exception:
            logger.exception("directive guard: terminal identity query failed")
            ident = None
        login, server = ident if ident else (None, None)
        return read_directive(account=account, login=login, server=server, now=now)

    def invalidate(self) -> None:
        """Drop the shared read so the next check reads the file again."""
        with self._lock:
            self._cached = None

    def reset(self) -> None:
        """Forget the shared read and the runner state (tests)."""
        with self._lock:
            self._cached = None
            self._account = _UNSET
            self.fallback_alerted = None


PROCESS_READER = _ProcessReader()


def _episode_key(r: DirectiveReading) -> tuple:
    return str(r.directive), str(r.source), (r.file.episode if r.file is not None else None)


@dataclass
class _Episode:
    key: tuple
    started_utc: datetime
    directive: AccountDirective
    source: DirectiveSource
    suppressed: Counter = field(default_factory=Counter)
    alerted_ops: set = field(default_factory=set)
    failed_alerted: set = field(default_factory=set)


class DirectiveGuard:
    """Per-strategy guard state and logic."""

    def __init__(self, strategy: "BaseStrategy", clock: Callable[[], datetime] | None = None,
                 reader: _ProcessReader | None = None):
        self.strategy = strategy
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.reader = reader or PROCESS_READER
        self._episode: _Episode | None = None
        self._last: DirectiveReading | None = None

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def supported(self) -> bool:
        return bool(getattr(self.strategy, "_GUARD_SUPPORTED", False))

    def _label(self) -> str:
        cfg = self.strategy.strategy_config
        return f"{cfg.name}/{cfg.symbol}#{getattr(cfg, 'magic', '')}"

    def _alert(self, level: str, title: str, body: str) -> None:
        notifier = self.strategy.notifier
        if notifier is None:
            return
        try:
            notifier.on_account_event(title, body, level)
        except Exception:
            logger.exception("directive guard: alert failed")

    def _record_op(self, op: GuardedOp, outcome: GuardOutcome, r: DirectiveReading, signal_bar_utc=None) -> None:
        binding = self.strategy.log_binding
        if not binding.is_bound:
            return
        f = r.file
        self.strategy._write_record(binding.system_factory().account_directive_op(
            op=op, outcome=outcome, directive=str(r.directive), directive_source=str(r.source),
            directive_sequence=f.sequence if f else None, directive_episode=f.episode if f else None,
            directive_reason=f.reason if f else r.detail, signal_bar_utc=signal_bar_utc))

    # ------------------------------------------------------------------------------------------ reading
    def current(self) -> DirectiveReading:
        """One read (shared within the second), with the change bookkeeping of spec §10.2-§10.3. Never raises."""
        try:
            reading = self.reader.read(self.strategy._guard_terminal_identity, self.clock)
        except Exception as exc:
            logger.exception("directive guard: read failed")
            reading = DirectiveReading(AccountDirective.NO_ENTRY_OPS, DirectiveSource.INVALID, detail=f"guard error: {exc!r}")
        try:
            self._note(reading)
        except Exception:
            logger.exception("directive guard: bookkeeping failed")
        return reading

    def _note(self, r: DirectiveReading) -> None:
        now = self.clock()
        key = _episode_key(r)
        prev = self._episode
        if prev is not None and prev.key == key:
            self._last = r
            return
        if prev is not None:
            self._summarise(prev, now)
        self._episode = _Episode(key=key, started_utc=now, directive=r.directive, source=r.source)
        self._last = r
        self._log_change(prev, r)
        binding = self.strategy.log_binding
        if binding.is_bound:
            f = r.file
            self.strategy._write_record(binding.system_factory().account_directive_applied(
                directive=str(r.directive), directive_source=str(r.source), directive_sequence=f.sequence if f else None,
                directive_episode=f.episode if f else None, directive_reason=f.reason if f else r.detail,
                previous_directive=str(prev.directive) if prev else None,
                previous_source=str(prev.source) if prev else None))
        self._runner_fallback(r)

    def _log_change(self, prev: _Episode | None, r: DirectiveReading) -> None:
        what = f"{self._label()}: account directive {r.directive.value} ({r.source.value}) - {r.detail}"
        if r.source is DirectiveSource.WRONG_ACCOUNT:
            logger.error(what)
        elif r.source in FALLBACK_SOURCES or (prev is not None and prev.source in FALLBACK_SOURCES):
            logger.warning(what)
        else:
            logger.info(what)

    def _runner_fallback(self, r: DirectiveReading) -> None:
        reader = self.reader
        if r.source in FALLBACK_SOURCES and reader.fallback_alerted != r.source:
            reader.fallback_alerted = r.source
            self._alert("critical", f"DIRECTIVE FALLBACK: {r.source.value.upper()}",
                        f"{self._label()} and every system of this runner apply NO_ENTRY_OPS: {r.detail}")
        elif r.source not in FALLBACK_SOURCES and reader.fallback_alerted is not None:
            left = reader.fallback_alerted
            reader.fallback_alerted = None
            self._alert("info", "DIRECTIVE FALLBACK ENDED", f"{left.value} -> {r.source.value}: {r.directive.value}")

    def _summarise(self, ep: _Episode, now: datetime) -> None:
        if not ep.suppressed:
            return
        counts = ", ".join(f"{n} x {op.value}" for op, n in sorted(ep.suppressed.items(), key=lambda kv: kv[0].value))
        self._alert("info", "DIRECTIVE EPISODE SUMMARY",
                    f"{self._label()}: {ep.directive.value} ({ep.source.value}) {ep.started_utc:%Y-%m-%d %H:%M}-"
                    f"{now:%H:%M} UTC: {counts} suppressed")

    # ------------------------------------------------------------------------------------------ the two uses
    def allow_entry(self, op: GuardedOp, *, detail: str, signal_bar_utc: Any = None) -> bool:
        """Whether an entry may leave (spec §10.1 choke point). A refused one is recorded, logged with its size in the
        text log, and alerted once per op per episode."""
        if not self.supported:
            return True
        r = self.current()
        if r.directive is AccountDirective.ALL_OPS:
            return True
        ep = self._episode
        self._record_op(op, GuardOutcome.SUPPRESSED, r, signal_bar_utc)
        logger.warning("%s: %s SUPPRESSED by %s (%s): %s", self._label(), op.value, r.directive.value, r.source.value, detail)
        if ep is not None:
            ep.suppressed[op] += 1
            if op not in ep.alerted_ops:
                ep.alerted_ops.add(op)
                self._alert("warning", f"ENTRY SUPPRESSED: {op.value}",
                            f"{self._label()}: {detail}\nunder {r.directive.value} ({r.source.value}): "
                            f"{r.file.reason if r.file else r.detail}\n(later suppressions this episode are counted, "
                            f"not alerted)")
        return False

    def sweep(self) -> None:
        """Cancel own pending orders under a restrictive directive; close own positions under NO_OPS (spec §10.1)."""
        if not self.supported:
            return
        r = self.current()
        if r.directive is AccountDirective.ALL_OPS:
            return
        s = self.strategy
        try:
            pending = s._guard_own_pending()
        except Exception:
            logger.exception("%s: could not list own pending orders", self._label())
            pending = []
        for order in pending:
            self._force(GuardedOp.CANCEL_PENDING, order.ticket, order.detail, r, lambda t=order.ticket: s._guard_cancel_pending(t))
        if r.directive is not AccountDirective.NO_OPS:
            return
        try:
            positions = s._guard_own_positions()
        except Exception:
            logger.exception("%s: could not list own positions", self._label())
            positions = []
        for pos in positions:
            op = GuardedOp.CLOSE_LONG if pos.long else GuardedOp.CLOSE_SHORT
            self._force(op, pos.ticket, pos.detail, r, lambda t=pos.ticket: s._guard_close_position(t, r.directive.value))

    def _force(self, op: GuardedOp, ticket: int, detail: str, r: DirectiveReading, action: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, why = action()
        except Exception as exc:
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if ok:
            self._record_op(op, GuardOutcome.FORCED, r)
            logger.warning("%s: %s #%s FORCED by %s (%s): %s", self._label(), op.value, ticket, r.directive.value,
                           r.source.value, detail)
            self._alert("warning", f"FORCED {op.value}", f"{self._label()} #{ticket}: {detail}\nby {r.directive.value} "
                                                        f"({r.source.value})")
            return
        logger.error("%s: forced %s #%s FAILED: %s (%s)", self._label(), op.value, ticket, why, detail)
        ep = self._episode
        if ep is not None and ticket not in ep.failed_alerted:
            ep.failed_alerted.add(ticket)
            self._alert("critical", f"FORCED {op.value} FAILED", f"{self._label()} #{ticket}: {why}\n{detail}\n"
                                                                f"retried every sweep; alerted once this episode")
