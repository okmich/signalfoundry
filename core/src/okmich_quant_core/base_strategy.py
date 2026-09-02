import json
import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, final

from .closed_trade import ClosedTrade, CloseReason
from .config import StrategyConfig
from .logging import BaseEventLogger, BarOutcome, JsonlEventLogger, LogBinding, LogicalSystemIdentity, RunnerIdentity
from .notification.base import BaseNotifier
from .signal import BaseSignal

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Wall-clock stamp for reconciliation bookkeeping. Deliberately UTC and never the broker clock: these are
    observation timestamps for the local record, not bar labels, and must stay comparable across brokers."""
    return datetime.now(timezone.utc)


#: Names a subclass MUST NOT override — the inference-log floor lives behind them (§5.1), and the
#: closed-position observation seam behind ``sync_positions``.
_SEALED_NAMES = frozenset({"run", "_emit_bar_record", "bind_runner_identity", "sync_positions"})


def _extract_tier1(ctx) -> dict:
    """Pull the contract's known Tier 1 keys out of a ``get_signal_context()`` result.

    Scalars are coerced (``direction``→int, ``confidence``/``bar_close``→float); ``label_bar_ts`` is passed through for
    the record factory to ISO-normalise. The free-form ``features``/ ``extras`` are serialise-tested here (§9) so a
    non-JSON value surfaces as a Tier 1 failure rather than a write-time crash deep in the logger.
    """
    known: dict = {}
    if ctx.get("direction") is not None:
        known["direction"] = int(ctx["direction"])
    if ctx.get("confidence") is not None:
        known["confidence"] = float(ctx["confidence"])
    if ctx.get("bar_close") is not None:
        known["bar_close"] = float(ctx["bar_close"])
    if "label_bar_ts" in ctx:
        known["label_bar_ts"] = ctx["label_bar_ts"]
    if "features" in ctx:
        known["features"] = dict(ctx["features"])
    if "extras" in ctx:
        known["extras"] = dict(ctx["extras"])
    # Serialise-test ALL Tier 1 content in isolation with allow_nan=False so non-finite floats (NaN/Inf —
    # common from indicators on warm-up bars, in features/extras AND the scalar confidence/bar_close)
    # surface as a Tier 1 failure rather than writing the bare ``NaN`` / ``Infinity`` tokens that are
    # invalid JSON for strict downstream readers (§8/§9).
    json.dumps({"features": known.get("features", {}), "extras": known.get("extras", {}),
                "confidence": known.get("confidence"), "bar_close": known.get("bar_close")}, allow_nan=False)
    return known


class BaseStrategy(ABC):
    """Template for a live trading strategy + the un-bypassable inference-log seam (LOGGING_CONTRACT §5).

    ``run()`` is the **sealed** per-bar entrypoint: it owns the Tier 0 ``bar`` heartbeat that fires
    on every new bar regardless of what the developer wrote inside ``on_new_bar()``. Subclasses
    implement the abstract hooks (``on_new_bar``/``is_new_bar``) and MAY enrich Tier 1 via their
    signal's ``get_signal_context()``; they MUST NOT override ``run()`` or the emission helpers —
    :meth:`__init_subclass__` rejects that at class-definition time.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in _SEALED_NAMES:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} may not override sealed BaseStrategy.{name}() — the inference-log floor "
                    f"(LOGGING_CONTRACT §5.1) and the closed-position observation seam live there. Implement "
                    f"on_new_bar()/is_new_bar() for per-bar logic, or manage_positions() for position work."
                )

    def __init__(self, config: StrategyConfig, signal: BaseSignal, notifier: Optional[BaseNotifier] = None,
                 *args, timeframe_minutes: Optional[int] = None,
                 inference_logger: Optional[BaseEventLogger] = None, log_base=None, **kwargs):
        self.strategy_config = config
        self.signal_generator = signal
        self.notifier = notifier
        self.args = args
        self.kwargs = kwargs
        self.latest_run_dt = None
        self.previous_run_dt = None
        self.prev_position_chk_dt = None
        self.open_position_count = 0
        #: key -> {first_seen, last_seen, close_intent}. The book as this strategy last observed it;
        #: a key that disappears from an observation is a position that left the book.
        self._open_trades: dict[str, dict[str, Any]] = {}
        #: key -> {tracked, vanished_at}. Positions that have left the book but whose close the broker cannot
        #: yet describe. Held here, unannounced, until it resolves or the grace period runs out.
        self._pending_closes: dict[str, dict[str, Any]] = {}
        self._warned_legacy_manage_positions = False

        # Fail-closed inference logging (§5): the logical identity (and therefore the file path)
        # is fully known here, so a default logger is always constructible even when the developer
        # injects none. The runner identity is bound later, at startup (bind_runner_identity).
        tf_min = timeframe_minutes if timeframe_minutes is not None else self._coerce_timeframe_minutes(config.timeframe)
        if not isinstance(tf_min, int) or tf_min <= 0:
            raise ValueError(
                f"timeframe_minutes must be a positive integer of minutes (got {tf_min!r} for "
                f"{config.name!r}/{config.symbol!r}). A broker base class MUST pass an explicit minute count "
                f"(MT5 number_of_minutes_in_timeframe, IB bar_size_to_minutes); a raw MT5 timeframe constant or "
                f"a non-positive value is rejected so the envelope, path, and asof_bar_ts stay consistent.")
        logical = LogicalSystemIdentity(strategy=config.name, symbol=config.symbol, timeframe_minutes=tf_min)
        logger_impl = inference_logger if inference_logger is not None else JsonlEventLogger(logical, log_base=log_base)
        self._log_binding = LogBinding(logical, logger_impl, order_tag=getattr(config, "magic", None))

    @staticmethod
    def _coerce_timeframe_minutes(timeframe) -> int:
        """Best-effort timeframe→minutes fallback for when a broker base class did not supply it.

        Broker subclasses MUST pass ``timeframe_minutes`` (MT5 ``number_of_minutes_in_timeframe``,
        IB ``bar_size_to_minutes``) — a raw MT5 timeframe constant is NOT minutes. This only covers
        configs whose ``timeframe`` is already integer minutes (e.g. test doubles); otherwise 0.
        """
        try:
            return int(timeframe)
        except (TypeError, ValueError):
            logger.warning("BaseStrategy: could not coerce timeframe %r to minutes; using 0. "
                           "A broker base class should pass timeframe_minutes explicitly.", timeframe)
            return 0

    @property
    def log_binding(self) -> LogBinding:
        """The strategy's two-phase identity/logger holder (used by the dispatch layer + runner)."""
        return self._log_binding

    @final
    def bind_runner_identity(self, runner: RunnerIdentity, *, runner_strategy: Optional[str] = None) -> None:
        """Complete the inference-log envelope with the runner identity (called once at startup, §5).

        ``runner_strategy`` is the runner-root strategy the runner applies once it knows the dispatch type
        (e.g. ``<strategy>-multi`` for a MultiTrader); it re-points this strategy's logical identity and
        inference path before the first bar (LOGGING_CONTRACT §7.1/§10)."""
        self._log_binding.bind(runner, strategy_override=runner_strategy)

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> Optional[list[dict]]:
        """Run position management and return the CURRENTLY OPEN positions. Default: no-op.

        The return value is an observation of the book, and the caller diffs successive observations to detect
        positions that left it (:meth:`sync_positions`). That makes the empty-vs-unknown distinction load-bearing:

        * ``list``  - the broker was queried successfully; these are the open positions (``[]`` = genuinely flat)
        * ``None``  - the book was NOT observed this call (debounced, disconnected, not implemented)

        Returning ``[]`` for "could not query" is the fail-OPEN bug that ``get_positions`` already refuses to
        commit at the MT5 layer: it reads as "every position closed" and fires phantom exits for the whole book
        during a terminal hiccup. When in doubt, return ``None`` — an unobserved book costs one cycle of latency,
        a falsely empty one corrupts the trade record.

        :param run_dt:   - datetime this call was initiated
        :param flag:bool - indicates this was called on a new bar
        """
        return None

    @final
    def sync_positions(self, run_dt: datetime, flag: bool = False) -> Optional[list[dict]]:
        """Sealed dispatch entrypoint: run position management, then reconcile what left the book.

        Sealed rather than merged into :meth:`manage_positions` because ``manage_positions`` is the method every
        broker base and strategy already overrides. Putting the diff inside it would mean any override that
        forgets ``super()`` silently loses close detection — the failure being invisible is exactly what makes it
        dangerous. Here the dispatch layer calls the sealed wrapper and the overridable hook keeps its old shape.

        A subclass that has not been migrated to return a list is not an error: it simply never observes the book,
        so reconciliation is skipped and its behaviour is unchanged.
        """
        positions = self.manage_positions(run_dt, flag)
        if positions is None:
            return None
        if not isinstance(positions, list):
            # Pre-migration subclass still returning a count. Warn ONCE per strategy - this runs every few
            # seconds, and a per-call warning would bury the log it is trying to be visible in.
            if not self._warned_legacy_manage_positions:
                self._warned_legacy_manage_positions = True
                logger.warning(
                    "%s.manage_positions() returned %s, not a list of open positions — closed-trade "
                    "reconciliation is DISABLED for %s/%s. Return the open positions to enable it.",
                    type(self).__name__, type(positions).__name__, self.strategy_config.name,
                    self.strategy_config.symbol)
            return None
        self.open_position_count = len(positions)
        try:
            self.observe_open_positions(positions)
        except Exception:
            # Reconciliation is bookkeeping; position MANAGEMENT already ran above. A fault in describing what
            # left the book must not take down the sweep that protects what is still in it — but it is a defect,
            # so it is logged at exception level rather than swallowed.
            logger.exception("%s/%s: closed-position reconciliation failed",
                             self.strategy_config.name, self.strategy_config.symbol)
        return positions

    # ------------------------------------------------------------------ closed-position reconciliation
    def _position_key(self, position: dict) -> Optional[str]:
        """Broker-neutral identity for one open position. Override where the broker's key is not ``ticket``.

        MT5 hands out a unique per-position ticket. IB has no such thing — a position is identified by contract
        (``conId``) within an account, which is why this is a hook and not a field read.
        """
        for candidate in ("ticket", "conId", "con_id", "position_id"):
            value = position.get(candidate)
            if value is not None:
                return str(value)
        return None

    def observe_open_positions(self, positions: list[dict]) -> list[ClosedTrade]:
        """Record the currently-open book and report anything that left it since the last observation.

        Public because the trigger is broker-specific even though the logic is not: the polled MT5 path reaches
        this through :meth:`sync_positions`, while the event-driven IB path calls it from its fill handler. Both
        get the same diff, neither is forced into the other's execution model.

        MUST NOT be called with a book the broker failed to return — see :meth:`manage_positions`.
        """
        seen: dict[str, dict] = {}
        for position in positions:
            key = self._position_key(position)
            if key is None:
                logger.warning("%s: open position without a usable identity, cannot track it: %r",
                               self.strategy_config.symbol, position)
                continue
            seen[key] = position

        # A key that comes BACK was never closed. Retract the pending close and restore the original tracking
        # record — rebuilding it from this observation would reset first_seen and discard the close intent.
        for key in [k for k in self._pending_closes if k in seen]:
            logger.info("%s: position %s is back in the book before its close resolved; close retracted",
                        self.strategy_config.symbol, key)
            self._open_trades[key] = self._pending_closes.pop(key)["tracked"]

        for key, position in seen.items():
            tracked = self._open_trades.get(key)
            if tracked is None:
                self._open_trades[key] = {"first_seen": _utc_now(), "last_seen": position, "close_intent": None}
            else:
                tracked["last_seen"] = position

        closed: list[ClosedTrade] = self._flush_pending_closes()
        for key in [k for k in self._open_trades if k not in seen]:
            trade = self._settle_vanished_position(key, self._open_trades.pop(key))
            if trade is not None:
                closed.append(trade)
        return closed

    #: How long to keep re-asking the broker about a position that has left the book before announcing it
    #: unresolved. Zero (the default) means announce immediately: correct wherever resolution is a one-shot
    #: payload already in hand, where a retry cannot learn anything the first attempt did not.
    #:
    #: A broker whose source is an eventually-consistent QUERY must override this. MT5 removes a position from
    #: ``positions_get`` and writes its deal to history as two separate events, so a sweep landing between them
    #: resolves nothing — and announcing then reports a real winner as a flat +0.00 that is never corrected,
    #: because the tracking record is gone by the time the deal shows up.
    #:
    #: INVARIANT for any broker setting this above zero: a position key must not be REUSED while a close on
    #: that key is still pending. MT5 tickets are unique per position, so this holds. IB's conId identifies a
    #: contract rather than a position and is reused on re-entry — which is safe only because IB resolves from
    #: a fill already in hand and therefore leaves this at zero. Raising it for IB needs that settled first.
    _CLOSE_RESOLUTION_GRACE_SECONDS: float = 0.0

    def _settle_vanished_position(self, key: str, tracked: dict) -> Optional[ClosedTrade]:
        """Announce a departed position, or park it for another try. Returns the trade only if announced."""
        trade = self._build_closed_trade(key, tracked)
        if trade is None:
            return None
        if trade.resolved or self._CLOSE_RESOLUTION_GRACE_SECONDS <= 0:
            self._on_position_closed(trade)
            return trade
        logger.info("%s: position %s left the book but the broker cannot describe it yet; holding up to %.0fs",
                    self.strategy_config.symbol, key, self._CLOSE_RESOLUTION_GRACE_SECONDS)
        self._pending_closes[key] = {"tracked": tracked, "vanished_at": _utc_now()}
        return None

    def _flush_pending_closes(self, force: bool = False) -> list[ClosedTrade]:
        """Re-attempt held closes; announce the ones that resolved and the ones that have waited long enough.

        ``force`` drains everything still held regardless of how long it has waited. Used at shutdown: holding
        a close is only ever a bet that the next observation will resolve it, and at teardown there is no next
        observation — so the bet has to be settled rather than abandoned with the process.
        """
        closed: list[ClosedTrade] = []
        for key in list(self._pending_closes):
            pending = self._pending_closes[key]
            trade = self._build_closed_trade(key, pending["tracked"])
            if trade is not None and trade.resolved:
                del self._pending_closes[key]
                self._on_position_closed(trade)
                closed.append(trade)
                continue
            waited = (_utc_now() - pending["vanished_at"]).total_seconds()
            if not force and waited < self._CLOSE_RESOLUTION_GRACE_SECONDS:
                continue
            del self._pending_closes[key]
            if trade is not None:
                logger.warning("%s: giving up on resolving position %s after %.0fs; reporting it unresolved",
                               self.strategy_config.symbol, key, waited)
                self._on_position_closed(trade)
                closed.append(trade)
        return closed

    def register_open_position(self, position: dict) -> None:
        """Track a position at the moment it is opened, before any sweep could have seen it.

        Without this, a position that opens and closes between two observations is never in the tracked map and
        its exit is invisible. Entry is the one moment the system knows about a position with certainty, so it is
        the right place to start tracking rather than waiting to notice it.
        """
        key = self._position_key(position)
        if key is None:
            logger.warning("%s: opened position without a usable identity, cannot track it: %r",
                           self.strategy_config.symbol, position)
            return
        self._open_trades.setdefault(key, {"first_seen": _utc_now(), "last_seen": position, "close_intent": None})

    def note_close_intent(self, key, reason: str) -> None:
        """Record that THIS system asked for a close, without announcing it.

        The announcement is deliberately left to the reconciler. Announcing here would report the unrealised P/L
        snapshotted before the close request instead of the realised fill, and would put two emitters on the same
        event — the duplicate that has to be deduplicated somewhere. One emitter, fed by broker truth, costs at
        most one sweep of latency and removes the need for dedup state entirely.
        """
        self._intent_record(key)["close_intent"] = reason

    def clear_close_intent(self, key) -> None:
        """Withdraw an intent recorded for a close that did not actually go through.

        Intent is recorded BEFORE the request, because a close that succeeds can be reconciled before the call
        returns. So a failed request leaves a claim that we closed a position we did not — and that claim does
        not expire. If a human later closes that same position in the terminal, the stale intent relabels their
        MANUAL close as ours, which is the exact inversion of the mislabelling intent exists to prevent.
        """
        record = self._open_trades.get(str(key)) or (self._pending_closes.get(str(key)) or {}).get("tracked")
        if record is not None:
            record["close_intent"] = None

    def _intent_record(self, key) -> dict:
        """The tracking record intent attaches to, creating one if this position is not tracked yet.

        A close can be requested on a position that was discovered mid-sweep or adopted after a restart, and on
        one already parked awaiting resolution. Dropping the intent in those cases does not merely lose a label:
        it lets the broker's coarse "closed by client" stand and reports OUR close as a human's.
        """
        key = str(key)
        tracked = self._open_trades.get(key)
        if tracked is not None:
            return tracked
        pending = self._pending_closes.get(key)
        if pending is not None:
            return pending["tracked"]
        tracked = {"first_seen": _utc_now(), "last_seen": {}, "close_intent": None}
        self._open_trades[key] = tracked
        return tracked

    def _build_closed_trade(self, key: str, tracked: dict) -> Optional[ClosedTrade]:
        """Resolve a vanished position into a :class:`ClosedTrade`, falling back to its last-seen state."""
        last_seen = tracked.get("last_seen") or {}
        intent = tracked.get("close_intent")
        try:
            resolved = self.resolve_closed_trade(key, last_seen)
        except Exception as e:
            logger.error("%s: could not resolve closed position %s (%s); reporting it unresolved",
                         self.strategy_config.symbol, key, e)
            resolved = None
        if resolved is not None:
            if not intent:
                return resolved
            # The intent is ALWAYS kept, but it only overrides a broker reason that is too coarse to be useful
            # ("closed by client" — only we know it was the CTL flip that asked). A take-profit or stop-loss is
            # a fact about how the trade ended and outranks intent: relabelling it would erase the very
            # distinction the exit record exists to capture. Both can be true — we asked, and the target filled
            # first — so the request is still recorded alongside the outcome.
            if resolved.reason in (CloseReason.STRATEGY, CloseReason.MANUAL, CloseReason.UNKNOWN):
                return replace(resolved, reason=CloseReason.STRATEGY, strategy_reason=intent)
            return replace(resolved, strategy_reason=intent)
        return ClosedTrade(
            key=key, symbol=self.strategy_config.symbol, magic=getattr(self.strategy_config, "magic", None),
            reason=CloseReason.STRATEGY if intent else CloseReason.UNKNOWN, strategy_reason=intent,
            # abs(): IB's "position" is SIGNED quantity, so a short would otherwise report a negative volume
            # and flip the sign of anything that aggregates it. Direction belongs in last_seen, not in size.
            volume=abs(float(last_seen.get("volume") or last_seen.get("position") or 0.0)),
            entry_price=float(last_seen.get("price_open") or last_seen.get("avg_cost") or 0.0),
            opened_at=tracked.get("first_seen"), closed_at=_utc_now(), last_seen=last_seen, resolved=False)

    def resolve_closed_trade(self, key: str, last_seen: dict) -> Optional[ClosedTrade]:
        """Broker hook: describe a position that has left the book. Return ``None`` if it cannot be resolved.

        Implemented per broker because the source differs — MT5 must query deal history, IB already holds the
        fill. Returning ``None`` is not a failure path: the caller still reports the close, marked unresolved.
        """
        return None

    def _on_position_closed(self, trade: ClosedTrade) -> None:
        """The single place a closed position is announced. Sole caller of ``notifier.on_trade_closed``.

        Never raises: reconciliation runs inside the position sweep, and a notifier or logging fault must not
        take down position management with it.
        """
        try:
            if trade.resolved:
                logger.info("%s", trade.describe())
            else:
                logger.warning("%s", trade.describe())
            if self.notifier:
                self.notifier.on_trade_closed(symbol=trade.symbol, ticket=trade.key, profit=trade.net_profit,
                                              price=trade.exit_price, reason=trade.reason.value)
        except Exception:
            logger.exception("%s: failed to report closed position %s", self.strategy_config.symbol, trade.key)

    @abstractmethod
    def on_new_bar(self):
        """Run the complete per-bar strategy logic (fetch data, generate signals, manage/open positions)."""
        pass

    @abstractmethod
    def is_new_bar(self, run_dt: datetime) -> bool:
        """Return True if ``run_dt`` represents a new bar for this strategy's timeframe (broker-specific)."""
        pass

    def _derive_asof_bar_ts(self, run_dt: datetime) -> datetime:
        """Framework-owned **open/label timestamp of the last COMPLETE bar** before ``run_dt`` (§5.3).

        This is the broker's bar-**label** (start-of-bar) timestamp, **not** the bar's close: brokers
        label bars by open time, so the just-completed 11:55–12:00 candle (acted on at run_dt≈12:00) is
        identified as ``11:55`` — that is the value emitted, and it's how the record names the bar this
        cycle acted on. (The formula floors ``run_dt`` to the timeframe and steps back one bar.)

        Derived purely from ``run_dt`` + the timeframe (no data fetch), so a ``bar`` record — including
        ``outcome=error`` — is fully populated even if ``on_new_bar()`` raises before fetching anything.
        A timezone-naive ``run_dt`` (the polled MT5 path emits ``datetime.now()``) is interpreted as
        **system-local** and converted to UTC. It is a monotonic per-bar **liveness** clock (ops keys
        wedged-vs-live off its +1-bar-per-bar progression); it equals the broker's own bar-label timestamp
        only when the runner host's local timezone matches the broker server timezone (risk F1). The
        accurate signal-aligned label is the Tier 1 ``label_bar_ts`` the developer supplies, not this field.
        """
        dt = run_dt
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.astimezone()  # naive → aware in the system timezone
        dt_utc = dt.astimezone(timezone.utc)
        tf = self._log_binding.logical.timeframe_minutes or 1
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        total_minutes = int((dt_utc - epoch).total_seconds()) // 60
        last_complete_idx = (total_minutes // tf) - 1
        return epoch + timedelta(minutes=last_complete_idx * tf)

    def _collect_tier1(self) -> tuple[dict, Optional[str]]:
        """Best-effort Tier 1 from ``get_signal_context()``, isolated so a failure can't drop Tier 0 (§8).

        Returns ``(fields, tier1_error)``: ``fields`` is the subset of known Tier 1 keys to populate;
        ``tier1_error`` is a non-null annotation string when capture/serialisation failed.
        """
        signal = self.signal_generator
        if signal is None:
            return {}, None
        try:
            ctx = signal.get_signal_context()
        except Exception as exc:
            return {}, f"get_signal_context() raised: {exc!r}"
        if not ctx:
            return {}, None
        try:
            return _extract_tier1(ctx), None
        except Exception as exc:
            return {}, f"Tier 1 capture failed (Tier 0 preserved): {exc!r}"

    def _write_record(self, record) -> None:
        """Hand a built record to the logger (guarded by bind state + never crashing the loop)."""
        binding = self._log_binding
        if not binding.is_bound:
            logger.error("inference log not bound for %s — %s record skipped",
                         binding.logical.logical_system_id, record.envelope.event)
            return
        try:
            binding.logger.write(record)
        except Exception:
            logger.exception("inference logger write failed for %s", binding.logical.logical_system_id)

    @final
    def _emit_bar_record(self, *, asof_bar_ts, outcome: BarOutcome) -> None:
        """Build + write one ``bar`` record. Tier 0 is guaranteed; Tier 1 degrades in isolation (§8).

        Tier 1 is collected ONLY for a cycle that completed cleanly (``ok``). A ``skipped_disabled``
        bar never ran ``on_new_bar``; an ``error`` bar ran it but it RAISED, so ``get_signal_context()``
        may return the *previous* bar's context — reporting that as current would be a stale-data lie.
        For both, the ``bar`` carries the Tier 0 floor only (no behavioural content).
        """
        binding = self._log_binding
        if not binding.is_bound:
            logger.error("inference log not bound for %s — bar heartbeat skipped (runner must bind at startup)",
                         binding.logical.logical_system_id)
            return
        factory = binding.system_factory()
        if outcome is BarOutcome.OK:
            tier1, tier1_error = self._collect_tier1()
        else:
            tier1, tier1_error = {}, None
        try:
            record = factory.bar(asof_bar_ts=asof_bar_ts, outcome=outcome, bar_close=tier1.get("bar_close"),
                                 label_bar_ts=tier1.get("label_bar_ts"), direction=tier1.get("direction"),
                                 confidence=tier1.get("confidence"), features=tier1.get("features", {}),
                                 extras=tier1.get("extras", {}), tier1_error=tier1_error)
        except Exception as exc:
            # Tier 1 content broke record construction — degrade to a Tier 0-only heartbeat (§8).
            record = factory.bar(asof_bar_ts=asof_bar_ts, outcome=outcome,
                                 tier1_error=f"Tier 1 build failed (Tier 0 preserved): {exc!r}")
        self._write_record(record)

    def emit_circuit_breaker_tripped(self, *, consecutive_errors: int, last_error: Optional[str] = None) -> None:
        """Emit a ``circuit_breaker_tripped`` record (dispatch layer / IB breaker, §7.3)."""
        if not self._log_binding.is_bound:
            logger.error("inference log not bound for %s — circuit_breaker_tripped skipped",
                         self._log_binding.logical.logical_system_id)
            return
        self._write_record(self._log_binding.system_factory().circuit_breaker_tripped(
            consecutive_errors=consecutive_errors, last_error=last_error))

    def emit_strategy_reenabled(self, *, reason: Optional[str] = None) -> None:
        """Emit a ``strategy_reenabled`` record (dispatch layer / IB breaker, §7.3)."""
        if not self._log_binding.is_bound:
            return
        self._write_record(self._log_binding.system_factory().strategy_reenabled(reason=reason))

    def emit_skipped_disabled_bar(self, run_dt) -> None:
        """Emit a ``bar`` with ``outcome=skipped_disabled`` for a disabled, skipped strategy (§5.3/§7.3)."""
        if not self._log_binding.is_bound:
            return
        self._emit_bar_record(asof_bar_ts=self._derive_asof_bar_ts(run_dt), outcome=BarOutcome.SKIPPED_DISABLED)

    @final
    def run(self, run_dt: datetime):
        """Sealed template (§5): dup-guard → manage positions → on a new bar, emit the Tier 0 heartbeat.

        On a new bar, ``on_new_bar()`` is wrapped: success emits ``outcome=ok``; an exception emits
        ``outcome=error`` and is then **re-raised** so the dispatch layer's ``StrategyHealth`` /
        circuit-breaker bookkeeping is unchanged. The heartbeat and the breaker are complementary.
        """
        # Prevent duplicate runs within 1 second
        if self.previous_run_dt and abs((run_dt - self.previous_run_dt).total_seconds()) <= 1:
            return

        is_new = self.is_new_bar(run_dt)
        if not is_new:
            # Intra-bar tick: manage positions only — there is no new bar this tick, so no heartbeat.
            self.sync_positions(run_dt, False)
            self.previous_run_dt = run_dt
            return

        # New bar: the WHOLE cycle — position management AND on_new_bar() — is the heartbeat. A failure
        # in EITHER must still produce the Tier 0 outcome=error record (the contract's per-new-bar floor),
        # and then re-raise so the dispatch layer's StrategyHealth/breaker bookkeeping is unaffected.
        self.latest_run_dt = run_dt
        asof_bar_ts = self._derive_asof_bar_ts(run_dt)
        try:
            self.sync_positions(run_dt, True)
            self.on_new_bar()
            outcome = BarOutcome.OK
        except Exception:
            outcome = BarOutcome.ERROR
            raise  # bare re-raise preserves the original traceback exactly (live-failure diagnosis)
        finally:
            # The whole cycle is the heartbeat — emit the Tier 0 outcome on BOTH the ok and error paths.
            # _emit_bar_record is self-guarding (never raises), so emitting inside finally cannot mask the
            # in-flight exception; the bare ``raise`` above still propagates for StrategyHealth/breaker.
            self._emit_bar_record(asof_bar_ts=asof_bar_ts, outcome=outcome)
            self.previous_run_dt = run_dt

    def cleanup(self):
        """Teardown on shutdown: flush/close the inference logger and the notifier.

        Called by ``MultiTrader.close()`` / ``Trader.close()``. Subclasses overriding this SHOULD call ``super().cleanup()``.
        The inference logger is closed FIRST and independently of the notifier: its close() drains the bounded bar queue
        (the ops-critical heartbeats), so a notifier failure must never prevent that drain — each close is isolated.
        """
        # Settle held closes FIRST, while the notifier is still open. A close held for resolution is
        # unannounced by design, so a runner stopping while one is held would drop the trade entirely —
        # the record silently missing a position rather than describing it poorly.
        try:
            self._flush_pending_closes(force=True)
        except Exception:
            logger.exception("error settling held closes for %s", self._log_binding.logical.logical_system_id)
        try:
            self._log_binding.logger.close()
        except Exception:
            logger.warning("error closing inference logger for %s", self._log_binding.logical.logical_system_id)
        if self.notifier:
            try:
                self.notifier.close()
            except Exception:
                logger.warning("error closing notifier for %s", self._log_binding.logical.logical_system_id)
