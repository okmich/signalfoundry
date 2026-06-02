import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional, final

from .config import StrategyConfig
from .logging import BaseEventLogger, BarOutcome, JsonlEventLogger, LogBinding, LogicalSystemIdentity, RunnerIdentity
from .notification.base import BaseNotifier
from .signal import BaseSignal

logger = logging.getLogger(__name__)

#: Names a subclass MUST NOT override — the inference-log floor lives behind them (§5.1).
_SEALED_NAMES = frozenset({"run", "_emit_bar_record", "bind_runner_identity"})


def _extract_tier1(ctx) -> dict:
    """Pull the contract's known Tier 1 keys out of a ``get_signal_context()`` result.

    Scalars are coerced (``direction``→int, ``confidence``/``bar_close``→float); ``label_bar_ts``
    is passed through for the record factory to ISO-normalise. The free-form ``features``/
    ``extras`` are serialise-tested here (§9) so a non-JSON value surfaces as a Tier 1 failure
    rather than a write-time crash deep in the logger.
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
                    f"{cls.__name__} may not override sealed BaseStrategy.{name}() — the inference-log "
                    f"floor lives there (LOGGING_CONTRACT §5.1). Implement on_new_bar()/is_new_bar() instead."
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

        # Fail-closed inference logging (§5): the logical identity (and therefore the file path)
        # is fully known here, so a default logger is always constructible even when the developer
        # injects none. The runner identity is bound later, at startup (bind_runner_identity).
        tf_min = timeframe_minutes if timeframe_minutes is not None else self._coerce_timeframe_minutes(config.timeframe)
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
    def bind_runner_identity(self, runner: RunnerIdentity) -> None:
        """Complete the inference-log envelope with the runner identity (called once at startup, §5)."""
        self._log_binding.bind(runner)

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        """Run position management and return the number of open positions. Default: no-op.

        :param run_dt:   - datetime this call was initiated
        :param flag:bool - indicates this was called on a new bar
        """
        return 0

    @abstractmethod
    def on_new_bar(self):
        """Run the complete per-bar strategy logic (fetch data, generate signals, manage/open positions)."""
        pass

    @abstractmethod
    def is_new_bar(self, run_dt: datetime) -> bool:
        """Return True if ``run_dt`` represents a new bar for this strategy's timeframe (broker-specific)."""
        pass

    def _derive_asof_bar_ts(self, run_dt: datetime) -> datetime:
        """Framework-owned bar-close timestamp (§5.3): the last COMPLETE bar boundary before ``run_dt``.

        Derived purely from ``run_dt`` + the timeframe (no data fetch), so a ``bar`` record — including
        ``outcome=error`` — is fully populated even if ``on_new_bar()`` raises before fetching anything.
        A timezone-naive ``run_dt`` (the polled MT5 path emits ``datetime.now()``) is interpreted as
        **system-local** and converted to UTC. This is a monotonic per-bar **liveness** clock (ops keys
        wedged-vs-live off its progression); it equals the broker's own bar-label timestamp only when the
        runner host's local timezone matches the broker server timezone (risk F1). The accurate
        signal-aligned label is the Tier 1 ``label_bar_ts`` the developer supplies, not this field.
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
            self.open_position_count = self.manage_positions(run_dt, False)
            self.previous_run_dt = run_dt
            return

        # New bar: the WHOLE cycle — position management AND on_new_bar() — is the heartbeat. A failure
        # in EITHER must still produce the Tier 0 outcome=error record (the contract's per-new-bar floor),
        # and then re-raise so the dispatch layer's StrategyHealth/breaker bookkeeping is unaffected.
        self.latest_run_dt = run_dt
        asof_bar_ts = self._derive_asof_bar_ts(run_dt)
        error: Optional[Exception] = None
        try:
            self.open_position_count = self.manage_positions(run_dt, True)
            self.on_new_bar()
        except Exception as exc:
            error = exc
        self._emit_bar_record(asof_bar_ts=asof_bar_ts,
                              outcome=BarOutcome.ERROR if error is not None else BarOutcome.OK)
        self.previous_run_dt = run_dt
        if error is not None:
            raise error

    def cleanup(self):
        """Teardown on shutdown: flush/close the inference logger and the notifier.

        Called by ``MultiTrader.close()`` / ``Trader.close()``. Subclasses overriding this SHOULD call
        ``super().cleanup()``. The inference logger is closed FIRST and independently of the notifier:
        its close() drains the bounded bar queue (the ops-critical heartbeats), so a notifier failure
        must never prevent that drain — each close is isolated.
        """
        try:
            self._log_binding.logger.close()
        except Exception:
            logger.warning("error closing inference logger for %s", self._log_binding.logical.logical_system_id)
        if self.notifier:
            try:
                self.notifier.close()
            except Exception:
                logger.warning("error closing notifier for %s", self._log_binding.logical.logical_system_id)
