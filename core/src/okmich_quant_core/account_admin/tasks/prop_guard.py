"""``prop_guard``: the account policy and the directive (ACCOUNT_ADMIN_SPEC §4-§6, §8.2, §11, §12.2).

Each cycle it rolls the trading day, updates the high-water marks, trips and clears latches, applies calendar windows
and operator overrides, combines every active condition into the most restrictive directive, and checks that the book
obeys it. Its output is ``directive.json``, the one file trading systems read. It never acts on the book: the directive
asks each system to cancel and close its own orders and positions.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any, Mapping

from ..directive import DIRECTIVE_FILE, DirectiveAccount, DirectiveFile
from ..enums import (RESTRICTIVE_DIRECTIVES, AccountDirective, AdminEvent, AdminTaskKind, AlertLevel, Condition,
                     DailyBase, DealEntry, DealKind, LatchScope, RequestKind, RequestOutcome, TargetMeasure)
from ..owners import describe_owner
from ..requests import AdminRequest
from ..snapshot import Deal
from ..state import StateLoad
from ..timeutil import iso_z, parse_utc
from .base import AdminTask, Alert, TaskContext, TaskEvent, TaskResult, register_task
from .prop_guard_policy import PropPolicy, compute_levels, day_boundary, parse_prop_guard

SLOT_SCHEMA_VERSION = 1

#: Deals are scanned incrementally from a cursor; this overlap re-reads the edge so a deal booked late (MT5 writes the
#: position and its deal as separate events) is not missed. Seen tickets inside the overlap are deduplicated.
DEAL_OVERLAP_S = 300

#: The fixed conditions and the directive each imposes (spec §6.4).
_FIXED = {Condition.BALANCE_OPERATION: AccountDirective.NO_ENTRY_OPS, Condition.STATE_LOST: AccountDirective.NO_ENTRY_OPS,
          Condition.ADMIN_DEGRADED: AccountDirective.NO_ENTRY_OPS}

#: Tie-break among causes of equal restrictiveness, for `causes` order and the `reason` (most serious first).
_PRIORITY = {"max_loss": 0, "daily_loss": 1, "max_warn": 2, "daily_warn": 3, "target": 4, "balance_operation": 5,
             "state_lost": 6, "admin_degraded": 7, "override": 8}

_ENTRY_KINDS = (DealKind.BUY, DealKind.SELL)
_ENTRY_ENTRIES = (DealEntry.IN, DealEntry.INOUT)


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.2f}"


def _fresh_slot() -> dict[str, Any]:
    return {"schema_version": SLOT_SCHEMA_VERSION, "trading_day": None,
            "high_water": {"equity": None, "equity_utc": None, "balance": None}, "latches": {}, "override": None,
            "sequence": 0, "episode": 0, "directive": None, "since_utc": None, "last_cycle_utc": None,
            "deal_cursor_utc": None, "recent_deals": [], "reported": {"obedience": [], "orphans": []},
            "episode_orphans": 0, "degraded": False, "currency": None, "last_eval": None, "last_holds": {}}


def _net_after(deals: list[Deal], instant: datetime) -> float:
    return sum(d.net for d in deals if d.time_utc > instant)


def _max_balance_along(deals: list[Deal], balance_now: float) -> float:
    """The highest closed balance reached along ``deals`` (oldest first), walking back from the current balance."""
    best = balance_now
    running = balance_now
    for deal in sorted(deals, key=lambda d: d.time_utc, reverse=True):
        running -= deal.net           # balance just before this deal
        best = max(best, running)
    return best


@register_task
class PropGuardTask(AdminTask):
    kind = AdminTaskKind.PROP_GUARD
    governs_directive = True

    def __init__(self, entry: Mapping[str, Any]):
        super().__init__(entry)
        policy, problems = parse_prop_guard(entry, "prop_guard")
        if policy is None:
            raise ValueError("; ".join(problems))
        self.policy: PropPolicy = policy

    @classmethod
    def validate(cls, entry: Mapping[str, Any], prefix: str) -> list[str]:
        return parse_prop_guard(entry, prefix)[1]

    @property
    def output_name(self) -> str:
        return DIRECTIVE_FILE

    # ------------------------------------------------------------------------------------------------ the cycle
    def on_cycle(self, ctx: TaskContext) -> TaskResult:
        slot = copy.deepcopy(ctx.slot) if ctx.slot else None
        result = TaskResult(slot={}, output={})
        if slot is None:
            slot = self._start_slot(ctx, result)
        now = ctx.now
        prev_directive = AccountDirective(slot["directive"]) if slot["directive"] else None
        prev_since = parse_utc(slot["since_utc"]) if slot["since_utc"] else None

        self._handle_requests(ctx, slot, result)
        self._track_degraded(ctx, slot, result)

        holds: dict[str, str] = {}      # cause id -> evidence text, for conditions holding this cycle
        levels = None
        if not ctx.degraded:
            info = ctx.snapshot.info
            slot["currency"] = info.currency
            new_deals = self._new_deals(ctx, slot, result)
            self._roll_day(ctx, slot, result, info.balance, info.equity)
            self._update_high_water(slot, info.balance, info.equity, new_deals, now)
            self._balance_operations(slot, result, new_deals, now)
            levels = compute_levels(self.policy, float(slot["trading_day"]["base"]), float(slot["high_water"]["balance"]),
                                    float(slot["high_water"]["equity"]))
            holds = self._evaluate_levels(info.balance, info.equity, levels)
            slot["last_holds"] = dict(holds)
            self._trip_latches(slot, result, holds, now, info.balance, info.equity)
            self._check_obedience(ctx, slot, result, new_deals, prev_directive, prev_since)
            slot["last_eval"] = {"levels": levels.to_dict(),
                                 "metrics": {"balance": round(info.balance, 2), "equity": round(info.equity, 2),
                                             "day_pnl": round(info.equity - levels.daily_base, 2),
                                             "total_pnl": round(info.equity - self.policy.initial_capital, 2)}}
        else:
            # The account cannot be read: what held at the last readable cycle is presumed to hold still. A breach
            # configured without a latch must not loosen because the Admin went blind (spec §7.1: "stays stricter").
            holds = {k: f"{v} (as of the last readable cycle)" for k, v in (slot.get("last_holds") or {}).items()
                     if Condition(k) in self.policy.conditions}
        self._expire_override(slot, result, now)
        causes = self._causes(ctx, slot, holds, now)
        directive = max((d for _, d, _ in causes), key=lambda d: d.rank, default=AccountDirective.ALL_OPS)
        reason = f"{causes[0][0]}: {causes[0][2]}" if causes else "no condition active"

        if directive != prev_directive:
            self._change_episode(ctx, slot, result, prev_directive, prev_since, directive, causes, reason)
        if not ctx.degraded:
            self._check_orphans(ctx, slot, result, directive)

        slot["sequence"] = int(slot["sequence"]) + 1
        slot["last_cycle_utc"] = iso_z(now)
        result.slot = slot
        result.output = self._directive_file(ctx, slot, directive, reason, [c for c, _, _ in causes]).to_dict()
        result.audit_ids = (slot["sequence"], slot["episode"])
        result.summary = {"directive": str(directive), "causes": [c for c, _, _ in causes],
                          **(slot.get("last_eval") or {})}
        return result

    # ------------------------------------------------------------------------------------------- start and state
    def _start_slot(self, ctx: TaskContext, result: TaskResult) -> dict[str, Any]:
        """No slot: a first-ever start, or lost history (spec §7.3). A previous directive file means the account was
        governed and its state is gone: rebuild what can be rebuilt, keep the numbering, and trip ``state_lost``."""
        slot = _fresh_slot()
        prev = ctx.previous_output
        lost = prev is not None and ctx.state_load in (StateLoad.MISSING, StateLoad.CORRUPT)
        if prev is not None:
            try:
                slot["sequence"] = int(prev.get("sequence") or 0)
                slot["episode"] = int(prev.get("episode") or 0)
            except (TypeError, ValueError):
                pass
        if lost:
            slot["latches"][Condition.STATE_LOST.value] = {
                "condition": Condition.STATE_LOST.value, "directive": str(AccountDirective.NO_ENTRY_OPS),
                "scope": str(LatchScope.MANUAL), "tripped_utc": iso_z(ctx.now),
                "evidence": {"state_load": str(ctx.state_load), "last_directive_sequence": slot["sequence"]}}
            result.events.append(TaskEvent(AdminEvent.LATCH_TRIPPED, {"condition": "state_lost", "scope": "manual",
                                                                      "state_load": str(ctx.state_load)}))
            result.alerts.append(Alert(AlertLevel.CRITICAL, "STATE LOST",
                                       f"The Admin's state file was {ctx.state_load} while a directive existed. Equity-based "
                                       f"figures are rebuilt as estimates; NO_ENTRY_OPS until an operator latch reset."))
        return slot

    def _new_deals(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult) -> list[Deal]:
        """Deals since the cursor, each exactly once.

        The first readable cycle of a new slot has no cursor: it rebuilds what the deal history can give (the
        closed-balance high-water mark, any balance operation since ``account_start_utc``) and starts the cursor now,
        so history before the start is never judged for obedience.
        """
        now = ctx.now
        overlap = timedelta(seconds=DEAL_OVERLAP_S)
        if not slot["deal_cursor_utc"]:
            history = ctx.deals(self.policy.account_start_utc, now)
            slot["high_water"]["balance"] = round(_max_balance_along(history, ctx.snapshot.info.balance), 2)
            self._balance_operations(slot, result, history, now)
            slot["recent_deals"] = sorted(d.ticket for d in history if d.time_utc >= now - overlap)
            slot["deal_cursor_utc"] = iso_z(now)
            return []
        cursor = parse_utc(slot["deal_cursor_utc"])
        seen = set(int(t) for t in slot["recent_deals"])
        fetched = ctx.deals(cursor - overlap, now)
        new = [d for d in fetched if d.ticket not in seen]
        # The next fetch starts at now - overlap: remember exactly the tickets it can return again.
        slot["recent_deals"] = sorted(d.ticket for d in fetched if d.time_utc >= now - overlap)
        slot["deal_cursor_utc"] = iso_z(now)
        return new

    def _track_degraded(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult) -> None:
        if ctx.degraded != bool(slot.get("degraded")):
            state = "entering" if ctx.degraded else "leaving"
            result.events.append(TaskEvent(AdminEvent.ADMIN_DEGRADED, {"state": state, "reason": ctx.degraded_reason}))
            result.alerts.append(Alert(AlertLevel.CRITICAL if ctx.degraded else AlertLevel.INFO,
                                       f"ADMIN DEGRADED ({state})",
                                       ctx.degraded_reason or "the account is readable and verified again"))
        slot["degraded"] = ctx.degraded

    # ------------------------------------------------------------------------------------------ day and marks
    def _roll_day(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult, balance: float, equity: float) -> None:
        """Roll the trading day at ``T_d`` and snapshot its base (spec §6.3). Equity at ``T_d`` is only known if this
        Admin was running across the boundary; otherwise the base is taken from the balance and marked estimated."""
        now = ctx.now
        t_d = day_boundary(self.policy, now)
        current = slot["trading_day"]
        if current is not None:
            start = parse_utc(current["start_utc"])
            # A real boundary is ~24 h after the last (23/25 h across DST). A policy edit that moves day_tz or
            # day_start_hour shifts T_d by less than 12 h: it must neither re-base the day nor clear today's latches
            # (spec §8.1), so the new boundary applies from the next genuine one.
            if t_d <= start or (t_d - start) < timedelta(hours=12):
                return
        last = parse_utc(slot["last_cycle_utc"]) if slot["last_cycle_utc"] else None
        window = ctx.valid_for_s
        observed = last is not None and last < t_d and (t_d - last).total_seconds() <= window \
            and (now - t_d).total_seconds() <= window
        deals = ctx.deals(t_d, now)
        boundary_balance = balance - _net_after(deals, t_d)
        boundary_equity = equity if observed else None
        if self.policy.daily_base is DailyBase.DAY_START_BALANCE:
            base, estimated = boundary_balance, False
        elif boundary_equity is None:
            base, estimated = boundary_balance, True
        else:
            base, estimated = max(boundary_balance, boundary_equity), False
        slot["trading_day"] = {"start_utc": iso_z(t_d), "base": round(base, 2), "boundary_balance": round(boundary_balance, 2),
                               "boundary_equity": None if boundary_equity is None else round(boundary_equity, 2),
                               "base_estimated": estimated}
        for cond, latch in list(slot["latches"].items()):
            if latch.get("scope") == LatchScope.TRADING_DAY.value:
                del slot["latches"][cond]
                result.events.append(TaskEvent(AdminEvent.LATCH_CLEARED, {"condition": cond, "why": "day_boundary"}))
                result.alerts.append(Alert(AlertLevel.INFO, "LATCH CLEARED", f"{cond}: new trading day {iso_z(t_d)}"))
        result.events.append(TaskEvent(AdminEvent.DAY_ROLLED, {"trading_day": dict(slot["trading_day"])}))
        if estimated:
            result.alerts.append(Alert(AlertLevel.WARNING, "DAY BASE ESTIMATED",
                                       f"The Admin was not running at the day boundary {iso_z(t_d)}: the base "
                                       f"{_money(base)} is the boundary balance. The account's own floor may be higher."))

    def _update_high_water(self, slot: dict[str, Any], balance: float, equity: float, new_deals: list[Deal],
                           now: datetime) -> None:
        hw = slot["high_water"]
        if hw["equity"] is None or equity > float(hw["equity"]):
            hw["equity"], hw["equity_utc"] = round(equity, 2), iso_z(now)
        closed_high = _max_balance_along(new_deals, balance)
        hw["balance"] = round(max(closed_high, float(hw["balance"]) if hw["balance"] is not None else closed_high), 2)

    def _balance_operations(self, slot: dict[str, Any], result: TaskResult, deals: list[Deal], now: datetime) -> None:
        ops = [d for d in deals if d.kind.is_balance_operation and d.time_utc > self.policy.account_start_utc]
        if not ops or Condition.BALANCE_OPERATION.value in slot["latches"]:
            return
        evidence = [{"ticket": d.ticket, "kind": str(d.kind), "amount": round(d.profit, 2), "time_utc": iso_z(d.time_utc)}
                    for d in ops]
        slot["latches"][Condition.BALANCE_OPERATION.value] = {
            "condition": Condition.BALANCE_OPERATION.value, "directive": str(AccountDirective.NO_ENTRY_OPS),
            "scope": str(LatchScope.MANUAL), "tripped_utc": iso_z(now), "evidence": {"deals": evidence}}
        result.events.append(TaskEvent(AdminEvent.LATCH_TRIPPED, {"condition": "balance_operation", "scope": "manual",
                                                                  "deals": evidence}))
        result.alerts.append(Alert(AlertLevel.CRITICAL, "BALANCE OPERATION",
                                   f"{len(ops)} deposit/withdrawal/credit deal(s) after account_start_utc: the policy's "
                                   f"initial capital no longer describes the account. Update the policy, then reset the latch."))

    # ------------------------------------------------------------------------------------------ conditions
    def _evaluate_levels(self, balance: float, equity: float, levels) -> dict[str, str]:
        holds: dict[str, str] = {}
        if equity <= levels.max_floor:
            holds[Condition.MAX_LOSS.value] = f"equity {_money(equity)} <= max floor {_money(levels.max_floor)}"
        if equity <= levels.daily_floor:
            holds[Condition.DAILY_LOSS.value] = f"equity {_money(equity)} <= daily floor {_money(levels.daily_floor)}"
        if equity <= levels.max_warn:
            holds[Condition.MAX_WARN.value] = f"equity {_money(equity)} <= max warn level {_money(levels.max_warn)}"
        if equity <= levels.daily_warn:
            holds[Condition.DAILY_WARN.value] = f"equity {_money(equity)} <= daily warn level {_money(levels.daily_warn)}"
        if levels.target is not None:
            measured = balance if self.policy.target_measure is TargetMeasure.BALANCE else equity
            if measured >= levels.target:
                holds[Condition.TARGET.value] = (f"{self.policy.target_measure.value} {_money(measured)} >= target "
                                                 f"{_money(levels.target)}")
        return {k: v for k, v in holds.items() if Condition(k) in self.policy.conditions}

    def _trip_latches(self, slot: dict[str, Any], result: TaskResult, holds: dict[str, str], now: datetime,
                      balance: float, equity: float) -> None:
        for cond, evidence in holds.items():
            rule = self.policy.conditions[Condition(cond)]
            if rule.latch is LatchScope.NONE or cond in slot["latches"]:
                continue
            slot["latches"][cond] = {"condition": cond, "directive": str(rule.directive), "scope": str(rule.latch),
                                     "tripped_utc": iso_z(now),
                                     "evidence": {"text": evidence, "balance": round(balance, 2), "equity": round(equity, 2)}}
            result.events.append(TaskEvent(AdminEvent.LATCH_TRIPPED, {"condition": cond, "scope": str(rule.latch),
                                                                      "directive": str(rule.directive), "evidence": evidence}))
            result.alerts.append(Alert(AlertLevel.CRITICAL, "LATCH TRIPPED",
                                       f"{cond} ({rule.latch.value}): {evidence} -> {rule.directive.value}"))

    def _expire_override(self, slot: dict[str, Any], result: TaskResult, now: datetime) -> None:
        ov = slot["override"]
        if ov is not None and parse_utc(ov["until_utc"]) <= now:
            slot["override"] = None
            result.events.append(TaskEvent(AdminEvent.OVERRIDE_EXPIRED, {"override": ov}))
            result.alerts.append(Alert(AlertLevel.INFO, "OVERRIDE EXPIRED",
                                       f"{ov['directive']} by {ov['operator']} ended at {ov['until_utc']}"))

    def _causes(self, ctx: TaskContext, slot: dict[str, Any], holds: dict[str, str],
                now: datetime) -> list[tuple[str, AccountDirective, str]]:
        """Every active condition as (cause id, directive, evidence), most restrictive first (spec §6.6)."""
        causes: dict[str, tuple[AccountDirective, str]] = {}
        for cond, evidence in holds.items():
            causes[cond] = (self.policy.conditions[Condition(cond)].directive, evidence)
        for cond, latch in slot["latches"].items():
            if cond not in causes:
                text = (latch.get("evidence") or {}).get("text") or "latched"
                causes[cond] = (AccountDirective(latch["directive"]), f"{text} (latched {latch['tripped_utc']}, "
                                                                       f"{latch['scope']})")
        for window in self.policy.calendar:
            if window.active(now):
                causes[f"calendar:{window.id}"] = (window.directive, f"calendar window {window.id} ({window.kind.value})")
        ov = slot["override"]
        if ov is not None:
            causes[Condition.OVERRIDE.value] = (AccountDirective(ov["directive"]),
                                                f"override by {ov['operator']} until {ov['until_utc']}: {ov.get('reason') or ''}")
        if ctx.degraded:
            causes[Condition.ADMIN_DEGRADED.value] = (_FIXED[Condition.ADMIN_DEGRADED], ctx.degraded_reason or "degraded")
        return sorted(((cid, d, t) for cid, (d, t) in causes.items()),
                      key=lambda c: (-c[1].rank, _PRIORITY.get(c[0], 50), c[0]))

    # ------------------------------------------------------------------------------------------ episodes
    def _change_episode(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult,
                        prev: AccountDirective | None, prev_since: datetime | None, directive: AccountDirective,
                        causes: list, reason: str) -> None:
        now = ctx.now
        if prev is AccountDirective.NO_OPS and slot["episode_orphans"]:
            result.alerts.append(Alert(AlertLevel.WARNING, "NO_OPS EPISODE ENDED",
                                       f"{slot['episode_orphans']} orphan(s) reported while NO_OPS stood since "
                                       f"{iso_z(prev_since) if prev_since else 'n/a'}"))
        slot["episode"] = int(slot["episode"]) + 1
        slot["directive"], slot["since_utc"] = str(directive), iso_z(now)
        slot["reported"] = {"obedience": [], "orphans": []}
        slot["episode_orphans"] = 0
        last_eval = slot.get("last_eval") or {}
        result.events.append(TaskEvent(AdminEvent.DIRECTIVE_CHANGED,
                                       {"from": None if prev is None else str(prev), "to": str(directive),
                                        "causes": [c for c, _, _ in causes], "reason": reason, **last_eval}))
        levels = last_eval.get("levels") or {}
        metrics = last_eval.get("metrics") or {}
        body = (f"{prev.value if prev else '(none)'} -> {directive.value}\n{reason}\n"
                f"equity {_money(metrics.get('equity'))} | daily floor {_money(levels.get('daily_floor'))} | "
                f"max floor {_money(levels.get('max_floor'))}")
        level = AlertLevel.INFO if directive is AccountDirective.ALL_OPS else AlertLevel.CRITICAL
        result.alerts.append(Alert(level, "DIRECTIVE CHANGED", body))

    def _check_obedience(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult, new_deals: list[Deal],
                         prev: AccountDirective | None, prev_since: datetime | None) -> None:
        """Entries after a restrictive directive plus grace are breaches (spec §11). Judged against the directive that
        stood while they happened, i.e. the one in force before this cycle."""
        if prev not in RESTRICTIVE_DIRECTIVES or prev_since is None:
            return
        threshold = prev_since + timedelta(seconds=self.policy.obey_grace_s)
        reported = set(slot["reported"]["obedience"])
        owners = None
        offenders: list[dict[str, Any]] = []
        for d in new_deals:
            if d.kind in _ENTRY_KINDS and d.entry in _ENTRY_ENTRIES and d.time_utc > threshold and d.ticket not in reported:
                offenders.append({"what": "entry_deal", "ticket": d.ticket, "magic": d.magic, "symbol": d.symbol,
                                  "volume": d.volume, "time_utc": iso_z(d.time_utc)})
        for o in ctx.snapshot.orders:
            if o.setup_utc is not None and o.setup_utc > threshold and o.ticket not in reported:
                offenders.append({"what": "pending_order", "ticket": o.ticket, "magic": o.magic, "symbol": o.symbol,
                                  "volume": o.volume, "time_utc": iso_z(o.setup_utc)})
        for item in offenders:
            owners = owners if owners is not None else ctx.owners()
            item["owner"] = describe_owner(item["magic"], owners)
            reported.add(item["ticket"])
            result.events.append(TaskEvent(AdminEvent.OBEDIENCE_BREACH, {**item, "directive": str(prev)}))
            result.alerts.append(Alert(AlertLevel.CRITICAL, "OBEDIENCE BREACH",
                                       f"{item['what']} #{item['ticket']} {item['symbol']} {item['volume']}L magic "
                                       f"{item['magic']} ({item['owner']}) under {prev.value}"))
        slot["reported"]["obedience"] = sorted(reported)

    def _check_orphans(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult,
                       directive: AccountDirective) -> None:
        """Under ``NO_OPS``, anything still on the book ``close_grace_s`` after it began is an orphan (spec §11)."""
        if directive is not AccountDirective.NO_OPS or not slot["since_utc"]:
            return
        if (ctx.now - parse_utc(slot["since_utc"])).total_seconds() < self.policy.close_grace_s:
            return
        reported = set(slot["reported"]["orphans"])
        owners = None
        items = [("position", p.ticket, p.magic, p.symbol, p.volume) for p in ctx.snapshot.positions] + \
                [("pending_order", o.ticket, o.magic, o.symbol, o.volume) for o in ctx.snapshot.orders]
        for what, ticket, magic, symbol, volume in items:
            if ticket in reported:
                continue
            owners = owners if owners is not None else ctx.owners()
            owner = describe_owner(magic, owners)
            reported.add(ticket)
            slot["episode_orphans"] = int(slot["episode_orphans"]) + 1
            result.events.append(TaskEvent(AdminEvent.ORPHAN, {"what": what, "ticket": ticket, "magic": magic,
                                                               "symbol": symbol, "volume": volume, "owner": owner}))
            result.alerts.append(Alert(AlertLevel.CRITICAL, "ORPHAN",
                                       f"{what} #{ticket} {symbol} {volume}L magic {magic} ({owner}) still open under NO_OPS"))
        slot["reported"]["orphans"] = sorted(reported)

    # ------------------------------------------------------------------------------------------ requests
    def _handle_requests(self, ctx: TaskContext, slot: dict[str, Any], result: TaskResult) -> None:
        for request in ctx.requests:
            try:
                outcome, why = self._apply_request(ctx, slot, request, result)
            except Exception as exc:   # a malformed request is rejected, never guessed at
                outcome, why = RequestOutcome.REJECTED, f"malformed: {exc}"
            result.request_outcomes[request.request_id] = (outcome, why)

    def _apply_request(self, ctx: TaskContext, slot: dict[str, Any], request: AdminRequest,
                       result: TaskResult) -> tuple[RequestOutcome, str]:
        try:
            kind = RequestKind(request.kind)
        except ValueError:
            return RequestOutcome.REJECTED, f"unknown request kind {request.kind!r}"
        fields = request.fields
        if kind is RequestKind.RESET_LATCH:
            if ctx.degraded:
                # The condition cannot be re-checked while the account is unreadable; clearing the latch now would hold
                # a breached limit open (spec §6.4). The operator retries once the Admin reads the account again.
                return RequestOutcome.REJECTED, "the account is unreadable this cycle (degraded): a reset cannot be " \
                                                "re-checked against it; retry when the Admin is healthy"
            target = str(fields.get("target") or "").strip()
            if not target:
                return RequestOutcome.REJECTED, "reset_latch needs a target: a condition id or 'all'"
            names = list(slot["latches"]) if target == "all" else [target]
            missing = [n for n in names if n not in slot["latches"]]
            if target != "all" and missing:
                return RequestOutcome.REJECTED, f"no latch {target!r} is tripped (tripped: {sorted(slot['latches']) or 'none'})"
            for name in names:
                latch = slot["latches"].pop(name)
                result.events.append(TaskEvent(AdminEvent.LATCH_CLEARED, {"condition": name, "why": "operator_reset",
                                                                          "operator": request.operator,
                                                                          "request_id": request.request_id,
                                                                          "latch": latch}))
            return RequestOutcome.APPLIED, f"reset {', '.join(names) or 'nothing (no latch tripped)'}; a condition " \
                                           f"that still holds re-trips this cycle"
        if kind is RequestKind.OVERRIDE:
            try:
                directive = AccountDirective(fields.get("directive"))
            except ValueError:
                return RequestOutcome.REJECTED, f"unknown directive {fields.get('directive')!r}"
            if directive not in RESTRICTIVE_DIRECTIVES:
                return RequestOutcome.REJECTED, "an override can only tighten: NO_ENTRY_OPS or NO_OPS"
            if not fields.get("until_utc"):
                return RequestOutcome.REJECTED, "until_utc is required"
            until = parse_utc(fields["until_utc"])
            if until <= ctx.now:
                return RequestOutcome.REJECTED, f"until_utc {iso_z(until)} is not in the future"
            if (until - ctx.now).total_seconds() > self.policy.max_override_s:
                return RequestOutcome.REJECTED, f"until_utc is more than max_override_s ({self.policy.max_override_s}s) ahead"
            replaced = slot["override"]
            slot["override"] = {"request_id": request.request_id, "directive": str(directive), "until_utc": iso_z(until),
                                "operator": request.operator, "reason": str(fields.get("reason") or "")}
            return RequestOutcome.APPLIED, f"override {directive.value} until {iso_z(until)}" + \
                (f" (replaced {replaced['request_id']})" if replaced else "")
        if slot["override"] is None:
            return RequestOutcome.REJECTED, "no override is active"
        cleared = slot["override"]
        slot["override"] = None
        return RequestOutcome.APPLIED, f"cleared override {cleared['request_id']}"

    # ------------------------------------------------------------------------------------------ output
    def _directive_file(self, ctx: TaskContext, slot: dict[str, Any], directive: AccountDirective, reason: str,
                        causes: list[str]) -> DirectiveFile:
        last_eval = slot.get("last_eval") or {}
        day = slot["trading_day"]
        ov = slot["override"]
        evidence = {"trading_day": None if day is None else {"start_utc": day["start_utc"], "base": day["base"],
                                                             "base_estimated": day["base_estimated"]},
                    "levels": last_eval.get("levels"), "metrics": last_eval.get("metrics"),
                    "override": None if ov is None else {k: ov[k] for k in ("directive", "until_utc", "operator", "reason")}}
        account = DirectiveAccount(broker=ctx.broker_label, server=ctx.expected_server, login=int(ctx.expected_login),
                                   currency=str(slot.get("currency") or ""))
        return DirectiveFile(account=account, directive=directive, reason=reason, causes=tuple(causes),
                             since_utc=parse_utc(slot["since_utc"]), heartbeat_utc=ctx.now, valid_for_s=ctx.valid_for_s,
                             sequence=int(slot["sequence"]), episode=int(slot["episode"]), evidence=evidence)
