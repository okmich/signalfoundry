"""``pending_order_cleanup``: cancel pending orders left resting too long (ACCOUNT_ADMIN_SPEC §12.3).

A pending order that has rested for hours is an entry decided on a market that has gone. Every cycle this task cancels
each pending order, of every type, whose age reaches ``max_age_s`` and whose magic is in scope. Age is measured from the
order's setup time converted to UTC by the broker library; an order whose setup time could not be converted has no
known age and is never cancelled.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .._validate import Problems, choice, integer, unknown_keys
from ..enums import AdminTaskKind, AlertLevel, BookActionOutcome, CleanupScope, RequestOutcome
from ..owners import describe_owner
from ..snapshot import PendingOrder
from ..timeutil import iso_z
from .base import AdminTask, Alert, TaskContext, TaskResult, register_task

SLOT_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
MIN_MAX_AGE_S = 60
_KEYS = frozenset({"kind", "max_age_s", "scope", "magics"})


@register_task
class PendingOrderCleanupTask(AdminTask):
    kind = AdminTaskKind.PENDING_ORDER_CLEANUP

    def __init__(self, entry: Mapping[str, Any]):
        super().__init__(entry)
        self.max_age_s = int(entry["max_age_s"])
        self.scope = CleanupScope(entry.get("scope") or CleanupScope.ALL)
        self.magics = frozenset(int(m) for m in entry.get("magics") or ())

    @classmethod
    def validate(cls, entry: Mapping[str, Any], prefix: str) -> list[str]:
        p = Problems(prefix)
        unknown_keys(entry, _KEYS, p)
        integer(entry, "max_age_s", p, minimum=MIN_MAX_AGE_S)
        scope = choice(entry, "scope", CleanupScope, p, required=False) or CleanupScope.ALL
        magics = entry.get("magics")
        if scope is CleanupScope.MAGICS:
            ok = isinstance(magics, list) and magics and all(isinstance(m, int) and not isinstance(m, bool) and m >= 0
                                                             for m in magics)
            if not ok:
                p.add(p.at("magics"), "must be a non-empty list of non-negative integers when scope is 'magics'")
            elif len(set(magics)) != len(magics):
                p.add(p.at("magics"), "must not repeat a magic")
        elif magics is not None:
            p.add(p.at("magics"), "only allowed when scope is 'magics'")
        return p.items

    def _in_scope(self, order: PendingOrder, owners) -> bool:
        if self.scope is CleanupScope.ALL:
            return True
        if self.scope is CleanupScope.MAGICS:
            return order.magic in self.magics
        owner = owners().get(order.magic)
        return owner is None or not owner.running     # orphaned: not this account's, or its system is not running

    def on_cycle(self, ctx: TaskContext) -> TaskResult:
        slot = copy.deepcopy(ctx.slot) if ctx.slot else {"schema_version": SLOT_SCHEMA_VERSION, "failing": {},
                                                         "count_date": None, "cancelled_today": 0, "cancelled_total": 0}
        result = TaskResult(slot=slot, output={})
        for request in ctx.requests:
            result.request_outcomes[request.request_id] = (RequestOutcome.REJECTED, "pending_order_cleanup takes no requests")
        today = iso_z(ctx.now)[:10]
        if slot["count_date"] != today:
            slot["count_date"], slot["cancelled_today"] = today, 0

        cancelled: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        unknown_age: list[int] = []
        if not ctx.degraded:
            owners_cache: dict | None = None

            def owners():
                nonlocal owners_cache
                if owners_cache is None:
                    owners_cache = ctx.owners()
                return owners_cache

            present = {o.ticket for o in ctx.snapshot.orders}
            for ticket in [t for t in slot["failing"] if int(t) not in present]:
                del slot["failing"][ticket]     # it left the book: nothing more to cancel
            for order in ctx.snapshot.orders:
                if order.setup_utc is None:
                    unknown_age.append(order.ticket)
                    continue
                age_s = (ctx.now - order.setup_utc).total_seconds()
                if age_s < self.max_age_s or not self._in_scope(order, owners):
                    continue
                record = {"ticket": order.ticket, "magic": order.magic, "symbol": order.symbol,
                          "type": str(order.order_type), "volume": order.volume, "price": order.price,
                          "age_s": int(age_s), "owner": describe_owner(order.magic, owners())}
                outcome = ctx.actions.cancel_pending(order, reason="max_age")
                if outcome.outcome is BookActionOutcome.DONE:
                    cancelled.append(record)
                    slot["failing"].pop(str(order.ticket), None)
                elif outcome.outcome is BookActionOutcome.SKIPPED_FILLED:
                    skipped.append(record)
                    slot["failing"].pop(str(order.ticket), None)
                else:
                    key = str(order.ticket)
                    first = key not in slot["failing"]
                    entry = slot["failing"].setdefault(key, {"first_failure_utc": iso_z(ctx.now)})
                    entry["last_error"] = outcome.error
                    if first:
                        result.alerts.append(Alert(AlertLevel.WARNING, "CANCEL FAILING",
                                                   f"#{order.ticket} {order.symbol} {order.order_type.value} magic "
                                                   f"{order.magic} ({record['owner']}), age {int(age_s)}s: {outcome.error}. "
                                                   f"Retried every cycle; alerted once."))
            if unknown_age and not slot.get("unknown_age_alerted"):
                result.alerts.append(Alert(AlertLevel.WARNING, "ORDER AGE UNKNOWN",
                                           f"{len(unknown_age)} pending order(s) have a setup time the broker clock could "
                                           f"not convert to UTC; they are not judged until it can. Alerted once."))
            slot["unknown_age_alerted"] = bool(unknown_age)
        slot["cancelled_today"] = int(slot["cancelled_today"]) + len(cancelled)
        slot["cancelled_total"] = int(slot["cancelled_total"]) + len(cancelled)
        result.output = {"task": self.kind.value, "schema_version": OUTPUT_SCHEMA_VERSION, "heartbeat_utc": iso_z(ctx.now),
                         "degraded": ctx.degraded,
                         "config": {"max_age_s": self.max_age_s, "scope": self.scope.value, "magics": sorted(self.magics)},
                         "cancelled_this_cycle": cancelled, "skipped_filled_this_cycle": skipped,
                         "unknown_age_tickets": unknown_age, "failing": slot["failing"],
                         "cancelled_today": slot["cancelled_today"], "cancelled_total": slot["cancelled_total"]}
        result.summary = {"cancelled": len(cancelled), "skipped_filled": len(skipped), "failing": len(slot["failing"]),
                          "unknown_age": len(unknown_age)}
        return result
