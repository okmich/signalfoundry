"""Persisted registry of the orders this strategy sent and WHY (their :class:`OrderRole`).

The role answers two questions later, when the venue no longer can: which close was a stop-loss vs a strategy exit
(close attribution), and which open orders shutdown may cancel (entries and exits) versus must keep (protective
stops - cancelling them would leave a position naked while the process is down).
"""
from typing import Optional

from .enums import OrderRole
from .state_store import StateStore
from .timeframe_utils import utc_now_ms

#: Orders older than this are pruned from the registry (the venue's own history covers older attribution).
_RETENTION_MS = 30 * 24 * 3600 * 1000


class OrderRegistry:

    def __init__(self, store: StateStore):
        self._store = store
        self._by_client = store.section("orders")

    def record(self, client_order_id: str, role: OrderRole, order_id: Optional[str] = None) -> None:
        entry = self._by_client.setdefault(client_order_id, {"role": role.value, "ts": utc_now_ms()})
        entry["role"] = role.value
        if order_id:
            entry["order_id"] = str(order_id)
        self._prune()
        self._store.save()

    def attach_order_id(self, client_order_id: Optional[str], order_id: Optional[str]) -> None:
        if not client_order_id or not order_id:
            return
        entry = self._by_client.get(client_order_id)
        if entry is not None and entry.get("order_id") != str(order_id):
            entry["order_id"] = str(order_id)
            self._store.save()

    def role_of(self, *, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Optional[OrderRole]:
        if client_order_id and client_order_id in self._by_client:
            return OrderRole(self._by_client[client_order_id]["role"])
        if order_id:
            for entry in self._by_client.values():
                if entry.get("order_id") == str(order_id):
                    return OrderRole(entry["role"])
        return None

    def order_ids(self) -> set[str]:
        return {e["order_id"] for e in self._by_client.values() if e.get("order_id")}

    def client_ids_with_role(self, *roles: OrderRole) -> set[str]:
        wanted = {r.value for r in roles}
        return {cid for cid, e in self._by_client.items() if e.get("role") in wanted}

    def _prune(self) -> None:
        cutoff = utc_now_ms() - _RETENTION_MS
        # Protective orders can rest for months; their role must outlive the retention window or a stop fill would
        # be attributed as MANUAL.
        keep = {OrderRole.STOP_LOSS.value, OrderRole.TAKE_PROFIT.value}
        stale = [cid for cid, e in self._by_client.items()
                 if int(e.get("ts", 0)) < cutoff and e.get("role") not in keep]
        for cid in stale:
            del self._by_client[cid]
