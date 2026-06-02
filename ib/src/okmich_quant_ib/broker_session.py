"""IB concrete of the broker-neutral :class:`okmich_quant_core.BrokerSession` (LOGGING_CONTRACT §7.4).

Supplies the runner-scoped envelope identity (``broker`` / ``account_id`` / ``broker_session_id``)
and a **proven** disconnect: ``ib.disconnect()`` followed by an ``isConnected()`` check, so a
``shutdown`` record may assert ``broker_disconnected: true`` only when the session is verified down.
"""

from __future__ import annotations

import logging

from ib_async import IB

from .resilience import is_ib_connected

logger = logging.getLogger(__name__)


class IBBrokerSession:
    """:class:`okmich_quant_core.BrokerSession` adapter wrapping a connected IB handle."""

    def __init__(self, ib: IB, host: str, port: int, client_id: int, broker_label: str = "IB"):
        self._ib = ib
        self._host = host
        self._port = port
        self._client_id = client_id
        self._broker = broker_label
        # managedAccounts() is populated by connect; capture once so the envelope build is cheap.
        try:
            accounts = list(ib.managedAccounts() or [])
        except Exception:
            accounts = []
        self._account_id = accounts[0] if accounts else ""
        self._disconnected: bool | None = None  # cached proven result (idempotent)

    @property
    def broker(self) -> str:
        return self._broker

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def broker_session_id(self) -> str | None:
        return f"clientId={self._client_id}@{self._host}:{self._port}"

    def disconnect(self) -> bool:
        """Release the IB session and return whether it is **proven** disconnected. Idempotent."""
        if self._disconnected is not None:
            return self._disconnected
        try:
            if is_ib_connected(self._ib):
                self._ib.disconnect()
        except Exception:
            logger.exception("IBBrokerSession: error during ib.disconnect()")
        self._disconnected = not is_ib_connected(self._ib)
        if not self._disconnected:
            logger.error("IBBrokerSession: disconnect could not be proven — ib.isConnected() still True")
        return self._disconnected
