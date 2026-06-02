"""MT5 concrete of the broker-neutral :class:`okmich_quant_core.BrokerSession` (LOGGING_CONTRACT §7.4).

Identity is sourced from the closest source of truth: ``broker`` / ``account_id`` from the runner's
``.env`` login info (``LOGIN_SERVER`` / ``LOGIN_ID``), ``broker_session_id`` from the live
``mt5.terminal_info()``. ``disconnect()`` performs ``mt5.shutdown()`` then verifies via
``is_mt5_connected()`` so a ``shutdown`` record asserts ``broker_disconnected: true`` only when proven.
"""

from __future__ import annotations

import logging
import os

import MetaTrader5 as mt5

from .resilience import is_mt5_connected

logger = logging.getLogger(__name__)


class MT5BrokerSession:
    """:class:`okmich_quant_core.BrokerSession` adapter for the process-global MT5 terminal."""

    def __init__(self, broker: str | None = None, account_id: str | None = None,
                 broker_session_id: str | None = None):
        self._broker = broker if broker is not None else os.environ.get("LOGIN_SERVER", "")
        self._account_id = account_id if account_id is not None else os.environ.get("LOGIN_ID", "")
        self._broker_session_id = broker_session_id if broker_session_id is not None else self._read_session_id()
        self._disconnected: bool | None = None  # cached proven result (idempotent)

    @staticmethod
    def _read_session_id() -> str | None:
        try:
            ti = mt5.terminal_info()
        except Exception:
            return None
        if ti is None:
            return None
        return f"{getattr(ti, 'name', 'mt5')}:build{getattr(ti, 'build', '?')}"

    @property
    def broker(self) -> str:
        return self._broker

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def broker_session_id(self) -> str | None:
        return self._broker_session_id

    def disconnect(self) -> bool:
        """Release the MT5 terminal and return whether it is **proven** disconnected. Idempotent."""
        if self._disconnected is not None:
            return self._disconnected
        try:
            mt5.shutdown()
        except Exception:
            logger.exception("MT5BrokerSession: error during mt5.shutdown()")
        self._disconnected = not is_mt5_connected()
        if not self._disconnected:
            logger.error("MT5BrokerSession: disconnect could not be proven — terminal still connected")
        return self._disconnected
