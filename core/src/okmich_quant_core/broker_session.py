"""Broker-neutral session abstraction (LOGGING_CONTRACT §7.4, design B).

The runner lifecycle owners (:class:`RunLoop` for MT5, ``IBEventLoop`` for IB) must stay
broker-agnostic, yet they need two broker facts to satisfy the contract:

1. **Identity** for the runner-scoped envelope — ``broker`` / ``account_id`` /
   ``broker_session_id``.
2. A **proven disconnect** at shutdown — ``shutdown.broker_disconnected: true`` may be written
   **only after** the broker session was released *and the release was verified* (§7.4), never
   inferred.

``BrokerSession`` is the seam: ``core`` depends only on this Protocol; the concrete adapters
live in the broker packages (which already import ``MetaTrader5`` / ``ib_async``). The runner
script owns the broker handle, so it constructs the concrete session and injects it.

``disconnect()`` MUST be **idempotent** — it may be reached from more than one teardown path
(KeyboardInterrupt, reconnect-failure, atexit). It performs the disconnect at most once,
verifies it, and returns the cached proven result on subsequent calls.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BrokerSession(Protocol):
    """The broker facts the framework needs for the inference-log envelope + clean shutdown."""

    @property
    def broker(self) -> str:
        """Broker / platform identifier (e.g. the MT5 server, ``"IB"``)."""
        ...

    @property
    def account_id(self) -> str:
        """Trading account the system runs under."""
        ...

    @property
    def broker_session_id(self) -> str | None:
        """Identifier of the broker session/terminal (MT5 terminal id, IB clientId); ``None`` if unavailable."""
        ...

    def disconnect(self) -> bool:
        """Release the broker session and return whether the release was **proven** clean.

        Performs the disconnect call (``mt5.shutdown()`` / ``ib.disconnect()``) and verifies it
        via the broker's own connection check. Returns ``True`` only if proven disconnected.
        Idempotent: subsequent calls return the cached proven result without re-disconnecting.
        """
        ...
