"""Phase 4 test — MT5BrokerSession identity + proven idempotent disconnect (LOGGING_CONTRACT §7.4)."""
from __future__ import annotations

import okmich_quant_mt5.broker_session as bs


def test_identity_from_env_and_proven_idempotent_disconnect(monkeypatch):
    monkeypatch.setenv("LOGIN_SERVER", "Deriv-Demo")
    monkeypatch.setenv("LOGIN_ID", "42")

    calls = {"shutdown": 0}
    monkeypatch.setattr(bs.mt5, "shutdown", lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1))
    # After shutdown the terminal is down — prove via the resilience check (patched here).
    monkeypatch.setattr(bs, "is_mt5_connected", lambda: False)

    sess = bs.MT5BrokerSession(broker_session_id="term-1")
    assert sess.broker == "Deriv-Demo"
    assert sess.account_id == "42"
    assert sess.broker_session_id == "term-1"

    assert sess.disconnect() is True           # proven down
    assert calls["shutdown"] == 1
    assert sess.disconnect() is True            # idempotent: cached, no second shutdown
    assert calls["shutdown"] == 1


def test_disconnect_unproven_returns_false(monkeypatch):
    monkeypatch.setattr(bs.mt5, "shutdown", lambda: None)
    monkeypatch.setattr(bs, "is_mt5_connected", lambda: True)  # still connected → not proven
    sess = bs.MT5BrokerSession(broker="B", account_id="1", broker_session_id="s")
    assert sess.disconnect() is False
