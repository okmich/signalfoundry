"""Shared fixtures for okmich_quant_ib unit tests."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _ops_log_base(tmp_path_factory, monkeypatch):
    """Redirect the fail-closed default inference logger (BaseStrategy.__init__, LOGGING_CONTRACT §5)
    to a per-test temp dir so building an IB strategy never writes to the production ops log root."""
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path_factory.mktemp("quant_logs")))


@pytest.fixture
def fake_contract():
    """Build a SimpleNamespace contract with conId, symbol, secType, exchange."""
    return SimpleNamespace(
        conId=12345, symbol="AAPL", secType="STK", exchange="SMART", currency="USD",
        multiplier="", primaryExchange="NASDAQ", tradingClass="",
        lastTradeDateOrContractMonth="", strike=0.0, right="",
    )


@pytest.fixture
def make_fill():
    """Factory: make_fill(symbol, conId, side, shares, price, orderRef)."""
    def _factory(symbol="AAPL", conId=12345, side="BOT", shares=10, price=100.0,
                 orderRef="sf_42"):
        return SimpleNamespace(
            contract=SimpleNamespace(conId=conId, symbol=symbol),
            execution=SimpleNamespace(side=side, shares=shares, price=price,
                                      orderRef=orderRef),
        )
    return _factory


@pytest.fixture
def make_realtime_bar():
    """Factory: make_realtime_bar(epoch_seconds, o, h, l, c, v)."""
    def _factory(epoch, o=100.0, h=101.0, l=99.0, c=100.5, v=1000.0):
        return SimpleNamespace(
            time=datetime.fromtimestamp(epoch, tz=timezone.utc),
            open_=o, high=h, low=l, close=c, volume=v,
        )
    return _factory


@pytest.fixture
def mock_ib():
    """A MagicMock ``IB`` instance with reasonable async stubs."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.openTrades.return_value = []
    ib.fills.return_value = []
    ib.accountSummary.return_value = []
    return ib
