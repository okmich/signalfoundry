"""Pending-order queries: magic is filtered client-side and a failed query raises (see ``get_pending_orders``)."""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import patch

import pytest

from okmich_quant_mt5.functions.mt5 import DataFetchError, cancel_all_pending_orders, get_pending_orders

_Order = namedtuple("_Order", "ticket symbol magic")

_BOOK = (_Order(1, "EURUSD", 111), _Order(2, "EURUSD", 222), _Order(3, "GBPUSD", 111))


class _FakeMt5:
    """orders_get as MT5 behaves: filters by symbol only and swallows keywords it does not know (e.g. magic)."""

    def __init__(self, orders):
        self._orders = orders
        self.calls = []

    def orders_get(self, symbol=None, group=None, ticket=None, **unknown):
        self.calls.append(dict(symbol=symbol, **unknown))
        if self._orders is None:
            return None
        return tuple(o for o in self._orders if symbol is None or o.symbol == symbol)

    @staticmethod
    def last_error():
        return (-10004, "No IPC connection")


@pytest.mark.parametrize("symbol, magic, expected", [
    ("EURUSD", 111, [1]),
    (None, 111, [1, 3]),
    ("EURUSD", None, [1, 2]),
    (None, None, [1, 2, 3]),
])
def test_filters_by_symbol_and_magic(symbol, magic, expected):
    fake = _FakeMt5(_BOOK)
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        orders = get_pending_orders(symbol=symbol, magic=magic)
    assert [o["ticket"] for o in orders] == expected
    assert all("magic" not in call for call in fake.calls)


def test_query_failure_raises_rather_than_reporting_no_orders():
    """None means 'could not ask'; returning [] would read a terminal hiccup as an empty book."""
    with patch("okmich_quant_mt5.functions.mt5.mt5", _FakeMt5(None)):
        with pytest.raises(DataFetchError):
            get_pending_orders(symbol="EURUSD", magic=111)


def test_genuinely_empty_book_returns_empty():
    with patch("okmich_quant_mt5.functions.mt5.mt5", _FakeMt5(())):
        assert get_pending_orders(symbol="EURUSD", magic=111) == []


def test_cancel_all_cancels_only_this_systems_orders():
    with patch("okmich_quant_mt5.functions.mt5.mt5", _FakeMt5(_BOOK)), \
         patch("okmich_quant_mt5.functions.mt5.cancel_pending_order", return_value=True) as cancel:
        cancelled = cancel_all_pending_orders(symbol="EURUSD", magic=111)
    assert cancelled == [1]
    cancel.assert_called_once_with(1)
