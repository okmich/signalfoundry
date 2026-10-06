"""Test doubles for the crypto package: a scripted CCXT-like exchange, a manual clock, recorders.

``FakeExchange`` implements the subset of CCXT's async API the package calls, with the behaviours the tests need to
script: candles with a forming last bar, a WebSocket OHLCV queue, order placement that fails (or accepts AND fails)
on demand, conditional orders, positions, fills and balances.
"""
from __future__ import annotations

import asyncio
import itertools
import math
from typing import Any, Optional

from ccxt.base import errors as ccxt_errors

from okmich_quant_core.logging import BaseEventLogger
from okmich_quant_core.notification.base import BaseNotifier

MIN = 60_000
T0 = 1_780_000_000_000 - (1_780_000_000_000 % (5 * MIN))  # a 5m-aligned epoch ms in 2026


class FakeClock:
    def __init__(self, now_ms: int = T0):
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> None:
        self.now_ms += ms

    async def sleep(self, seconds: float) -> None:
        """Virtual sleep: advances the clock instead of waiting, then yields to the loop."""
        self.now_ms += int(seconds * 1000)
        await asyncio.sleep(0)


def perp_market(symbol: str = "BTC/USDT:USDT", contract_size: float = 1.0, amount_step: float = 0.001,
                min_amount: float = 0.001, min_cost: float = 5.0, tick: float = 0.1) -> dict:
    base, rest = symbol.split("/")
    quote, settle = rest.split(":")
    return {"id": base + quote, "symbol": symbol, "base": base, "quote": quote, "settle": settle, "type": "swap",
            "spot": False, "swap": True, "linear": True, "contract": True, "contractSize": contract_size,
            "precision": {"amount": amount_step, "price": tick},
            "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}}}


def spot_market(symbol: str = "BTC/USDT", amount_step: float = 0.000001, min_amount: float = 0.000048,
                min_cost: float = 1.0, tick: float = 0.01) -> dict:
    base, quote = symbol.split("/")
    return {"id": base + quote, "symbol": symbol, "base": base, "quote": quote, "settle": None, "type": "spot",
            "spot": True, "swap": False, "linear": None, "contract": False, "contractSize": None,
            "precision": {"amount": amount_step, "price": tick},
            "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}}}


_CREATE_FEATURES = {"attachedStopLossTakeProfit": {"price": True}, "stopLossPrice": True, "takeProfitPrice": True}

DEFAULT_HAS = {cap: True for cap in (
    "fetchOHLCV", "createOrder", "cancelOrder", "fetchOpenOrders", "fetchClosedOrders", "fetchBalance", "fetchTicker",
    "fetchMyTrades", "fetchPositions", "setLeverage", "setMarginMode", "setPositionMode", "watchOHLCV", "watchOrders",
    "watchMyTrades", "watchPositions", "watchTicker")}


class FakeExchange:
    id = "fakex"

    def __init__(self, clock: FakeClock, *, markets: Optional[list[dict]] = None, has: Optional[dict] = None,
                 ohlcv_limit: int = 3, timeframes: Optional[dict] = None):
        self.clock = clock
        self.markets = {m["symbol"]: m for m in (markets or [perp_market(), spot_market()])}
        self.has = dict(DEFAULT_HAS if has is None else has)
        block = {"fetchOHLCV": {"limit": ohlcv_limit}}
        self.features = {"spot": {"createOrder": dict(_CREATE_FEATURES), **block},
                         "swap": {"linear": {"createOrder": dict(_CREATE_FEATURES), **block}}}
        self.timeframes = timeframes or {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "1d": "D", "1w": "W"}
        self.options: dict[str, Any] = {}
        self.urls: dict[str, Any] = {"test": {"public": "https://test"}}
        self.session: Any = object()
        self.clients: dict = {"wss://fake": object()}
        self.tcp_connector = None
        self.candles: dict[tuple[str, str], dict[int, list]] = {}
        self.forming: dict[tuple[str, str], list] = {}
        self.ws_ohlcv: asyncio.Queue = asyncio.Queue()
        self.orders: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.positions: dict[str, dict] = {}
        self.balance = {"total": {"USDT": 10_000.0, "BTC": 0.0}, "free": {"USDT": 10_000.0, "BTC": 0.0}}
        self.ticker = {"symbol": "BTC/USDT:USDT", "bid": 99.9, "ask": 100.1, "last": 100.0, "markPrice": 100.0}
        self.create_script: list = []
        self.calls: list[tuple] = []
        self.ohlcv_requests: list[tuple] = []
        self._ids = itertools.count(1)
        self.closed = False

    # ------------------------------------------------------------------ CCXT surface: metadata
    def market(self, symbol: str) -> dict:
        if symbol not in self.markets:
            raise ccxt_errors.BadSymbol(f"fakex does not have market symbol {symbol}")
        return self.markets[symbol]

    def milliseconds(self) -> int:
        return self.clock()

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        step = self.market(symbol)["precision"]["amount"]
        value = math.floor(amount / step + 1e-9) * step
        if value <= 0:
            raise ccxt_errors.InvalidOrder("fakex amount must be greater than minimum amount precision")
        return repr(round(value, 12))

    def price_to_precision(self, symbol: str, price: float) -> str:
        tick = self.market(symbol)["precision"]["price"]
        return repr(round(round(price / tick) * tick, 12))

    async def load_markets(self, reload: bool = False):
        self.calls.append(("load_markets",))
        return self.markets

    async def load_time_difference(self):
        self.calls.append(("load_time_difference",))

    # ------------------------------------------------------------------ candles
    def add_candles(self, symbol: str, timeframe: str, rows: list[list]) -> None:
        book = self.candles.setdefault((symbol, timeframe), {})
        for row in rows:
            book[int(row[0])] = list(row)

    def set_forming(self, symbol: str, timeframe: str, row: Optional[list]) -> None:
        if row is None:
            self.forming.pop((symbol, timeframe), None)
        else:
            self.forming[(symbol, timeframe)] = list(row)

    async def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
        self.ohlcv_requests.append((symbol, timeframe, since, limit))
        rows = sorted(self.candles.get((symbol, timeframe), {}).values(), key=lambda r: r[0])
        forming = self.forming.get((symbol, timeframe))
        if forming is not None:
            rows = rows + [forming]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return [list(r) for r in rows[: (limit or 200)]]

    async def watch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
        return await self.ws_ohlcv.get()

    # ------------------------------------------------------------------ orders
    async def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = dict(params or {})
        self.calls.append(("create_order", symbol, type, side, amount, price, params))
        action = self.create_script.pop(0) if self.create_script else None
        if isinstance(action, tuple) and action[0] == "accept_then":
            self._new_order(symbol, type, side, amount, price, params)
            raise action[1]
        if isinstance(action, BaseException):
            raise action
        return self._new_order(symbol, type, side, amount, price, params)

    def _new_order(self, symbol, type, side, amount, price, params) -> dict:
        oid = str(next(self._ids))
        trigger = any(k in params for k in ("stopLossPrice", "takeProfitPrice", "triggerPrice"))
        order = {"id": oid, "clientOrderId": params.get("clientOrderId"), "symbol": symbol, "type": type,
                 "side": side, "amount": amount, "price": price, "status": "open", "filled": 0.0, "trigger": trigger,
                 "params": params, "info": {}}
        self.orders[oid] = order
        return dict(order)

    async def cancel_order(self, id, symbol=None, params=None):
        self.calls.append(("cancel_order", id, symbol, dict(params or {})))
        order = self.orders.get(str(id))
        if order is None or order["status"] != "open":
            raise ccxt_errors.OrderNotFound(f"fakex order {id} not found")
        order["status"] = "canceled"
        return dict(order)

    async def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        trigger = bool((params or {}).get("trigger"))
        return [dict(o) for o in self.orders.values()
                if o["status"] == "open" and (symbol is None or o["symbol"] == symbol)
                and (not (params or {}) or o["trigger"] == trigger)]

    async def fetch_closed_orders(self, symbol=None, since=None, limit=None, params=None):
        return [dict(o) for o in self.orders.values() if o["status"] != "open" and o["symbol"] == symbol]

    # ------------------------------------------------------------------ fills, positions, balances
    def fill(self, order_id: str, price: float, *, qty: Optional[float] = None, fee: Optional[dict] = None,
             ts: Optional[int] = None, info: Optional[dict] = None) -> dict:
        """Execute an order (fully by default): records the trade, updates position / balance. Returns the trade."""
        order = self.orders[str(order_id)]
        qty = float(order["amount"]) if qty is None else qty
        order["filled"] = float(order.get("filled") or 0.0) + qty
        if order["filled"] >= float(order["amount"]) - 1e-12:
            order["status"] = "closed"
        trade = self.make_trade(order["symbol"], order["side"], qty, price, order_id=order["id"],
                                client_order_id=order.get("clientOrderId"), fee=fee, ts=ts, info=info)
        return trade

    def make_trade(self, symbol, side, qty, price, *, order_id=None, client_order_id=None, fee=None, ts=None,
                   info=None) -> dict:
        tid = f"t{next(self._ids)}"
        trade = {"id": tid, "order": order_id, "clientOrderId": client_order_id, "symbol": symbol, "side": side,
                 "amount": qty, "price": price, "timestamp": ts if ts is not None else self.clock(),
                 "fee": fee, "info": dict(info or {})}
        self.trades.append(trade)
        market = self.market(symbol)
        if market["swap"]:
            self._apply_perp(symbol, side, qty, price)
        else:
            self._apply_spot(market, side, qty, price, fee)
        return dict(trade)

    def _apply_perp(self, symbol, side, qty, price):
        pos = self.positions.get(symbol)
        signed = (qty if side == "buy" else -qty) * self.market(symbol)["contractSize"]
        current = 0.0
        if pos is not None:
            current = pos["contracts"] * pos["contractSize"] * (1 if pos["side"] == "long" else -1)
        new = current + signed
        if abs(new) < 1e-12:
            self.positions.pop(symbol, None)
            return
        entry = price if pos is None or (current > 0) != (new > 0) else pos["entryPrice"]
        if pos is not None and (current > 0) == (signed > 0):
            entry = (pos["entryPrice"] * abs(current) + price * abs(signed)) / abs(new)
        self.positions[symbol] = {"symbol": symbol, "contracts": abs(new) / self.market(symbol)["contractSize"],
                                  "contractSize": self.market(symbol)["contractSize"],
                                  "side": "long" if new > 0 else "short", "entryPrice": entry, "markPrice": price,
                                  "timestamp": self.clock(), "stopLossPrice": None, "takeProfitPrice": None,
                                  "hedged": False, "unrealizedPnl": 0.0}

    def _apply_spot(self, market, side, qty, price, fee):
        base, quote = market["base"], market["quote"]
        fee_cost = float((fee or {}).get("cost") or 0.0)
        fee_ccy = (fee or {}).get("currency")
        for bucket in ("total", "free"):
            b = self.balance[bucket]
            if side == "buy":
                b[base] = b.get(base, 0.0) + qty - (fee_cost if fee_ccy == base else 0.0)
                b[quote] = b.get(quote, 0.0) - qty * price
            else:
                b[base] = b.get(base, 0.0) - qty
                b[quote] = b.get(quote, 0.0) + qty * price - (fee_cost if fee_ccy == quote else 0.0)

    def set_position(self, symbol: str, side: Optional[str], contracts: float = 0.0, entry: float = 100.0, **extra):
        if side is None or contracts == 0:
            self.positions.pop(symbol, None)
            return
        self.positions[symbol] = {"symbol": symbol, "contracts": contracts,
                                  "contractSize": self.market(symbol)["contractSize"], "side": side,
                                  "entryPrice": entry, "markPrice": entry, "timestamp": self.clock(),
                                  "stopLossPrice": None, "takeProfitPrice": None, "hedged": False, **extra}

    async def fetch_positions(self, symbols=None, params=None):
        return [dict(p) for s, p in self.positions.items() if symbols is None or s in symbols]

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        rows = [t for t in self.trades if (symbol is None or t["symbol"] == symbol)
                and (since is None or t["timestamp"] >= since)]
        rows.sort(key=lambda t: t["timestamp"])
        return [dict(t) for t in rows[: (limit or 100)]]

    async def fetch_balance(self, params=None):
        return {"total": dict(self.balance["total"]), "free": dict(self.balance["free"])}

    async def fetch_ticker(self, symbol, params=None):
        return dict(self.ticker, symbol=symbol)

    async def watch_ticker(self, symbol, params=None):
        await asyncio.sleep(3600)

    async def watch_orders(self, symbol=None, since=None, limit=None, params=None):
        await asyncio.sleep(3600)

    async def watch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        await asyncio.sleep(3600)

    async def watch_positions(self, symbols=None, since=None, limit=None, params=None):
        await asyncio.sleep(3600)

    async def set_position_mode(self, hedged, symbol=None, params=None):
        self.calls.append(("set_position_mode", hedged, symbol))

    async def set_leverage(self, leverage, symbol=None, params=None):
        self.calls.append(("set_leverage", leverage, symbol))

    async def set_margin_mode(self, margin_mode, symbol=None, params=None):
        self.calls.append(("set_margin_mode", margin_mode, symbol))

    def set_sandbox_mode(self, enabled):
        self.calls.append(("set_sandbox_mode", enabled))

    async def close(self):
        self.closed = True
        self.session = None
        self.clients = {}


class RecordingLogger(BaseEventLogger):
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class RecordingNotifier(BaseNotifier):
    def __init__(self):
        self.events: list[tuple] = []

    def on_trade_opened(self, symbol, direction, volume, price, sl, tp, magic, ticket):
        self.events.append(("opened", symbol, direction, volume, price))

    def on_trade_closed(self, symbol, ticket, profit, price=0.0, reason=""):
        self.events.append(("closed", symbol, ticket, profit, price, reason))

    def on_trade_modified(self, symbol, ticket, sl, tp):
        self.events.append(("modified", symbol, ticket, sl, tp))

    def on_trade_filled(self, symbol, order_id, qty, avg_price):
        self.events.append(("filled", symbol, order_id, qty, avg_price))

    def on_trade_failed(self, symbol, direction, reason, retcode=None, context=None):
        self.events.append(("failed", symbol, direction, reason))

    def on_error(self, strategy_name, error_message, context=None):
        self.events.append(("error", strategy_name, error_message))

    def on_circuit_breaker_tripped(self, strategy_name, consecutive_errors):
        self.events.append(("breaker", strategy_name, consecutive_errors))

    def on_connection_lost(self, strategy_name):
        self.events.append(("lost", strategy_name))

    def on_connection_restored(self, strategy_name):
        self.events.append(("restored", strategy_name))

    def close(self):
        pass

    def of(self, kind: str) -> list[tuple]:
        return [e for e in self.events if e[0] == kind]
