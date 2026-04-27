"""Low-level IB API wrappers. All functions take a connected ``IB`` as first arg."""
import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from ib_async import (
    IB, Contract, LimitOrder, MarketOrder, StopOrder, Trade, util,
)

from ..contract import IBContractConfig, make_order_ref, resolve_contract
from ..resilience import IBPermanentError, IBTransientError
from ..timeframe_utils import BAR_SIZE_MINUTES, MAX_DURATION_DAYS, required_duration

logger = logging.getLogger(__name__)


# ===== Connection =====

async def connect_ib(ib: IB, host: str = "127.0.0.1", port: int = 4002,
                     client_id: int = 1, timeout: float = 10.0) -> IB:
    """Connect to IB Gateway. Raises IBConnectionError on failure."""
    from ..resilience import IBConnectionError
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        return ib
    except Exception as e:
        raise IBConnectionError(f"Failed to connect to IB at {host}:{port}: {e}") from e


def disconnect_ib(ib: IB) -> None:
    if ib.isConnected():
        ib.disconnect()


# ===== Contract metadata =====

async def fetch_contract_info(ib: IB, contract: Contract) -> dict[str, Any]:
    """Return key contract attributes as a plain dict for position-manager math."""
    details = await ib.reqContractDetailsAsync(contract)
    if not details:
        raise IBPermanentError(f"No contract details for {contract.symbol}", 200)
    d = details[0]
    return {
        "symbol": contract.symbol,
        "sec_type": contract.secType,
        "exchange": contract.exchange,
        "currency": contract.currency,
        "min_tick": d.minTick,
        "multiplier": float(contract.multiplier) if contract.multiplier else 1.0,
        "min_size": getattr(d, "minSize", 1.0),
        "size_increment": getattr(d, "sizeIncrement", 1.0),
        "long_name": d.longName,
    }


# ===== Market data =====

async def fetch_historical_bars(ib: IB, contract: Contract, bar_size: str, bars_to_copy: int,
                                what_to_show: str, use_rth: bool) -> pd.DataFrame:
    """Single-request fetch ending at now. Use the paginated variant for bootstrap warm-up.

    Returns a DataFrame indexed by UTC-aware datetime with [open, high, low, close, volume].
    """
    duration = required_duration(bar_size, bars_to_copy, use_rth)
    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime="", durationStr=duration, barSizeSetting=bar_size,
        whatToShow=what_to_show, useRTH=use_rth, formatDate=2, keepUpToDate=False,
    )
    if not bars:
        raise IBTransientError(
            f"reqHistoricalData returned empty for {contract.symbol} "
            f"({bar_size}, {duration}, useRTH={use_rth})", 162,
        )
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


async def fetch_historical_bars_paginated(ib: IB, contract: Contract, bar_size: str,
                                          bars_to_copy: int, what_to_show: str,
                                          use_rth: bool) -> pd.DataFrame:
    """Paginated warm-up fetch ending at now.

    Splits the request into MAX_DURATION_DAYS[bar_size]-day chunks walking backward,
    so bootstrap works for any bars_to_copy regardless of IB per-request limits.
    """
    bar_minutes = BAR_SIZE_MINUTES[bar_size]
    minutes_per_day = 390 if use_rth else 1440
    bars_per_day = minutes_per_day / bar_minutes
    total_days_needed = math.ceil(bars_to_copy / bars_per_day * 1.3)
    max_chunk_days = MAX_DURATION_DAYS[bar_size]

    chunks: list[pd.DataFrame] = []
    cursor = datetime.now(tz=timezone.utc)
    remaining_days = total_days_needed

    while remaining_days > 0:
        chunk_days = min(remaining_days, max_chunk_days)
        duration = f"{max(1, chunk_days)} D"
        end_str = cursor.strftime("%Y%m%d %H:%M:%S UTC")
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime=end_str, durationStr=duration, barSizeSetting=bar_size,
            whatToShow=what_to_show, useRTH=use_rth, formatDate=2, keepUpToDate=False,
        )
        if bars:
            df = util.df(bars)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            chunks.append(df.set_index("date")[["open", "high", "low", "close", "volume"]])
        cursor -= timedelta(days=chunk_days)
        remaining_days -= chunk_days

    if not chunks:
        raise IBTransientError(
            f"fetch_historical_bars_paginated returned empty for {contract.symbol} ({bar_size})", 162,
        )
    result = pd.concat(chunks).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    if len(result) < bars_to_copy:
        raise IBTransientError(
            f"fetch_historical_bars_paginated returned only {len(result)} bars for "
            f"{contract.symbol} ({bar_size}); required {bars_to_copy}", 162,
        )
    return result.iloc[-bars_to_copy:]


async def fetch_tick(ib: IB, contract: Contract, timeout: float = 2.0) -> dict[str, float]:
    """One-shot bid/ask/last snapshot. Suspends the caller for ``timeout`` seconds."""
    ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    await asyncio.sleep(timeout)
    ib.cancelMktData(contract)
    return {
        "bid": ticker.bid or 0.0,
        "ask": ticker.ask or 0.0,
        "last": ticker.last or 0.0,
        "bid_size": ticker.bidSize or 0.0,
        "ask_size": ticker.askSize or 0.0,
    }


# ===== Orders =====
#
# @with_retry is intentionally NOT applied to placement: ib.placeOrder is sync and
# raises only on socket-level failures. Order rejections arrive asynchronously via
# errorEvent and would otherwise re-submit already-accepted orders on transient
# reconnects, producing duplicate positions.

async def place_market_order(ib: IB, contract: Contract, action: str, quantity: float,
                             magic: int) -> Trade:
    order = MarketOrder(action, quantity)
    order.orderRef = make_order_ref(magic)
    trade = ib.placeOrder(contract, order)
    await asyncio.sleep(0.2)
    return trade


async def place_limit_order(ib: IB, contract: Contract, action: str, quantity: float,
                            limit_price: float, magic: int) -> Trade:
    order = LimitOrder(action, quantity, limit_price)
    order.orderRef = make_order_ref(magic)
    trade = ib.placeOrder(contract, order)
    await asyncio.sleep(0.2)
    return trade


async def place_stop_order(ib: IB, contract: Contract, action: str, quantity: float,
                           stop_price: float, magic: int) -> Trade:
    order = StopOrder(action, quantity, stop_price)
    order.orderRef = make_order_ref(magic)
    trade = ib.placeOrder(contract, order)
    await asyncio.sleep(0.2)
    return trade


async def place_bracket_order(ib: IB, contract: Contract, action: str, quantity: float,
                              limit_price: Optional[float], take_profit: float, stop_loss: float,
                              magic: int) -> list[Trade]:
    """Atomic parent + TP + SL bracket. Returns the three Trade objects.

    Pre-assigns ``parent.orderId`` via ``ib.client.getReqId()`` so children can
    reference it before any placeOrder call.
    """
    ref = make_order_ref(magic)
    parent = LimitOrder(action, quantity, limit_price) if limit_price else MarketOrder(action, quantity)
    parent.orderId = ib.client.getReqId()
    parent.orderRef = ref
    parent.transmit = False

    tp_action = "SELL" if action == "BUY" else "BUY"
    tp = LimitOrder(tp_action, quantity, take_profit)
    tp.orderRef = ref
    tp.parentId = parent.orderId
    tp.transmit = False

    sl = StopOrder(tp_action, quantity, stop_loss)
    sl.orderRef = ref
    sl.parentId = parent.orderId
    sl.transmit = True

    trades = [ib.placeOrder(contract, o) for o in (parent, tp, sl)]
    await asyncio.sleep(0.2)
    bad = [t for t in trades if t.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive")]
    if bad:
        for t in trades:
            if t.orderStatus.status not in ("Cancelled", "ApiCancelled", "Filled"):
                try:
                    ib.cancelOrder(t.order)
                except Exception:
                    pass
        raise IBPermanentError(
            f"Bracket order may be rejected: {len(bad)}/3 leg(s) appear inactive at 0.2 s. "
            f"Statuses: {[t.orderStatus.status for t in trades]}. "
            "Cancelled any remaining live legs.",
            0,
        )
    return trades


async def close_position(ib: IB, position: dict, magic: int) -> Trade:
    """Close by market order in the opposite direction of signed position."""
    size = position["position"]
    if size == 0.0:
        raise ValueError("Cannot close zero-size position")
    action = "SELL" if size > 0 else "BUY"
    return await place_market_order(ib, position["contract"], action, abs(size), magic)


async def modify_order(ib: IB, trade: Trade, limit_price: Optional[float] = None,
                       stop_price: Optional[float] = None, quantity: Optional[float] = None) -> Trade:
    order = trade.order
    if limit_price is not None:
        order.lmtPrice = limit_price
    if stop_price is not None:
        order.auxPrice = stop_price
    if quantity is not None:
        order.totalQuantity = quantity
    return ib.placeOrder(trade.contract, order)


async def cancel_order(ib: IB, trade: Trade, timeout: float = 5.0) -> bool:
    """Cancel an order and wait for broker acknowledgement."""
    ib.cancelOrder(trade.order)
    try:
        await asyncio.wait_for(trade.cancelledEvent, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"Cancel timeout for order {trade.order.orderId} — status: {trade.orderStatus.status}"
        )
    return trade.orderStatus.status in ("Cancelled", "ApiCancelled")


def get_pending_orders(ib: IB, symbol: str, magic: int, con_id: Optional[int] = None,
                       action_filter: Optional[str] = None) -> list[dict]:
    """Return open orders for the given strategy.

    ``con_id`` filter is preferred over symbol — different contracts (e.g. futures
    expiries, options) share a root symbol but have distinct conIds. ``action_filter``
    excludes opposite-side protective stops from entry-only counts.
    """
    ref = make_order_ref(magic)

    def _matches(t) -> bool:
        if t.order.orderRef != ref:
            return False
        if action_filter is not None and t.order.action != action_filter:
            return False
        return t.contract.conId == con_id if con_id is not None else t.contract.symbol == symbol

    return [{
        "order_id": t.order.orderId,
        "action": t.order.action,
        "order_type": t.order.orderType,
        "qty": t.order.totalQuantity,
        "lmt_price": getattr(t.order, "lmtPrice", 0.0),
        "stop_price": getattr(t.order, "auxPrice", 0.0),
        "status": t.orderStatus.status,
        "trade": t,
    } for t in ib.openTrades() if _matches(t)]


async def cancel_all_pending_orders(ib: IB, symbol: str, magic: int,
                                    con_id: Optional[int] = None) -> list[int]:
    cancelled: list[int] = []
    for p in get_pending_orders(ib, symbol, magic, con_id=con_id):
        if await cancel_order(ib, p["trade"]):
            cancelled.append(p["order_id"])
    return cancelled


# ===== Positions (diagnostic; prefer IBPositionCache for hot paths) =====

def get_positions(ib: IB, symbol: str, magic: int, con_id: Optional[int] = None) -> list[dict]:
    """Session-fill-derived position estimate. NOT broker-authoritative.

    Use ``IBPositionCache.get_open()`` in production code paths. Reversal handling
    is intentionally simplified; avg_cost is unreliable after a direction change.
    """
    ref = make_order_ref(magic)
    net: dict[int, dict] = {}
    for fill in ib.fills():
        if fill.execution.orderRef != ref:
            continue
        if con_id is not None:
            if fill.contract.conId != con_id:
                continue
        else:
            if fill.contract.symbol != symbol:
                continue
        conId = fill.contract.conId
        signed = fill.execution.shares if fill.execution.side == "BOT" else -fill.execution.shares
        entry = net.setdefault(conId, {
            "contract": fill.contract, "position": 0.0, "cost_basis": 0.0,
        })
        entry["position"] += signed
        entry["cost_basis"] += signed * fill.execution.price
    return [{
        "contract": v["contract"], "position": v["position"],
        "avg_cost": v["cost_basis"] / v["position"] if v["position"] else 0.0,
    } for v in net.values() if abs(v["position"]) > 1e-9]


# ===== Account =====

def get_account_summary(ib: IB) -> dict[str, float]:
    """Return account metrics {NetLiquidation, AvailableFunds, MaintMarginReq, ...}."""
    out: dict[str, float] = {}
    for v in ib.accountSummary():
        try:
            out[v.tag] = float(v.value)
        except ValueError:
            pass
    return out
