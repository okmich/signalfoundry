"""IB contract definitions and resolution.

Encapsulates broker-specific fields (secType, exchange, primaryExch, lastTrade...)
so that core ``StrategyConfig`` stays broker-agnostic.
"""
from dataclasses import dataclass
from enum import StrEnum

from ib_async import Contract, IB, Stock, Forex, Future, CFD, Option


class SecType(StrEnum):
    STK = "STK"
    CASH = "CASH"
    FUT = "FUT"
    CFD = "CFD"
    OPT = "OPT"
    CRYPTO = "CRYPTO"  # PAXOS: BTC, ETH, ... (cash account only)


@dataclass
class IBContractConfig:
    sec_type: SecType
    exchange: str
    currency: str
    primary_exchange: str = ""
    last_trade_date: str = ""
    multiplier: str = ""
    trading_class: str = ""
    strike: float = 0.0
    right: str = ""


DEFAULT_WHAT_TO_SHOW: dict[SecType, str] = {
    SecType.STK: "TRADES",
    SecType.CASH: "MIDPOINT",
    SecType.FUT: "TRADES",
    SecType.CFD: "MIDPOINT",
    SecType.OPT: "TRADES",
    SecType.CRYPTO: "AGGTRADES",  # PAXOS exposes aggregated trades only
}

DEFAULT_USE_RTH: dict[SecType, bool] = {
    SecType.STK: True,
    SecType.CASH: False,
    SecType.FUT: False,
    SecType.CFD: False,
    SecType.OPT: True,
    SecType.CRYPTO: False,        # 24/7
}


def make_order_ref(magic: int) -> str:
    """Build the orderRef tag used to namespace a strategy's orders."""
    return f"sf_{magic}"


async def resolve_contract(ib: IB, symbol: str, cfg: IBContractConfig) -> Contract:
    """Build and qualify a Contract. Raises ValueError if unresolvable or ambiguous."""
    if cfg.sec_type == SecType.CASH:
        symbol = symbol.replace(".", "")
    if cfg.sec_type == SecType.STK:
        raw = Stock(symbol, cfg.exchange, cfg.currency, primaryExchange=cfg.primary_exchange)
    elif cfg.sec_type == SecType.CASH:
        raw = Forex(symbol, cfg.exchange or "IDEALPRO", currency=cfg.currency)
    elif cfg.sec_type == SecType.FUT:
        raw = Future(symbol, cfg.last_trade_date, cfg.exchange, currency=cfg.currency,
                     multiplier=cfg.multiplier, tradingClass=cfg.trading_class)
    elif cfg.sec_type == SecType.CFD:
        raw = CFD(symbol, cfg.exchange, cfg.currency)
    elif cfg.sec_type == SecType.OPT:
        raw = Option(symbol, cfg.last_trade_date, cfg.strike, cfg.right, cfg.exchange,
                     currency=cfg.currency)
    elif cfg.sec_type == SecType.CRYPTO:
        # IB crypto via PAXOS: secType='CRYPTO', exchange defaults to 'PAXOS'.
        raw = Contract(symbol=symbol, secType="CRYPTO",
                       exchange=cfg.exchange or "PAXOS", currency=cfg.currency)
    else:
        raw = Contract(symbol=symbol, secType=str(cfg.sec_type), exchange=cfg.exchange,
                       currency=cfg.currency, primaryExchange=cfg.primary_exchange,
                       lastTradeDateOrContractMonth=cfg.last_trade_date,
                       multiplier=cfg.multiplier, tradingClass=cfg.trading_class,
                       strike=cfg.strike, right=cfg.right)
    details = await ib.reqContractDetailsAsync(raw)
    if not details:
        raise ValueError(f"No contract found for {symbol} ({cfg.sec_type})")
    if len(details) > 1:
        raise ValueError(
            f"Ambiguous contract for {symbol}: {len(details)} matches. "
            "Narrow via primary_exchange, last_trade_date, or trading_class."
        )
    return details[0].contract
