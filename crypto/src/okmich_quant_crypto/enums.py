"""Fixed value sets for the crypto package.

Every choice the configuration or the venue profiles make is one of these, never a bare string: a typo in a string
literal is a silent wrong branch, a typo in an enum member is an import-time error.
"""
from enum import StrEnum


class MarketType(StrEnum):
    """The market a strategy trades. Linear perpetuals are USDT (or other stablecoin) margined swaps."""
    SPOT = "spot"
    LINEAR_PERP = "linear_perp"


class VenueEnvironment(StrEnum):
    """Which copy of the venue the account lives on.

    ``DEMO`` is paper trading against LIVE prices (e.g. Bybit demo trading); ``TESTNET`` is a separate venue with its
    own order book and prices. Testnet books are thin and unrealistic: use them for plumbing tests only, never for any
    measurement of fills or slippage.
    """
    LIVE = "live"
    TESTNET = "testnet"
    DEMO = "demo"


class StopMode(StrEnum):
    """Where stop-loss / take-profit levels live.

    ``NATIVE`` - exchange-side orders or position stops; they work while this process is down.
    ``MANAGED`` - this process watches prices and sends a reduce-only market close; NOT live while disconnected.
    ``AUTO`` - native where the venue profile supports it for the market type, otherwise managed.
    """
    NATIVE = "native"
    MANAGED = "managed"
    AUTO = "auto"


class FeedMode(StrEnum):
    """How bars, orders and positions reach the strategy: pushed over WebSocket or pulled by REST polling."""
    STREAM = "stream"
    POLL = "poll"


class SizingUnit(StrEnum):
    """What ``PositionSizingConfig.units`` means for a FIXED size (there are no lots in crypto)."""
    BASE_QTY = "base_qty"
    QUOTE_NOTIONAL = "quote_notional"
    CONTRACTS = "contracts"


class MarginMode(StrEnum):
    CROSS = "cross"
    ISOLATED = "isolated"


class MarginModeScope(StrEnum):
    """What a margin-mode change applies to on a venue. On Bybit's unified account it is the WHOLE account."""
    ACCOUNT = "account"
    SYMBOL = "symbol"


class PositionMode(StrEnum):
    """Only one-way is supported: in hedge mode a symbol carries two positions and ``magic`` cannot separate them."""
    ONE_WAY = "one_way"


class StopTrigger(StrEnum):
    """The price a native stop triggers on. MARK avoids single-venue wicks but will not match a last-price chart."""
    LAST = "last"
    MARK = "mark"
    INDEX = "index"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderRole(StrEnum):
    """Why this system sent an order. Drives close attribution and what shutdown may cancel."""
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class Dataset(StrEnum):
    """Historical datasets the downloader reads through CCXT's unified REST methods."""
    OHLCV = "ohlcv"
    MARK_OHLCV = "mark_ohlcv"
    INDEX_OHLCV = "index_ohlcv"
    PREMIUM_INDEX_OHLCV = "premium_index_ohlcv"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LONG_SHORT_RATIO = "long_short_ratio"
    TRADES = "trades"


class RecordStream(StrEnum):
    """Live streams the market recorder captures through CCXT's unified WebSocket methods."""
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    TICKER = "ticker"
    LIQUIDATIONS = "liquidations"


class FillKind(StrEnum):
    """What produced a fill, as far as the venue tells us. Generic venues report everything as TRADE."""
    TRADE = "trade"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    LIQUIDATION = "liquidation"
    ADL = "adl"
    FUNDING = "funding"
    SETTLEMENT = "settlement"
