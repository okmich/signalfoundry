# okmich-quant-crypto

Crypto exchange integration for signalfoundry via [CCXT](https://github.com/ccxt/ccxt): spot and USDT-margined linear
perpetuals, event-driven like `okmich-quant-ib`, built on `okmich-quant-core` without changing it.

Trading runs only on the exchanges in the [supported list](#supported-exchanges). Each one is added deliberately, on
its own feature branch, with a hand-written venue profile and integration tests. The [market-data tools](#market-data)
use only CCXT's unified API and work with any CCXT exchange.

## Quick start

The runnable example lives in the lab repo: `signalfoundry-lab/examples/crypto_examples/` (`run.py` +
`system.bybit.demo.json`) runs one Bybit demo account:

```
# signalfoundry-lab/examples/crypto_examples/.env.bybit.demo   (git-ignored there)
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
```

```python
from okmich_quant_crypto import CryptoEventLoop, CryptoSystemConfig, GenericBasicCryptoStrategy

system = CryptoSystemConfig.load_from_file("system.bybit.demo.json")
loop = CryptoEventLoop(system.venue, runner_name=system.name)
for cfg in system.all_strategies():
    loop.add_strategy(GenericBasicCryptoStrategy(cfg, MySignal(**cfg.signal_params)))
loop.start()   # Ctrl+C: working entry/exit orders cancelled, protective stops KEPT on the venue
```

Credentials come from the environment variables NAMED in the venue config; they are never logged or serialised. The
env file is loaded with `python-dotenv` directly, because `core.env_loader.EnvLoader.load` validates MT5 variables.

Market data: see [Market data](#market-data) (`fetch-crypto-data`, `record-crypto-market`).

## Configuration

`CryptoSystemConfig` = `name` + one `venue` + `strategy` | `strategies`.

### Venue (`CryptoVenueConfig`) - shared by every strategy on one account

| field | default | meaning |
|---|---|---|
| `exchange_id` | - | a [supported exchange](#supported-exchanges) id (`bybit`); anything else is rejected at load |
| `environment` | **required** | `live`, `testnet`, `demo` - no default, so LIVE is never an accident |
| `sub_account` | `main` | label of the (sub-)account the key belongs to; part of the isolation key |
| `api_key_env`, `secret_env`, `password_env` | `CRYPTO_API_KEY`, `CRYPTO_API_SECRET`, - | env var NAMES |
| `margin_mode` | `null` | `cross` / `isolated`, applied at startup. On Bybit this changes the **whole account** |
| `state_dir` | `.crypto_state` | lifecycle ids, order roles, managed stop levels, spot ledger |
| `rate_limit_ms`, `ccxt_options` | - | CCXT overrides |

### Strategy (`CryptoStrategyConfig`, a subclass of `core.StrategyConfig`)

| field | default | meaning |
|---|---|---|
| `market_symbol` | - | CCXT symbol: `BTC/USDT` (spot) or `BTC/USDT:USDT` (perp) |
| `symbol` | derived | log-safe token core uses: `BTC/USDT:USDT` -> `BTC/USDT-USDT` (core rejects `:`) |
| `timeframe` | - | any timeframe the venue offers, up to `1d` (checked at startup) |
| `market_type` | `linear_perp` | `spot` or `linear_perp` |
| `feed_mode` | `stream` | `stream` (WebSocket) or `poll` (REST) |
| `stop_mode` | `auto` | `native`, `managed`, `auto` |
| `stop_trigger` | `last` | `last`, `mark`, `index` - the price native stops trigger on |
| `leverage` | - | perps only, set at startup |
| `sizing_unit` | `base_qty` | what FIXED `units` means: `base_qty`, `quote_notional`, `contracts` |
| `close_grace_seconds` / `close_max_wait_seconds` | 3 / 30 | POLL wake-up delay / how late a bar may be and still run the strategy |
| `position_poll_seconds` / `reconcile_seconds` | 5 / 60 | REST polling (POLL) / REST reconciliation (STREAM) |
| `managed_stop_poll_seconds` | 1 | ticker polling for managed stops when there is no WebSocket ticker |

Validation worth knowing: `max_number_of_open_positions` must be 1 (a venue nets a symbol into one position);
point-based position managers need an explicit `point_size` in quote price units; `leverage` is rejected for spot;
MANAGED stops on a POLL feed slower than 5 s are rejected unless `allow_slow_managed_stops`.

**Sizing.** FIXED = `units` in `sizing_unit`. RISK_PCT_OF_EQUITY and KELLY_CRITERION size so that the loss at the stop
equals `fraction x equity` (`risk_pct` / `kelly_fraction` - the package does not estimate Kelly). Both need a stop:
explicit `stop_loss=`, or the position manager's initial stop. Orders below the venue minimum amount or notional are
rejected with a logged, notified failure - never a crash.

## Environments

| | prices | fills | use it for |
|---|---|---|---|
| `live` | real | real | trading |
| `demo` | **live** | simulated (no queue) | paper trading; passive fills look better than reality |
| `testnet` | separate, thin book | simulated | plumbing tests only - never measure anything there |

Bybit demo is switched on with `enable_demo_trading(True)` (never combined with sandbox). Demo has no UID endpoint, so
`account_id` is a non-reversible fingerprint of the API key.

## Feeds: STREAM and POLL produce identical bars

Both feed one `ClosedBarSource` contract: exactly one REST-reconciled closed bar per timeframe boundary.

- **STREAM:** `watch_ohlcv` + a close detector (Bybit's `confirm` flag, else timestamp rollover) TRIGGERS the close; the
  bar itself is re-read over REST (the last WebSocket update of a candle can be lost). A watchdog closes the bar from
  REST if the socket has not, `close_max_wait / 2` after the boundary.
- **POLL:** wake at boundary + `close_grace_seconds`, accept the candle only when the expected open timestamp is
  returned AND its close has passed; retry until `close_max_wait_seconds`, then report a missed bar.
- **Gaps** (reconnects, missed polls) are backfilled from REST. Any bar that closed more than `close_max_wait_seconds`
  ago is **stale**: it updates the price buffer but does not run the strategy or write a heartbeat - acting late on an
  old signal is worse than skipping it, and the heartbeat shows the outage honestly.
- Orders / fills / positions are account-wide streams on every venue: consumed once by the event loop, dispatched by
  symbol, and REST-reconciled every `reconcile_seconds`. Without the WebSocket capabilities they are polled.

## Stops

| mode | where levels live | survives this process being down |
|---|---|---|
| `native` | on the venue | yes |
| `managed` | here (persisted); a crossing sends a reduce-only market close | **no** - after a price-feed or process outage, 1m candles of the outage (from no earlier than the position's open and the last level change) are checked for a crossing |
| `auto` | native when the venue can protect an EXISTING position, else managed | - |

Native, per venue capability:

- **Position-level stops** (Bybit perps): entries carry attached SL/TP; changes go through `/v5/position/trading-stop` in
  Full mode, replacing both levels in one call - the position is never unprotected.
- **Standalone conditional orders** (everything else): placed when the position appears, resized when the position
  grows or shrinks, changed new-before-old (the replacement is placed before the old order is cancelled). Every stop
  operation runs under one lock; an order whose cancel cannot be confirmed is kept and retried, and untracked stop
  orders carrying our client-id prefix are cancelled on reconcile. Spot stops are cancelled BEFORE a strategy close,
  since a spot stop is a plain sell - and put back if the close then fails. Entries never attach stops here: a
  venue-created stop has no id this package could move.

The resolved stop mode is logged at startup for every strategy.

## Isolation rules

- At most one strategy per (exchange, environment, account, market type, symbol) - enforced in `CryptoSystemConfig`
  and in `CryptoEventLoop.add_strategy`, before anything connects. Perp positions are netted per symbol per account,
  so `magic` cannot separate two strategies there.
- Every order carries a client order id `sf{magic}x{token}` (letters and digits only, the strictest venue rule; CCXT
  does not validate ids, this package does).
- **Perps:** the strategy's position is the venue position for the symbol.
- **Spot:** a `SpotInventoryLedger` built only from the strategy's own fills (base-coin fees netted out), persisted, and
  caught up from fill history on startup. Sell size is `min(ledger qty, free balance)`; balance the strategy did not
  buy is never touched.

## How core's contracts are met without changing core

| core contract | crypto reality | handled here by |
|---|---|---|
| log identity rejects `:` | perp symbols are `BTC/USDT:USDT` | `market_symbol` + derived `symbol` |
| sealed sync `run()` | async, event-driven | sealed async `_on_bar_close` + per-strategy breaker (IB pattern) |
| bar labels from the epoch-minute grid | weekly candles start Monday, epoch weeks Thursday | timeframes capped at `1d` |
| position key must not be reused while a close is pending | netted perps have no position id (and Bybit's position timestamp repeats) | lifecycle id `{symbol}@{detected_ms}` from the local clock, persisted; a flip starts a new one |
| `resolve_closed_trade` is sync | fills via WS, funding via REST | resolved asynchronously, then handed to core (IB pattern) |
| `ClosedTrade` has commission + swap, no funding | fees in base / quote / third coins; funding | `commission` = -fees in quote, `swap` = funding (positive = received); a fee in an unvaluable coin marks the trade unresolved |
| `BrokerSession.disconnect()` is sync | `exchange.close()` is async | loop awaits `aclose()`, `disconnect()` returns the cached proof |
| inference log: bar / breaker / re-enable events only | stop mode per position | text log + notifier |

## Error handling

CCXT exceptions are classified most-specific-first, with venue codes first (`VenueProfile.classify_venue_error`):
`TRANSIENT` (retry with backoff), `PERMANENT`, `UNKNOWN_STATE` (an order MAY exist), `BANNED` (non-JSON 403/418/451,
revoked key - stop trading, never hammer), `NO_CHANGE`, `NOT_FOUND`, `ALREADY_DONE`, `DUPLICATE`, `CLOCK_SKEW`.
Order placement never resends blindly: every retry reuses the same client id, an ambiguous failure is followed by a
lookup by that id, and a "duplicate id" rejection means "go and find it".

## Market data

Both tools use only CCXT's unified methods - no venue endpoints, no venue file servers - so they work with any CCXT
exchange (not just the supported trading list). What history exists is the venue's business; the tools report it
honestly rather than pad or fake it.

### History: `fetch-crypto-data`

```
fetch-crypto-data BTC/USDT:USDT --exchange bybit --dataset ohlcv --timeframe 5m --start 2024-01-01 --output btc_5m.parquet
fetch-crypto-data BTC/USDT:USDT --exchange bybit --dataset funding_rate --start 2024-01-01 --output btc_funding.parquet
fetch-crypto-data BTC/USDT:USDT --exchange bybit --dataset open_interest --timeframe 5m --start 2026-09-01 --output btc_oi.parquet
```

| `--dataset` | CCXT method | `--timeframe` | columns (index `date`, UTC) |
|---|---|---|---|
| `ohlcv` | `fetch_ohlcv` | required | `open, high, low, close, volume` (index = candle open) |
| `mark_ohlcv` / `index_ohlcv` / `premium_index_ohlcv` | `fetch_mark_ohlcv` / ... | required | same; `volume` is NaN where the venue has none |
| `funding_rate` | `fetch_funding_rate_history` | - | `funding_rate` |
| `open_interest` | `fetch_open_interest_history` | required (the venue's periods, e.g. 5m..1d) | `open_interest_amount, open_interest_value` |
| `long_short_ratio` | `fetch_long_short_ratio_history` | required | `long_short_ratio` |
| `trades` | `fetch_trades` | - | `trade_id, side` (aggressor)`, price, amount, cost` - one file per UTC day in the `--output` directory |

- Every dataset is checked against `exchange.has` before any request; derivative datasets refuse spot symbols.
- Pages always carry an explicit size (`--page-limit` overrides); files resume from the last stored record, merge, and
  are replaced atomically. Candles: the forming candle is never written, off-grid candles are refused, gaps reported.
- Candles are stored as the exchange returns them (weekends included, `volume`, no `tick_volume`).
  `okmich_quant_pipeline`'s dataset builder expects `tick_volume` and drops weekends, so it cannot read them unchanged.
- **Trades:** some venues ignore `since` and return only their latest trades (Bybit does). Before downloading, the
  tool asks for trades from two different start times; the same latest page twice means the venue does not serve
  trade history, and nothing is written. A first page that starts more than `--max-start-gap-minutes` (default 60)
  after `--start` is refused too. Trades are written as each UTC day completes, so a long range needs memory for one
  day and survives a failure part-way.
- An empty page is not the end of the data: several CCXT methods ask the venue for a fixed window after `since`
  (e.g. an hour of trades), so the cursor steps over quiet windows and pre-listing ranges.
- `--full-refetch` re-downloads the whole range instead of resuming, but still merges with the stored file.
- A series that starts later than requested is reported, and the message distinguishes "the venue returned only its
  latest records" (e.g. CCXT's Bybit long/short ratio passes no start time: you get the latest 50 readings) from "listing
  date or retention limit".

### Live: `record-crypto-market`

Order-book history has no unified REST method anywhere, so the generic way to have it is to record it - from the day
the recorder starts, on a machine that is always on.

```
record-crypto-market --exchange bybit --symbols BTC/USDT:USDT,ETH/USDT:USDT --output D:/crypto_data --levels 30 --interval 3
record-crypto-market --config recorder.json        # a RecorderConfig as JSON
```

| setting | default | meaning |
|---|---|---|
| `--levels` / `book_levels` | 50 | levels per side kept in each book snapshot |
| `--interval` / `book_interval_seconds` | 1 | seconds between book snapshots, on a UTC grid (:00, :01, ... for 1 s) |
| `--ticker-interval` | = interval | seconds between ticker samples |
| `--subscribe-depth` | - | depth to subscribe at when the venue only accepts specific values (Bybit: 1/50/200/1000) |
| `--streams` | all | `order_book, trades, ticker, liquidations` |
| `--flush-seconds` | 60 | seconds between writes to disk (a crash loses at most this much) |

Streams per symbol, under `<output>/<exchange>/<SYMBOL>/<stream>/<YYYY-MM-DD>/<HH>.parquet`:

- `order_book_l{levels}_{interval}s`: `exchange_ts_ms, book_age_ms, bid_px_1, bid_sz_1, ... ask_px_N, ask_sz_N`; levels
  the book does not have are NaN; a book (or ticker) not updated for 30 s, or whose stream is down, is not sampled;
  intervals are at least 0.1 s;
- `trades`: every public trade, `trade_id, side, price, amount, cost, recv_ms`, de-duplicated by id;
- `ticker_{interval}s`: `bid, ask, bid_size, ask_size, last, mark, index, base_volume, quote_volume, ...`;
- `liquidations`: where the venue streams them for that market (perps);
- `gaps`: one row per stream outage (`stream, end_ms, error`) - mask these windows in research.

What "generic" costs: CCXT maintains the book from the venue's updates and the recorder SAMPLES it, so this is a
snapshot series, not an event-by-event replay. It suits depth, imbalance, spread and refill features at minute
horizons, not tick-level queue modelling. If the venue rejects the requested depth as a subscription value, the
recorder subscribes at the venue's default and keeps the top `book_levels` (warning if the book is thinner). Size per
symbol per day at 1 s x 50 levels: 86,400 snapshots x 202 columns, plus every trade. Rows are flushed as small
`HH.part-*.parquet` files and compacted into `HH.parquet` when the hour closes (and at shutdown; leftovers from a crash
are compacted on the next start).

## Tests

- Unit (default `python build.py --test crypto`, no network): a scripted `FakeExchange` covers the feeds (incl. STREAM ==
  POLL), precision, client ids, isolation, the spot ledger, stop-mode resolution, native cancel-replace order, managed
  stops incl. crossings while disconnected, error classification and idempotent placement, the broker session, closed-
  trade P&L with fees and funding, inference-logging parity with IB, every history dataset (paging, resume, the trades
  safeguard) and the recorder (sampling, depth fallback, gaps, hourly compaction).
- Integration (`tests/integration`, skipped unless configured; never LIVE):

  ```
  CRYPTO_IT_EXCHANGE=bybit CRYPTO_IT_ENV=demo CRYPTO_IT_API_KEY=... CRYPTO_IT_API_SECRET=...
  ```

## Supported exchanges

| exchange | `exchange_id` | spot | USDT linear perps | stops | paper environments | added |
|---|---|---|---|---|---|---|
| Bybit (v5 unified account) | `bybit` | yes | yes | native: position-level on perps, TP/SL orders on spot | `demo`, `testnet` | 2026-10 |

The list lives in `venue/registry.py`; nothing outside it can trade.

### Adding an exchange (one feature branch per exchange)

1. `venue/<id>.py`: subclass `VenueProfile` and override what CCXT gets wrong or leaves out for this venue -
   environment switch (testnet / demo), stop capabilities per market type, attached / standalone / position stop
   parameters, `stop_order_params`, client order id rule, venue error codes (`classify_venue_error`), funding history
   with its sign normalised (`fetch_funding`), fill kinds incl. liquidation / ADL / stop fills (`fill_kind`,
   `labels_stop_fills`), order lookup by client id, fill history including liquidations, account UID, margin-mode
   scope. Verify each against the installed CCXT source and the venue's API docs, not memory.
2. Register it in `venue/registry.py`.
3. Unit tests for the profile's request shapes and code mappings (see `tests/test_venue_profiles.py`), and the
   integration suite run on the venue's demo / testnet (`CRYPTO_IT_EXCHANGE=<id>`).
4. Add its row to the table above; check it is available for your jurisdiction before trading it live.

## Known limitations

- Supported exchanges: Bybit only (see [Supported exchanges](#supported-exchanges)).
- Managed stops are not live while the process is disconnected or stopped.
- Hedge (two-sided) position mode is unsupported; the account must be one-way.
- Spot shorting is unsupported; stop-entry orders are not supported in v1.
- Spot native stops on Bybit (TP/SL orders) are verified by integration tests only. If Bybit locks the balance for a
  spot TP/SL order, an SL and a TP cannot both cover the full quantity; the TP placement then fails (logged and
  notified) and only the SL protects the position.
- On Windows no event-loop policy is needed: CCXT works on the default Proactor loop (aiodns is not installed).
