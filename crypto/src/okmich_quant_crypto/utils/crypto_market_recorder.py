"""``record-crypto-market`` CLI - record order books, trades, tickers and liquidations from any CCXT exchange.

    record-crypto-market --exchange bybit --symbols BTC/USDT:USDT,ETH/USDT:USDT --output D:/crypto_data \\
        --levels 30 --interval 3

or from a JSON file holding a ``RecorderConfig`` (``--config recorder.json``). Runs until Ctrl+C, then flushes and
compacts what it holds. It needs an always-on machine: every hour it is not running is a hole in the order-book
history (recorded in the ``gaps`` stream for outages it sees; a stopped process cannot record its own absence).
"""
import argparse
import asyncio
import logging

from ..data.recorder import MarketRecorder, RecorderConfig
from ..enums import RecordStream, VenueEnvironment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="record-crypto-market", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="JSON RecorderConfig file (other options then ignored)")
    parser.add_argument("--exchange", default="bybit", help="Any CCXT exchange id with WebSocket support")
    parser.add_argument("--symbols", default=None, help="Comma-separated CCXT symbols, e.g. BTC/USDT:USDT,ETH/USDT")
    parser.add_argument("--output", default=None, help="Root directory for the recorded parquet files")
    parser.add_argument("--streams", default=",".join(s.value for s in RecordStream),
                        help=f"Comma-separated streams (default: all of {[s.value for s in RecordStream]})")
    parser.add_argument("--levels", type=int, default=50, help="Book levels per side kept per snapshot (default 50)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between book snapshots (default 1)")
    parser.add_argument("--ticker-interval", type=float, default=None,
                        help="Seconds between ticker samples (default: --interval)")
    parser.add_argument("--subscribe-depth", type=int, default=None,
                        help="Book depth to subscribe at when the venue only accepts specific values (e.g. 200)")
    parser.add_argument("--flush-seconds", type=float, default=60.0, help="Seconds between writes to disk")
    parser.add_argument("--environment", default=VenueEnvironment.LIVE.value,
                        choices=[e.value for e in VenueEnvironment])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.config:
        cfg = RecorderConfig.load_from_file(args.config)
    else:
        if not args.symbols or not args.output:
            raise SystemExit("--symbols and --output are required without --config")
        cfg = RecorderConfig(exchange_id=args.exchange, environment=VenueEnvironment(args.environment),
                             symbols=args.symbols.split(","), output_dir=args.output,
                             streams=[RecordStream(s.strip()) for s in args.streams.split(",") if s.strip()],
                             book_levels=args.levels, book_interval_seconds=args.interval,
                             ticker_interval_seconds=args.ticker_interval, book_subscribe_depth=args.subscribe_depth,
                             flush_seconds=args.flush_seconds)
    asyncio.run(MarketRecorder(cfg).run())
