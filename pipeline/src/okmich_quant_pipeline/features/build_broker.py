"""
build_broker.py — CLI entry point to build processed datasets for a broker.

Examples:
    # Build all symbols discovered in the broker's data directory
    build-feature-dataset --broker DerivSVG-Server

    # Build specific symbols
    build-feature-dataset --broker FXPIG-Server --symbols "EURUSD.r,GBPUSD.r,XAUUSD.r"

    # Build TopOneTrader with custom horizon and window
    build-feature-dataset --broker TopOneTrader-MT5 --horizon 6 --window 14

    # Build without vol-normalizing the label
    build-feature-dataset --broker FXPIG-Server --no-vol-normalize

    # Custom data paths
    build-feature-dataset --broker TopOneTrader-MT5 \\
        --raw-dir D:/data_dump/market_data/raw \\
        --output-dir D:/data_dump/feature_data \\
        --metastore D:/data_dump/market_data/raw/labelling_metastore.json
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_RAW_DIR = "D:/data_dump/market_data/raw"
DEFAULT_OUTPUT_DIR = "D:/data_dump/feature_data"
DEFAULT_METASTORE = "D:/data_dump/market_data/raw/labelling_metastore.json"
DEFAULT_TIMEFRAME = 5
DEFAULT_HORIZON = 12
DEFAULT_WINDOW = 20


def discover_symbols(raw_dir: str, broker: str, timeframe: int) -> list[str]:
    """Discover all parquet symbols in the broker's data directory."""
    folder = Path(raw_dir) / broker / str(timeframe)
    if not folder.exists():
        raise FileNotFoundError(f"Data directory not found: {folder}")
    symbols = [p.stem for p in sorted(folder.glob("*.parquet"))]
    # Filter out metadata/summary files
    symbols = [s for s in symbols if not s.startswith("_")]
    return symbols


def main():
    parser = argparse.ArgumentParser(
        description="Build processed feature+label datasets from raw OHLCV parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--broker", required=True, help="Broker/server name (e.g. TopOneTrader-MT5)")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols. If omitted, all symbols in the data directory are built.",
    )
    parser.add_argument("--timeframe", type=int, default=DEFAULT_TIMEFRAME, help=f"Timeframe in minutes (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help=f"Forward return horizon in bars (default: {DEFAULT_HORIZON})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help=f"Default lookback window for indicators (default: {DEFAULT_WINDOW})")
    parser.add_argument("--raw-dir", type=str, default=DEFAULT_RAW_DIR, help=f"Root directory for raw parquet files (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory for processed datasets (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--metastore", type=str, default=DEFAULT_METASTORE, help=f"Path to labelling_metastore.json (default: {DEFAULT_METASTORE})")
    parser.add_argument("--no-vol-normalize", action="store_true", help="Disable vol-normalization of the label (raw log-return instead)")

    args = parser.parse_args()

    os.environ["SYMBOL_METASTORE_FILE"] = args.metastore

    # Resolve symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        logger.info(f"No --symbols provided. Discovering all symbols for {args.broker} / M{args.timeframe}...")
        symbols = discover_symbols(args.raw_dir, args.broker, args.timeframe)
        logger.info(f"Found {len(symbols)} symbols: {symbols}")

    if not symbols:
        logger.error("No symbols found. Exiting.")
        sys.exit(1)

    builder = DatasetBuilder(
        broker=args.broker,
        timeframe=args.timeframe,
        raw_data_root=args.raw_dir,
        output_dir=args.output_dir,
        horizon=args.horizon,
        default_window=args.window,
        vol_normalize_label=not args.no_vol_normalize,
    )

    logger.info("=" * 60)
    logger.info(f"Broker:    {args.broker}")
    logger.info(f"Timeframe: M{args.timeframe}")
    logger.info(f"Horizon:   {args.horizon} bars ({args.horizon * args.timeframe} min)")
    logger.info(f"Window:    {args.window}")
    logger.info(f"Symbols:   {len(symbols)}")
    logger.info(f"Output:    {args.output_dir}")
    logger.info("=" * 60)

    results = builder.build_all(symbols)

    # Summary
    successes = {s: p for s, p in results.items() if not isinstance(p, Exception)}
    failures = {s: e for s, e in results.items() if isinstance(e, Exception)}

    print()
    print(f"{'Symbol':<25} {'Rows':>8}  {'Features':>8}  {'NaN':>5}  Status")
    print("-" * 62)

    for sym, path in successes.items():
        df = pd.read_parquet(path)
        feat_cols = [c for c in df.columns if c.startswith(("feat_", "tm_", "temporal_", "candle_"))]
        nans = df[feat_cols + ["label"]].isna().sum().sum()
        print(f"{sym:<25} {len(df):>8,}  {len(feat_cols):>8}  {nans:>5}  OK")

    for sym, err in failures.items():
        print(f"{sym:<25} ERROR: {err}")

    print()
    logger.info(f"Done. {len(successes)}/{len(results)} succeeded.")

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
