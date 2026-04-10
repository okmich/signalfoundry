"""
Market Data Downloader - Convenient wrapper for downloading historical MT5 data.

Combines env_loader with MT5MarketDataFetcher for easy multi-broker data downloads.

Usage:
    # Command line
    python market_data_downloader.py --broker deriv --account demo \\
        --symbols "EURUSD,GBPUSD" --timeframe H1 --start-date "2024-01-01" \\
        --output ./data

    # In code
    downloader = MarketDataDownloader('deriv', 'demo')
    downloader.download(['EURUSD', 'GBPUSD'], timeframe='H1',
                       start_date='2024-01-01', output_dir='./data')
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any, Union

import MetaTrader5 as mt5

from okmich_quant_core.env_loader import load_broker_env, EnvLoader
from okmich_quant_mt5.utils.mt5_data_fetcher import MT5MarketDataFetcher, WriteMode

logger = logging.getLogger(__name__)


class MarketDataDownloader:
    """
    Convenient wrapper for downloading historical MT5 market data.

    Handles environment loading, configuration, and orchestrates data download
    for multiple symbols and timeframes.
    """

    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }

    def __init__(self, broker: str = None, account_type: str = "demo", base_dir: str = None, auto_load: bool = False):
        """
        Initialize MarketDataDownloader.

        Args:
            broker: Broker name (e.g., 'deriv', 'icmarkets')
            account_type: Account type ('demo' or 'live')
            base_dir: Base directory for .env files (defaults to current working directory)
            auto_load: If True and broker is None, auto-detect from environment/CLI (not recommended for production)
        """
        self.broker = broker
        self.account_type = account_type
        self.base_dir = base_dir
        self.env_loaded = False

        if broker or auto_load:
            self._load_environment()

    def _load_environment(self) -> bool:
        """Load environment configuration."""
        try:
            if self.broker:
                loader = EnvLoader(self.base_dir)
                success = loader.load(self.broker, self.account_type)
            else:
                success = load_broker_env(base_dir=self.base_dir)

            if not success:
                logger.error("Failed to load environment configuration")
                return False

            self.env_loaded = True
            logger.info(f"Environment loaded: {os.getenv('BROKER_NAME', self.broker)} {os.getenv('ACCOUNT_TYPE', self.account_type)}")
            return True

        except Exception as e:
            logger.error(f"Error loading environment: {e}")
            return False

    @staticmethod
    def _coerce_write_mode(write_mode: Union[WriteMode, str]) -> WriteMode:
        """Coerce a string or WriteMode into a validated WriteMode enum."""
        if isinstance(write_mode, WriteMode):
            return write_mode
        try:
            return WriteMode(write_mode)
        except ValueError:
            raise ValueError(f"Invalid write_mode: '{write_mode}'. Must be one of: {[m.value for m in WriteMode]}")

    def _build_config(self, symbols: List[str], timeframe: str, start_date: str, output_dir: str, write_mode: Union[WriteMode, str] = WriteMode.PARQUET) -> Dict[str, Any]:
        """Build configuration dict for MT5MarketDataFetcher."""
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {list(self.TIMEFRAMES.keys())}")

        write_mode = self._coerce_write_mode(write_mode)

        # Validate required env vars before casting to catch all missing fields
        env_vars = {
            "TERMINAL_PATH": os.getenv("TERMINAL_PATH"),
            "LOGIN_SERVER": os.getenv("LOGIN_SERVER"),
            "LOGIN_ID": os.getenv("LOGIN_ID"),
            "LOGIN_PASSWORD": os.getenv("LOGIN_PASSWORD"),
        }
        missing = [k for k, v in env_vars.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

        return {
            "TERMINAL_PATH": env_vars["TERMINAL_PATH"],
            "LOGIN_SERVER": env_vars["LOGIN_SERVER"],
            "LOGIN_ID": int(env_vars["LOGIN_ID"]),
            "LOGIN_PASSWORD": env_vars["LOGIN_PASSWORD"],
            "root_dest_path": output_dir,
            "SYMBOLS": ",".join(symbols),
            "START_DATE": start_date,
            "TIMEFRAME": self.TIMEFRAMES[timeframe],
            "WRITE_MODE": write_mode.value,
        }

    def download(self, symbols: List[str], timeframe: str = "H1", start_date: str = "2020-01-01", output_dir: str = "./market_data", write_mode: Union[WriteMode, str] = WriteMode.PARQUET) -> Dict[str, bool]:
        """
        Download historical market data for specified symbols.

        Args:
            symbols: List of symbol names (e.g., ['EURUSD', 'GBPUSD'])
            timeframe: Timeframe string (default: 'H1')
            start_date: Start date for data download (default: '2020-01-01')
            output_dir: Output directory for data files (default: './market_data')
            write_mode: Storage format — WriteMode enum or string 'parquet'/'hdf5' (default: WriteMode.PARQUET)

        Returns:
            Dict mapping symbol name to success status (True/False)
        """
        if not symbols:
            raise ValueError("No symbols specified. Provide at least one valid symbol.")

        if not self.env_loaded:
            logger.error("Environment not loaded. Cannot download data.")
            return {s: False for s in symbols}

        write_mode = self._coerce_write_mode(write_mode)

        try:
            config = self._build_config(symbols, timeframe, start_date, output_dir, write_mode)

            logger.info("=" * 70)
            logger.info("MARKET DATA DOWNLOAD")
            logger.info("=" * 70)
            logger.info(f"Broker: {os.getenv('BROKER_NAME', self.broker)}")
            logger.info(f"Account: {os.getenv('ACCOUNT_TYPE', self.account_type)}")
            logger.info(f"Symbols: {', '.join(symbols)}")
            logger.info(f"Timeframe: {timeframe}")
            logger.info(f"Start date: {start_date}")
            logger.info(f"Output: {output_dir}")
            logger.info(f"Format: {write_mode.value}")
            logger.info("=" * 70)

            fetcher = MT5MarketDataFetcher(config)
            try:
                results = fetcher.fetch_and_save_data()
            finally:
                fetcher.shutdown_mt5()

            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            if total_count == 0:
                raise RuntimeError("No symbols were processed during download.")

            if success_count == total_count:
                logger.info(f"Download completed successfully: {success_count}/{total_count} symbols")
            elif success_count == 0:
                logger.error(f"Download failed: 0/{total_count} symbols succeeded")
            else:
                logger.warning(f"Download completed with failures: {success_count}/{total_count} symbols succeeded")

            return results

        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            return {s: False for s in symbols}

    def download_multiple_timeframes(self, symbols: List[str], timeframes: List[str], start_date: str = "2020-01-01", output_dir: str = "./market_data", write_mode: Union[WriteMode, str] = WriteMode.PARQUET) -> Dict[str, Dict[str, bool]]:
        """
        Download data for multiple timeframes, reusing a single MT5 connection.

        Args:
            symbols: List of symbol names
            timeframes: List of timeframe strings (e.g., ['M5', 'H1', 'D1'])
            start_date: Start date for downloads
            output_dir: Output directory base
            write_mode: Storage format — WriteMode enum or string 'parquet'/'hdf5'

        Returns:
            Dict mapping timeframe to per-symbol success dict
        """
        if not symbols:
            raise ValueError("No symbols specified. Provide at least one valid symbol.")

        if not self.env_loaded:
            logger.error("Environment not loaded. Cannot download data.")
            return {tf: {s: False for s in symbols} for tf in timeframes}

        if not timeframes:
            raise ValueError("No timeframes specified. Provide at least one valid timeframe.")

        write_mode = self._coerce_write_mode(write_mode)

        # Validate all timeframes up front
        for tf in timeframes:
            if tf not in self.TIMEFRAMES:
                raise ValueError(f"Invalid timeframe: {tf}. Valid options: {list(self.TIMEFRAMES.keys())}")

        results = {}
        logger.info(f"\nDownloading {len(timeframes)} timeframes for {len(symbols)} symbols...")

        # Build config and initialize MT5 once
        first_config = self._build_config(symbols, timeframes[0], start_date, output_dir, write_mode)
        fetcher = MT5MarketDataFetcher(first_config)

        try:
            for tf in timeframes:
                logger.info(f"\n--- Processing timeframe: {tf} ---")
                config = self._build_config(symbols, tf, start_date, output_dir, write_mode)
                fetcher.reconfigure(config)

                tf_results = fetcher.fetch_and_save_data()
                if not tf_results:
                    raise RuntimeError(f"No symbols were processed for timeframe '{tf}'.")
                results[tf] = tf_results

                success_count = sum(1 for v in tf_results.values() if v)
                if success_count == 0:
                    logger.error(f"Timeframe {tf}: 0/{len(tf_results)} symbols succeeded")
                elif success_count < len(tf_results):
                    logger.warning(f"Timeframe {tf}: {success_count}/{len(tf_results)} symbols succeeded")
        finally:
            fetcher.shutdown_mt5()

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        for tf, tf_results in results.items():
            success_count = sum(1 for v in tf_results.values() if v)
            total = len(tf_results)
            status = "ALL OK" if success_count == total else f"{success_count}/{total} OK"
            logger.info(f"{tf:6} ... {status}")
        logger.info("=" * 70)

        return results

    @staticmethod
    def list_available_brokers(base_dir: str = None):
        """List available broker configurations from the specified directory."""
        loader = EnvLoader(base_dir)
        loader.print_available_configs()


def main():
    """Command line interface for market data downloader."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(
        description="Download historical MT5 market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download EURUSD H1 data from Deriv demo
  fetch-mt5-data --broker deriv --account demo \\
      --symbols EURUSD --timeframe H1 --start-date 2024-01-01

  # Download multiple symbols and timeframes
  fetch-mt5-data --broker icmarkets --account demo \\
      --symbols "EURUSD,GBPUSD,USDJPY" --timeframes "M5,H1,D1" \\
      --start-date 2023-01-01 --output ./my_data

  # Use custom directory for .env files
  fetch-mt5-data --broker deriv --account demo \\
      --symbols EURUSD --timeframe H1 \\
      --env-dir /path/to/env/files

  # List available broker configurations
  fetch-mt5-data --list-brokers

  # List brokers from custom directory
  fetch-mt5-data --list-brokers --env-dir ./playground/systems
        """,
    )

    parser.add_argument("--broker", type=str, help="Broker name (e.g., deriv, icmarkets)")
    parser.add_argument("--account", type=str, default="demo", choices=["demo", "live"], help="Account type (default: demo)")
    parser.add_argument("--symbols", type=str, help='Comma-separated list of symbols (e.g., "EURUSD,GBPUSD")')
    parser.add_argument("--timeframe", "--timeframes", type=str, dest="timeframes", help='Timeframe or comma-separated timeframes (e.g., "H1" or "M5,H1,D1")')
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (format: YYYY-MM-DD, default: 2020-01-01)")
    parser.add_argument("--output", type=str, default="./market_data", help="Output directory (default: ./market_data)")
    parser.add_argument("--format", type=str, default="parquet", choices=[m.value for m in WriteMode], help="Storage format (default: parquet)")
    parser.add_argument("--env-dir", type=str, default=None, help="Directory containing .env files (default: current directory)")
    parser.add_argument("--list-brokers", action="store_true", help="List available broker configurations")

    args = parser.parse_args()

    if args.list_brokers:
        MarketDataDownloader.list_available_brokers(base_dir=args.env_dir)
        return

    if not args.broker:
        parser.error("--broker is required (or use --list-brokers to see available brokers)")
    if not args.symbols:
        parser.error("--symbols is required")
    if not args.timeframes:
        parser.error("--timeframe is required")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols produced no valid symbol names after parsing")
    timeframes = [tf.strip().upper() for tf in args.timeframes.split(",") if tf.strip()]
    if not timeframes:
        parser.error("--timeframe produced no valid timeframes after parsing")
    write_mode = WriteMode(args.format)

    downloader = MarketDataDownloader(args.broker, args.account, base_dir=args.env_dir)

    if len(timeframes) == 1:
        results = downloader.download(symbols=symbols, timeframe=timeframes[0], start_date=args.start_date, output_dir=args.output, write_mode=write_mode)
        all_ok = all(results.values())
        sys.exit(0 if all_ok else 1)
    else:
        results = downloader.download_multiple_timeframes(symbols=symbols, timeframes=timeframes, start_date=args.start_date, output_dir=args.output, write_mode=write_mode)
        all_ok = all(v for tf_results in results.values() for v in tf_results.values())
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
