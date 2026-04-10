import enum
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, Callable, Optional

import MetaTrader5 as mt5
import pandas as pd
import pytz

from okmich_quant_mt5.functions import fetch_data_date_range
from okmich_quant_mt5.timeframe_utils import timeframe_minutes_dict
from okmich_quant_utils.data_storage import DataStorage

logger = logging.getLogger(__name__)
standard_tz = pytz.timezone("Etc/UTC")
metadata_filename = "_metadata.json"

# Forex markets trade ~5 days/week, ~23h/day (maintenance breaks vary).
# Crypto trades 24/7. We use a conservative floor for forex so the heuristic
# doesn't reject valid data during holiday-shortened weeks.
_FOREX_TRADING_HOURS_PER_DAY = 20
_CRYPTO_TRADING_HOURS_PER_DAY = 24
_CRYPTO_PREFIXES = ("BTC", "ETH", "LTC", "XRP", "BNB", "SOL", "ADA", "DOGE", "DOT", "AVAX")

# If actual bars are below this fraction of the expected count, the download
# is considered implausible and rejected.
_PLAUSIBILITY_RATIO = 0.15


class WriteMode(enum.StrEnum):
    PARQUET = "parquet"
    HDF5 = "hdf5"


class Config:
    """Holds configuration settings for MT5 data fetching."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.terminal_path = config_dict.get("TERMINAL_PATH")
        self.server = config_dict.get("LOGIN_SERVER")
        self.username = config_dict.get("LOGIN_ID")
        self.password = config_dict.get("LOGIN_PASSWORD")
        self.root_dest_path = config_dict.get("root_dest_path")
        symbols_str = config_dict.get("SYMBOLS", "")
        self.symbols = [s.strip() for s in symbols_str.split(",") if s.strip()] if symbols_str else []

        # Parse start_date to datetime to ensure type consistency
        start_date_str = config_dict.get("START_DATE")
        if isinstance(start_date_str, str):
            self.start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=standard_tz)
        elif isinstance(start_date_str, datetime):
            self.start_date = start_date_str if start_date_str.tzinfo else start_date_str.replace(tzinfo=standard_tz)
        else:
            self.start_date = start_date_str  # None or other

        self.timeframe = config_dict.get("TIMEFRAME")

        raw_write_mode = config_dict.get("WRITE_MODE", "parquet")
        try:
            self.write_mode = WriteMode(raw_write_mode)
        except ValueError:
            raise ValueError(f"Invalid WRITE_MODE: '{raw_write_mode}'. Must be one of: {[m.value for m in WriteMode]}")

    def validate(self) -> bool:
        """Validates that required configuration fields are present."""
        required_fields = ["terminal_path", "server", "username", "password", "root_dest_path", "symbols", "start_date", "timeframe"]
        missing = [field for field in required_fields if not getattr(self, field, None)]

        if missing:
            logger.error(f"Missing required configuration fields: {missing}")
            return False

        if not self.symbols:
            logger.error("No symbols specified in configuration")
            return False

        return True

    @property
    def destination_path(self):
        return f"{self.root_dest_path}/{self.server}/{self.timeframe}"


class MetadataHandler:
    """Handles reading and writing metadata to a JSON file."""

    def __init__(self, folder_path: str, file_name: str = "_metadata.json"):
        self.folder_path = folder_path
        self.metadata_filename = file_name

    def read_metadata(self) -> Dict[str, Any]:
        """Reads metadata from the JSON file."""
        json_path = os.path.join(self.folder_path, self.metadata_filename)
        if not os.path.exists(json_path):
            logger.warning(f"Metadata file not found at {json_path}")
            return {}

        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                logger.error(f"Invalid metadata format in {json_path}: Expected a dictionary.")
                return {}
            logger.debug(f"Successfully read metadata from {json_path}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata file {json_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error reading metadata file {json_path}: {e}")
            return {}

    def save_metadata(self, metadata_dict: Dict[str, Any]):
        """Saves or updates metadata to the JSON file."""
        json_path = os.path.join(self.folder_path, self.metadata_filename)
        try:
            existing_metadata = self.read_metadata()
            existing_metadata.update(metadata_dict)

            # Write atomically using temp file
            temp_path = json_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(existing_metadata, f, indent=4, default=str)
            os.replace(temp_path, json_path)

            logger.debug(f"Saved/updated metadata to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {json_path}: {e}")

    def update_symbol_metadata(self, symbol: str, updates: Dict[str, Any]):
        """Update metadata for a single symbol incrementally."""
        metadata = self.read_metadata()
        if symbol not in metadata:
            metadata[symbol] = {}
        metadata[symbol].update(updates)
        self.save_metadata({symbol: metadata[symbol]})


def _estimate_expected_bars(timeframe: int, start_date: datetime, end_date: datetime, symbol: str) -> int:
    """Estimate the plausible number of bars for a date range and timeframe.

    Uses conservative assumptions so that only truly implausible downloads
    (e.g. 12 bars for 2 weeks of H1 data) get rejected.
    """
    bar_minutes = timeframe_minutes_dict.get(timeframe)
    if bar_minutes is None:
        return 0  # unknown timeframe — skip heuristic

    total_minutes = (end_date - start_date).total_seconds() / 60

    is_crypto = any(symbol.upper().startswith(prefix) for prefix in _CRYPTO_PREFIXES)
    if is_crypto:
        trading_fraction = 1.0  # 24/7
    else:
        # Forex: ~5/7 days, ~20/24 hours
        trading_fraction = (5.0 / 7.0) * (_FOREX_TRADING_HOURS_PER_DAY / 24.0)

    available_minutes = total_minutes * trading_fraction
    return int(available_minutes / bar_minutes)


def _validate_download_plausibility(df: pd.DataFrame, timeframe: int, start_date: datetime, end_date: datetime, symbol: str) -> Optional[str]:
    """Check whether the downloaded bar count is plausible.

    Returns None if plausible, or an error message string if not.
    """
    expected = _estimate_expected_bars(timeframe, start_date, end_date, symbol)
    if expected <= 0:
        return None  # can't estimate — pass through

    actual = len(df)
    ratio = actual / expected

    if ratio < _PLAUSIBILITY_RATIO:
        return (
            f"Download for {symbol} looks implausible: got {actual} bars but expected ~{expected} "
            f"(ratio {ratio:.2%}, threshold {_PLAUSIBILITY_RATIO:.0%}). "
            f"Range: {start_date} → {end_date}, timeframe minutes: {timeframe_minutes_dict.get(timeframe)}"
        )
    return None


def _merge_dataframes(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and new data, keeping latest on duplicate timestamps.

    New data is appended after existing so that keep='last' prefers the
    newer bar when timestamps collide (e.g., broker corrections).
    Deduplication is on the index (timestamp), not row content.
    """
    merged = pd.concat([existing_df, new_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _validate_post_merge(merged_df: pd.DataFrame, existing_df: Optional[pd.DataFrame], new_df: pd.DataFrame, symbol: str) -> Optional[str]:
    """Validate that the merged result is a proper superset of old + new, minus duplicates.

    Returns None if valid, or an error message string if not.
    """
    if merged_df.index.duplicated().any():
        dup_count = merged_df.index.duplicated().sum()
        return f"Post-merge validation failed for {symbol}: {dup_count} duplicate timestamps remain"

    if existing_df is not None:
        if len(merged_df) < len(existing_df):
            return (
                f"Post-merge validation failed for {symbol}: merged has {len(merged_df)} rows "
                f"but existing had {len(existing_df)} — data was lost"
            )

    if len(merged_df) < len(new_df):
        return (
            f"Post-merge validation failed for {symbol}: merged has {len(merged_df)} rows "
            f"but new data had {len(new_df)} — data was lost"
        )

    return None


class MT5MarketDataFetcher:
    """Main class to orchestrate MT5 market data fetching and storage."""

    def __init__(self, conf: Dict[str, Any], initialize: bool = True):
        self.config = Config(conf)
        os.makedirs(self.config.destination_path, exist_ok=True)
        self.metadata_handler = MetadataHandler(self.config.destination_path)
        self.data_storage = DataStorage(self.config.destination_path)
        if initialize:
            self.initialize_mt5()

    def initialize_mt5(self):
        """Initializes MetaTrader5 connection."""
        if not mt5.initialize(path=self.config.terminal_path, server=self.config.server, login=self.config.username, password=self.config.password):
            raise Exception(f"Failed to initialize MT5. Error Code: {mt5.last_error()}")
        logger.info(f"MT5 initialized successfully for {self.config.server}")

    def reconfigure(self, conf: Dict[str, Any]) -> None:
        """Reconfigure the fetcher for a new timeframe/symbol set without re-initializing MT5."""
        self.config = Config(conf)
        os.makedirs(self.config.destination_path, exist_ok=True)
        self.metadata_handler = MetadataHandler(self.config.destination_path)
        self.data_storage = DataStorage(self.config.destination_path)

    def shutdown_mt5(self):
        """Shuts down MetaTrader5 connection."""
        mt5.shutdown()
        logger.info("MT5 connection closed")

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is available on the MT5 server."""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol '{symbol}' not found on server. Available symbols can be checked in MT5 terminal.")
            return False

        if not symbol_info.visible:
            logger.warning(f"Symbol '{symbol}' exists but is not visible. Attempting to select...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol '{symbol}'")
                return False

        return True

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate fetched data for basic integrity issues."""
        if df is None or df.empty:
            logger.warning(f"No data received for {symbol}")
            return False

        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            logger.warning(f"Data for {symbol} contains NaN values in columns: {nan_cols}")

        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"Data for {symbol} contains {dup_count} duplicate timestamps")

        if len(df) < 1:
            logger.error(f"Data for {symbol} has insufficient rows: {len(df)}")
            return False

        logger.info(f"Data validation passed for {symbol}: {len(df)} bars")
        return True

    def _save_merged_atomically(self, merged_df: pd.DataFrame, symbol: str) -> None:
        """Write merged data to a temp file, then atomically replace the final file."""
        final_path = os.path.join(self.config.destination_path, f"{symbol}.parquet")
        temp_fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=self.config.destination_path)
        try:
            os.close(temp_fd)
            merged_df.to_parquet(temp_path, compression="gzip")
            shutil.move(temp_path, final_path)
            logger.info(f"Atomically saved {len(merged_df)} bars for {symbol} at {final_path}")
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _save_merged_hdf5(self, merged_df: pd.DataFrame, symbol: str) -> None:
        """Save merged data to HDF5 with pre-validation via temp file.

        Writes to a temporary HDF5 file first to verify serialization succeeds.
        Only then writes to the real file, preserving the original on failure.
        """
        hdf5_path = os.path.join(self.config.destination_path, "price_data.h5")
        temp_fd, temp_path = tempfile.mkstemp(suffix=".h5", dir=self.config.destination_path)
        try:
            os.close(temp_fd)
            # Write to temp file to validate serialization before touching the real file
            merged_df.to_hdf(temp_path, key=symbol, mode="w", format="table", data_columns=True)
            # Verify the temp file is readable and has the expected row count
            verify_df = pd.read_hdf(temp_path, key=symbol)
            if len(verify_df) != len(merged_df):
                raise IOError(
                    f"HDF5 verification failed for {symbol}: wrote {len(merged_df)} rows "
                    f"but read back {len(verify_df)}"
                )
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        # Temp write succeeded — commit to real file
        try:
            merged_df.to_hdf(hdf5_path, key=symbol, mode="a", format="table", data_columns=True)
            logger.info(f"Saved {len(merged_df)} bars for {symbol} to HDF5 at {hdf5_path}")
        except Exception:
            logger.error(f"Failed to write {symbol} to {hdf5_path}. Validated temp file preserved at {temp_path}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _load_existing_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load existing data file for a symbol, if it exists."""
        if self.config.write_mode == WriteMode.PARQUET:
            file_path = os.path.join(self.config.destination_path, f"{symbol}.parquet")
            if os.path.exists(file_path):
                return pd.read_parquet(file_path)
        elif self.config.write_mode == WriteMode.HDF5:
            hdf5_path = os.path.join(self.config.destination_path, "price_data.h5")
            if os.path.exists(hdf5_path):
                try:
                    return pd.read_hdf(hdf5_path, key=symbol)
                except KeyError:
                    return None
        return None

    def fetch_and_save_data(self, post_process: Callable[[pd.DataFrame], pd.DataFrame] = None) -> Dict[str, bool]:
        """Fetches and saves market data for all configured symbols.

        Pipeline per symbol:
        1. Fetch new bars from MT5
        2. Basic data validation
        3. Pre-merge plausibility check (bar-count heuristic)
        4. Load existing data and merge (keep latest on duplicates)
        5. Post-merge validation (no data loss, no duplicates)
        6. Atomic write to final location
        7. Update metadata only after successful persist
        """
        if not self.config.validate():
            raise ValueError("Invalid configuration: Missing required fields.")

        os.makedirs(self.config.destination_path, exist_ok=True)
        date_to = datetime.now(tz=standard_tz)
        metadata = self.metadata_handler.read_metadata()
        results = {}

        logger.info("Warming up MT5 for 10 seconds before ingestion")
        print("MT5 warmup: ", end="", flush=True)
        for _ in range(10):
            time.sleep(1)
            print(".", end="", flush=True)
        print()

        for symbol in self.config.symbols:
            sym_metadata = metadata.get(symbol, {})
            try:
                start_date = sym_metadata.get("last_datetime", self.config.start_date)

                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date)

                if not self.validate_symbol(symbol):
                    logger.error(f"Symbol validation failed for {symbol}")
                    results[symbol] = False
                    continue

                logger.info(f"Fetching {symbol} from {start_date} to {date_to}")

                time_diff_minutes = (date_to - start_date).total_seconds() / 60
                if time_diff_minutes < 15:
                    logger.warning(
                        f"Skipping {symbol}: Date range too small ({time_diff_minutes:.1f} minutes). "
                        f"Minimum 15 minutes required."
                    )
                    results[symbol] = True
                    continue

                df = fetch_data_date_range(symbol, self.config.timeframe, start_date, date_to)

                if not self.validate_data(df, symbol):
                    logger.error(f"Data validation failed for {symbol}")
                    results[symbol] = False
                    continue

                # Pre-merge plausibility check
                plausibility_error = _validate_download_plausibility(df, self.config.timeframe, start_date, date_to, symbol)
                if plausibility_error is not None:
                    logger.error(plausibility_error)
                    results[symbol] = False
                    continue

                if post_process:
                    df = post_process(df)

                # Load existing data and merge
                existing_df = self._load_existing_data(symbol)
                if existing_df is not None:
                    merged_df = _merge_dataframes(existing_df, df)
                    logger.info(f"{symbol}: existing={len(existing_df)}, new={len(df)}, merged={len(merged_df)}")
                else:
                    merged_df = df
                    logger.info(f"{symbol}: no existing data, new={len(df)}")

                # Post-merge validation
                merge_error = _validate_post_merge(merged_df, existing_df, df, symbol)
                if merge_error is not None:
                    logger.error(merge_error)
                    results[symbol] = False
                    continue

                # Atomic write to final location
                if self.config.write_mode == WriteMode.HDF5:
                    self._save_merged_hdf5(merged_df, symbol)
                else:
                    self._save_merged_atomically(merged_df, symbol)

                # Metadata written ONLY after successful persist
                # Use index[-2] because the last bar may still be forming (unclosed candle)
                if len(merged_df) >= 2:
                    last_closed_dt = merged_df.index[-2]
                else:
                    last_closed_dt = merged_df.index[-1]

                sym_metadata["last_datetime"] = last_closed_dt
                sym_metadata["last_update"] = datetime.now(tz=standard_tz)
                sym_metadata["bars_count"] = len(merged_df)
                self.metadata_handler.update_symbol_metadata(symbol, sym_metadata)

                logger.info(f"Successfully saved {len(merged_df)} total bars for {symbol}")
                results[symbol] = True

            except Exception as e:
                logger.error(f"Failed to fetch/save data for {symbol}: {e}", exc_info=True)
                results[symbol] = False

        # Summary
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(f"Data fetch completed: {success_count}/{total_count} symbols successful")

        return results
