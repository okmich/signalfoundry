import logging
import os
from datetime import datetime
from typing import Dict, Any, Callable

import MetaTrader5 as mt5
import pandas as pd
import pytz

from okmich_quant_mt5.functions import fetch_tick_data_date_range
from .mt5_data_fetcher import MT5MarketDataFetcher

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
standard_tz = pytz.timezone("Etc/UTC")
metadata_filename = "_metadata.json"


class MT5MarketTickDataFetcher(MT5MarketDataFetcher):
    """Main class to orchestrate MT5 market tick data fetching and storage."""

    def __init__(self, conf: Dict[str, Any]):
        super().__init__(conf)

    def fetch_and_save_data(
        self, post_process: Callable[[pd.DataFrame], pd.DataFrame] = None
    ) -> None:
        """Fetches and saves market data for all configured symbols."""
        if not self.config.validate():
            raise ValueError("Invalid configuration: Missing required fields.")

        os.makedirs(self.config.destination_path, exist_ok=True)
        self.initialize_mt5()
        date_to = datetime.now(tz=standard_tz)
        metadata = self.metadata_handler.read_metadata()

        for symbol in self.config.symbols:
            sym_metadata = metadata.get(symbol, {})
            try:
                start_date = sym_metadata.get("last_datetime", self.config.start_date)
                df = fetch_tick_data_date_range(symbol, start_date, date_to)
                if df is None:
                    continue

                if post_process:
                    df = post_process(df)

                if self.config.write_mode == "hdf5":
                    self.data_storage.save_to_hdf5(df, symbol)
                else:
                    self.data_storage.save_to_parquet(df, symbol)

                sym_metadata["last_datetime"] = date_to
                metadata[symbol] = sym_metadata
            except Exception as e:
                logging.error(f"Failed to fetch data for {symbol}: {e}")

        self.metadata_handler.save_metadata(metadata)
        mt5.shutdown()
