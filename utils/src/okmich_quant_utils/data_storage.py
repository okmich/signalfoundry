import logging
import os
import shutil
import tempfile

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataStorage:
    """Handles saving data to Parquet or HDF5 files."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def list_files(self):
        return os.listdir(self.folder_path)

    def read_parquet(self, symbol: str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.folder_path, f"{symbol}.parquet"))

    def read_hdf5(self, file_name: str, symbol: str) -> pd.DataFrame:
        try:
            df = pd.read_hdf(os.path.join(self.folder_path, file_name), key=symbol)
            return df
        except KeyError:
            raise ValueError(f"Key '{symbol}' not found in the HDF5 file.")
        except Exception as e:
            raise Exception(f"Error reading HDF5 file: {e}")

    def save_to_parquet(self, df: pd.DataFrame, symbol: str):
        """Saves DataFrame to a Parquet file, merging with existing data if present."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            temp_path = temp_file.name
            df.to_parquet(temp_path, compression="gzip")

        file_path = os.path.join(self.folder_path, f"{symbol}.parquet")
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            merged_df = (
                pd.concat([existing_df, df]).sort_index().drop_duplicates(keep="last")
            )
            merged_df.to_parquet(file_path, compression="gzip")
            logging.info(f"Merged and updated Parquet file for {symbol} at {file_path}")
        else:
            shutil.copy(temp_path, file_path)
            logging.info(f"Saved new Parquet file for {symbol} at {file_path}")
        os.unlink(temp_path)

    def save_to_hdf5(self, df: pd.DataFrame, symbol: str):
        """Saves DataFrame to an HDF5 file, merging with existing data if present."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            temp_path = temp_file.name
            df.to_parquet(temp_path)

        hdf5_path = os.path.join(self.folder_path, "price_data.h5")
        temp_df = pd.read_parquet(temp_path)
        try:
            existing_df = pd.read_hdf(hdf5_path, key=symbol)
            merged_df = (
                pd.concat([existing_df, temp_df])
                .sort_index()
                .drop_duplicates(keep="last")
            )
            merged_df.to_hdf(
                hdf5_path, key=symbol, mode="a", format="table", data_columns=True
            )
            logging.info(f"Updated HDF5 key '{symbol}' in {hdf5_path}")
        except (KeyError, FileNotFoundError):
            temp_df.to_hdf(
                hdf5_path, key=symbol, mode="a", format="table", data_columns=True
            )
            logging.info(f"Created new HDF5 key '{symbol}' in {hdf5_path}")
        os.unlink(temp_path)
