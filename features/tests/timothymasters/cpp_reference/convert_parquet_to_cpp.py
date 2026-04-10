"""
Convert parquet market data to C++ SINGLE.exe format.

Reads US500 data and converts to: YYYYMMDD Open High Low Close Volume
"""

import pandas as pd
from pathlib import Path


def convert_parquet_to_market_file(
    parquet_path: str,
    output_filename: str,
    n_records: int = 1000,
    skip_records: int = 0
):
    """
    Convert parquet data to C++ market file format.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file
    output_filename : str
        Output filename (will be created in test_data/)
    n_records : int
        Number of records to extract (default 1000)
    skip_records : int
        Number of records to skip from start (default 0)
    """
    print(f"\nReading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Total records available: {len(df)}")

    # Take subset
    df = df.iloc[skip_records:skip_records + n_records]
    print(f"Using records {skip_records} to {skip_records + len(df)}")

    # Prepare output directory
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / output_filename

    print(f"\nConverting to C++ format...")

    with open(filepath, 'w') as f:
        for idx, row in df.iterrows():
            # Convert datetime to YYYYMMDD format
            if isinstance(idx, pd.Timestamp):
                date_str = idx.strftime('%Y%m%d')
            else:
                # If index is not datetime, use sequential dates
                date_str = f"{20230522 + len(df.index[:idx.name])}"

            # Format: YYYYMMDD Open High Low Close Volume
            f.write(
                f"{date_str} "
                f"{row['open']:.6f} "
                f"{row['high']:.6f} "
                f"{row['low']:.6f} "
                f"{row['close']:.6f} "
                f"{int(row['tick_volume'])}\n"
            )

    print(f"\nCreated {filepath}")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    return filepath


def create_real_data_files():
    """Create multiple real data files with different characteristics."""

    parquet_path = "D:/data_dump/market_data/raw/FXPIG-Server/5/US500.r.parquet"

    # Read full data to find interesting periods
    df = pd.read_parquet(parquet_path)
    print(f"Total data available: {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    datasets = {
        'us500_first_1000.txt': {
            'skip': 0,
            'n_records': 1000,
            'description': 'First 1000 bars'
        },
        'us500_middle_500.txt': {
            'skip': len(df) // 2,
            'n_records': 500,
            'description': 'Middle 500 bars'
        },
        'us500_recent_500.txt': {
            'skip': len(df) - 500,
            'n_records': 500,
            'description': 'Most recent 500 bars'
        },
    }

    print("\n" + "="*60)
    print("Creating real market data files for C++ reference testing")
    print("="*60)

    for filename, params in datasets.items():
        print(f"\n--- {params['description']} ---")
        convert_parquet_to_market_file(
            parquet_path,
            filename,
            n_records=params['n_records'],
            skip_records=params['skip']
        )

    print("\n" + "="*60)
    print("All real data files created successfully!")
    print("="*60)


if __name__ == "__main__":
    create_real_data_files()
