def resample_ohlcv(df, freq="5min", volume_col_name="tick_volume"):
    """
    Resample OHLCV data from its original timeframe to an arbitrary timeframe.

    Parameters:
        df (pd.DataFrame): Input OHLCV data with a DateTimeIndex.
                           Must contain columns: 'open', 'high', 'low', 'close', 'volume'.
        freq (str): Resampling frequency. Accepts any valid pandas offset alias
                    (e.g., '5min', '15min', '1H', '1D', etc.).

    :param df:
    :param freq:
    :param volume_col_name: name for the volume column to use - default is 'tick_volume'
    :returns
        pd.DataFrame: Resampled OHLCV data.
    """
    resampled = df.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            volume_col_name: "sum",
        }
    )
    return resampled
