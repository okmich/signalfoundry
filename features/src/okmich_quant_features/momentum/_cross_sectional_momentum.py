import pandas as pd
import talib


def imom(df: pd.DataFrame, market_df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Idiosyncratic Momentum — asset vs market ROC difference.

    Description:
        Uses TA-Lib ROC to measure momentum differential between
        asset and benchmark (e.g., SPX, BTC, DXY).

    High-Impact Use Case:
        - Long-short construction
        - Alpha separation from market beta

    High-Impact Asset Classes:
        - Equities (sector rotation)
        - FX crosses (relative currency strength)
    """
    asset_roc = talib.ROC(df["close"], timeperiod=n)
    market_roc = talib.ROC(market_df["close"], timeperiod=n)
    return pd.Series(asset_roc - market_roc, index=df.index)


def csz_mom(df: pd.DataFrame, universe_dfs: list, n: int = 14) -> pd.Series:
    """
    Cross-Sectional Z-Momentum.

    Description:
        Computes Z-score of asset’s ROC relative to universe of peers.

    High-Impact Use Case:
        - Momentum ranking and portfolio sorting
        - Cross-sectional factor construction

    High-Impact Asset Classes:
        - Equities, FX baskets, crypto portfolios
    """
    rocs = [talib.ROC(u["close"], timeperiod=n) for u in universe_dfs]
    stacked = pd.concat(rocs, axis=1)
    mean_roc = stacked.mean(axis=1)
    std_roc = stacked.std(axis=1)
    asset_roc = talib.ROC(df["close"], timeperiod=n)
    return (asset_roc - mean_roc) / std_roc


def imom(df: pd.DataFrame, market_df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Idiosyncratic Momentum (IMOM): asset's standalone strength vs market beta.
    Formula: ROC(asset, n) - ROC(market, n)

    Parameters
    ----------
    df : DataFrame
        Asset data with 'close'.
    market_df : DataFrame
        Market index or benchmark with 'close'.
    period : int
        Lookback period for ROC.

    Returns
    -------
    pd.Series
        Difference between asset and market ROC.

    High-Impact Use Case
    --------------------
    - Relative momentum models: highlight stocks outperforming the market.
    - Asset classes: equities, forex crosses, crypto vs BTC.
    """
    asset_roc = talib.ROC(df["close"], timeperiod=period)
    market_roc = talib.ROC(market_df["close"], timeperiod=period)
    return asset_roc - market_roc


def idiomatic_intraday_mom(
        df: pd.DataFrame, sector_df: pd.DataFrame, window: int = 30
) -> pd.Series:
    """
    Sector-relative Intraday Momentum.
    Measures intraday strength relative to sector index.

    Parameters
    ----------
    df : DataFrame
        Asset intraday OHLC data.
    sector_df : DataFrame
        Sector or index-level OHLC data.
    window : int
        Rolling lookback for smoothing.

    Returns
    -------
    pd.Series
        Intraday relative momentum.

    High-Impact Use Case
    --------------------
    - Intraday equity or FX basket analysis.
    - Detects when an asset diverges from its sector or correlated peers.
    """
    asset_intraday_roc = df["close"].pct_change().rolling(window).mean()
    sector_intraday_roc = sector_df["close"].pct_change().rolling(window).mean()
    return asset_intraday_roc - sector_intraday_roc


def csz_mom(df: pd.DataFrame, universe_dfs: dict, period: int = 20) -> pd.Series:
    """
    Cross-sectional Z-Momentum.
    Computes Z-score of ROC relative to peer universe.

    Parameters
    ----------
    df : DataFrame
        Target asset data.
    universe_dfs : dict
        Dictionary of {symbol: DataFrame} peers.
    period : int
        ROC lookback.

    Returns
    -------
    pd.Series
        Z-scored momentum value.

    High-Impact Use Case
    --------------------
    - Equity or crypto cross-sectional ranking.
    - Helps normalize performance among peers (e.g., top decile momentum).
    """
    asset_roc = talib.ROC(df["close"], timeperiod=period)
    peer_rocs = [
        talib.ROC(udf["close"], timeperiod=period) for udf in universe_dfs.values()
    ]
    peer_concat = pd.concat(peer_rocs, axis=1)
    z_mom = (asset_roc - peer_concat.mean(axis=1)) / peer_concat.std(axis=1)
    return z_mom


def peer_rel_z_mom(df: pd.DataFrame, universe_dfs: dict, period: int = 20) -> pd.Series:
    """
    Peer-relative Z-Momentum.
    Similar to csz_mom but normalizes by the peer’s mean absolute deviation.

    Parameters
    ----------
    df : DataFrame
        Target asset data.
    universe_dfs : dict
        Dictionary of {symbol: DataFrame} peers.
    period : int
        ROC lookback.

    Returns
    -------
    pd.Series
        Peer-normalized momentum score.

    High-Impact Use Case
    --------------------
    - Multi-asset strategies.
    - Detects abnormal outliers within correlated clusters.
    """
    asset_roc = talib.ROC(df["close"], timeperiod=period)
    peer_rocs = [
        talib.ROC(udf["close"], timeperiod=period) for udf in universe_dfs.values()
    ]
    peer_concat = pd.concat(peer_rocs, axis=1)
    mad = peer_concat.mad(axis=1)
    z_rel = (asset_roc - peer_concat.mean(axis=1)) / mad
    return z_rel
