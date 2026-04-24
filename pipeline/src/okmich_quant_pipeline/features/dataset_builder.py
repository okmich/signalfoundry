"""
DatasetBuilder — raw OHLCV parquet → feature-enriched + labeled dataset.

Reads per-symbol tuned parameters from SymbolMetastore (via SYMBOL_METASTORE_FILE env var).
Produces one parquet per symbol containing OHLCV + all features + forward log-return label.

Usage:
    import os
    os.environ["SYMBOL_METASTORE_FILE"] = "<path to data>/raw/labelling_metastore.json"

    builder = DatasetBuilder(
        broker="broker name",
        timeframe=5,
        raw_data_root="<path to data>/raw",
        output_dir="<path to data>/processed",
        horizon=12,
        default_window=20,
    )
    builder.build("EURUSD")
    builder.build_all(["EURUSD", "GBPUSD", "XAUUSD", "US100", "BTCUSD"])
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from okmich_quant_features.candle import candle_features
from okmich_quant_features.fractional_diff import fractional_differentiate_series
from okmich_quant_features.microstructure import core_microstructure_features
from okmich_quant_features.momentum import core_momentum_features
from okmich_quant_features.path_structure import core_path_structure_features
from okmich_quant_features.path_structure._zigzag_density import zigzag_density
from okmich_quant_features.temporal import TemporalFeature
from okmich_quant_features.timothymasters.utils.single_features_computer import compute_features
from okmich_quant_features.trend import continuous_ma_trend_labeling
from okmich_quant_features.trend.trend_persistence import trend_persistence_labeling
from okmich_quant_features.trend.z_score_trend import zscore_trend_labeling
from okmich_quant_features.volatility import core_volatility_features, rolling_volatility
from okmich_quant_utils.symbol_metastore import SymbolMetastore

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds a train-ready dataset from raw OHLCV parquet files.

    Steps:
      1. Load & clean raw OHLCV
      2. Compute core feature suites (momentum, volatility, path_structure, microstructure)
      3. Compute candle + temporal features
      4. Compute Timothy Masters single-market features
      5. Compute metastore-tuned special features (frac_diff, zigzag_density,
         zscore_trend, ctl_ma_trend, trend_persistence)
      6. Compute forward log-return label
      7. Drop warmup rows and lookahead tail
      8. Save to output_dir as parquet
    """

    def __init__(self, broker: str, timeframe: int, raw_data_root: str, output_dir: str, horizon: int = 12,
                 default_window: int = 20, vol_normalize_label: bool = True,
                 metastore: Optional[SymbolMetastore] = None):
        """
        Args:
            broker: Broker/server name matching the metastore key (e.g. "TopOneTrader-MT5").
            timeframe: Timeframe in minutes (e.g. 5 for M5).
            raw_data_root: Root directory containing broker subfolders with parquet files.
            output_dir: Directory where processed datasets are saved.
            horizon: Forward return horizon in bars (default 12 → 1 hour on M5).
            default_window: Baseline lookback for standard indicators (default 20).
            vol_normalize_label: If True, divide label by rolling std for vol-normalization.
            metastore: Pre-initialized SymbolMetastore. If None, initializes from env var.
        """
        self.broker = broker
        self.timeframe = timeframe
        self.raw_data_root = Path(raw_data_root)
        self.output_dir = Path(output_dir)
        self.horizon = horizon
        self.default_window = default_window
        self.vol_normalize_label = vol_normalize_label

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if metastore is not None:
            self.metastore = metastore
        else:
            self.metastore = SymbolMetastore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, symbol: str) -> Path:
        """Build dataset for a single symbol. Returns path to saved parquet."""
        logger.info(f"Building dataset: {self.broker} / M{self.timeframe} / {symbol}")

        df = self._load(symbol)
        df = self._clean(df)
        df = self._add_core_features(df)
        df = self._add_candle_temporal(df)
        df = self._add_tm_features(df)
        df = self._add_metastore_features(df, symbol)
        df = self._add_label(df)
        df = self._trim(df)

        out_path = self._output_path(symbol)
        df.to_parquet(out_path)
        logger.info(f"Saved {len(df):,} rows × {len(df.columns)} cols → {out_path}")
        return out_path

    def build_all(self, symbols: List[str]) -> dict:
        """Build datasets for multiple symbols. Returns {symbol: path_or_error}."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.build(symbol)
            except Exception as e:
                logger.error(f"Failed {symbol}: {e}", exc_info=True)
                results[symbol] = e
        return results

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------

    def _load(self, symbol: str) -> pd.DataFrame:
        path = self.raw_data_root / self.broker / str(self.timeframe) / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found: {path}")
        df = pd.read_parquet(path)
        logger.debug(f"Loaded {len(df):,} rows from {path}")
        return df

    # ------------------------------------------------------------------
    # Step 2: Clean
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Normalize index to UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Drop zero-volume bars (closed market / gaps)
        df = df[df["tick_volume"] > 0]

        # Drop weekends
        df = df[df.index.dayofweek < 5]

        # Drop OHLCV rows with any NaN
        ohlcv_cols = ["open", "high", "low", "close", "tick_volume"]
        df = df.dropna(subset=ohlcv_cols)

        logger.debug(f"After clean: {len(df):,} rows")
        return df

    # ------------------------------------------------------------------
    # Step 3: Core feature suites
    # ------------------------------------------------------------------

    def _add_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.default_window
        long_w = w * 2

        # --- Momentum (~40 columns) ---
        mom_df = core_momentum_features(df, window=w, long_window=long_w)

        # --- Volatility (~20 columns, + realized vol if DatetimeIndex) ---
        vol_df = core_volatility_features(df, window=w, long_window=long_w, atr_period=max(w - 6, 8),
                                          freq_minutes=self.timeframe, volume_col="tick_volume")

        # --- Path structure (~18 columns) ---
        path_df = core_path_structure_features(df)

        # --- Microstructure (~50 columns, OHLCV only — no spread data) ---
        ms_df = core_microstructure_features(df, window=w, volume_col="tick_volume")

        def _prefix(feat_df):
            return feat_df.rename(columns=lambda c: f"feat_{c}")

        df = pd.concat([df, _prefix(mom_df), _prefix(vol_df), _prefix(path_df), _prefix(ms_df)], axis=1)
        logger.debug(f"Core features: {len([c for c in df.columns if c.startswith('feat_')])} columns")
        return df

    # ------------------------------------------------------------------
    # Step 4: Candle + temporal
    # ------------------------------------------------------------------

    def _add_candle_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        # Candle features (returns columns already prefixed with candle_)
        candle_df = candle_features(df)
        candle_cols = [c for c in candle_df.columns if c not in df.columns]

        # Temporal features
        tf = TemporalFeature(df)
        hour_sin, hour_cos = tf.hour_of_day_cyclic()
        dow_sin, dow_cos = tf.day_of_week_cyclic()
        temporal_df = pd.DataFrame({
            "temporal_hour_sin": hour_sin,
            "temporal_hour_cos": hour_cos,
            "temporal_dow_sin": dow_sin,
            "temporal_dow_cos": dow_cos,
        }, index=df.index)

        df = pd.concat([df, candle_df[candle_cols], temporal_df], axis=1)
        return df

    # ------------------------------------------------------------------
    # Step 5: Timothy Masters features
    # ------------------------------------------------------------------

    def _add_tm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.default_window
        long_w = w * 2
        half_w = max(w // 2, 5)
        params = {
            # Momentum
            "rsi":              {"period": w},
            "detrended_rsi":    {"short_period": half_w, "long_period": w, "reg_len": long_w},
            "stochastic":       {"period": w},
            "stoch_rsi":        {"rsi_period": w, "stoch_period": w},
            "ma_difference":    {"short_period": half_w, "long_period": long_w},
            "macd":             {"short_period": w, "long_period": long_w, "signal_period": max(w // 4, 5)},
            "ppo":              {"short_period": w, "long_period": long_w},
            "price_change_osc": {"short_period": half_w},
            "close_minus_ma":   {"period": w, "atr_period": w * 3},
            # Trend
            "linear_trend":     {"period": w, "atr_period": w * 3},
            "quadratic_trend":  {"period": w, "atr_period": w * 3},
            "cubic_trend":      {"period": w, "atr_period": w * 3},
            "linear_deviation":    {"period": w},
            "quadratic_deviation": {"period": w},
            "cubic_deviation":     {"period": w},
            "adx":              {"period": w},
            "aroon_up":         {"period": w},
            "aroon_down":       {"period": w},
            "aroon_diff":       {"period": w},
            # Variance
            "price_variance_ratio":  {"short_period": half_w},
            "change_variance_ratio": {"short_period": half_w},
            # Volume
            "intraday_intensity": {"period": w},
            "money_flow":         {"period": w},
            "price_volume_fit":   {"period": w},
            "vwma_ratio":         {"period": w},
            "normalized_obv":     {"period": w},
            "delta_obv":          {"period": w, "delta_period": half_w},
            "normalized_pvi":     {"period": w},
            "normalized_nvi":     {"period": w},
            "volume_momentum":    {"short_period": half_w},
        }
        tm_df = compute_features(df, groups="all", params=params, prefix="tm_")
        tm_cols = [c for c in tm_df.columns if c not in df.columns]
        df = pd.concat([df, tm_df[tm_cols]], axis=1)
        return df

    # ------------------------------------------------------------------
    # Step 6: Metastore-tuned special features
    # ------------------------------------------------------------------

    def _add_metastore_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        close = df["close"]

        # -- Fractional differentiation --
        frac_params = self.metastore.get_property_value(
            self.broker, self.timeframe, symbol, "frac_diff"
        )
        if frac_params:
            d = frac_params["optimal_d"]
            win = frac_params.get("opt_window_size", 100)
            frac_series, _ = fractional_differentiate_series(close, d=d, window_size=win)
            df["feat_frac_diff"] = frac_series
            logger.debug(f"{symbol}: frac_diff d={d}, window={win}")

        # -- ZigZag density --
        zz_params = self.metastore.get_property_value(
            self.broker, self.timeframe, symbol, "zigzag_density_params"
        )
        if zz_params:
            threshold = zz_params["threshold"]
            window = zz_params.get("window")
            align = zz_params.get("align", "causal")
            zz_series, _ = zigzag_density(close, threshold=threshold, window=window, align=align)
            df["feat_zigzag_density"] = zz_series
            logger.debug(f"{symbol}: zigzag_density threshold={threshold}, window={window}")

        # -- Z-score trend --
        zt_params = self.metastore.get_property_value(
            self.broker, self.timeframe, symbol, "zscore_trend_params"
        )
        if zt_params:
            window = zt_params["window"]
            z_threshold = zt_params.get("z_threshold", 1.0)
            zscore_series, _ = zscore_trend_labeling(close, window=window, z_threshold=z_threshold)
            df["feat_zscore_trend"] = zscore_series
            logger.debug(f"{symbol}: zscore_trend window={window}")

        # -- CTL MA trend --
        ctl_params = self.metastore.get_property_value(
            self.broker, self.timeframe, symbol, "ctl_ma_params"
        )
        if ctl_params:
            omega = ctl_params["omega"]
            trend_window = ctl_params.get("trend_window", self.default_window)
            smooth_window = ctl_params.get("smooth_window", 3)
            ctl_series = continuous_ma_trend_labeling(
                close, omega=omega, trend_window=trend_window, smooth_window=smooth_window
            )
            df["feat_ctl_ma_trend"] = ctl_series
            logger.debug(f"{symbol}: ctl_ma omega={omega}, trend_window={trend_window}")

        # -- Trend persistence --
        tp_params = self.metastore.get_property_value(
            self.broker, self.timeframe, symbol, "trend_persistence_params"
        )
        if tp_params:
            window = tp_params.get("window", self.default_window)
            smooth = tp_params.get("smooth", 3)
            zscore_norm = tp_params.get("zscore_norm", False)
            tp_series = trend_persistence_labeling(
                close, window=window, smooth=smooth, zscore_norm=zscore_norm
            )
            df["feat_trend_persistence"] = tp_series
            logger.debug(f"{symbol}: trend_persistence window={window}, smooth={smooth}")

        return df.copy()

    # ------------------------------------------------------------------
    # Step 7: Label
    # ------------------------------------------------------------------

    def _add_label(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        future_close = close.shift(-self.horizon)
        log_ret = np.log(future_close / close)

        if self.vol_normalize_label:
            vol = rolling_volatility(close, window=max(self.default_window * 5, 100))
            vol = vol.replace(0, np.nan).ffill()
            df["label"] = log_ret / vol
        else:
            df["label"] = log_ret

        return df

    # ------------------------------------------------------------------
    # Step 8: Trim warmup + lookahead tail
    # ------------------------------------------------------------------

    # Maximum fraction of NaN allowed in a feature column before it is dropped entirely.
    MAX_COL_NAN_RATE = 0.20

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop tail rows where label is NaN (no future close)
        df = df.iloc[: len(df) - self.horizon]

        feat_cols = [c for c in df.columns if c.startswith(("feat_", "tm_", "candle_", "temporal_"))]
        check_cols = feat_cols + ["label"]

        # Drop feature columns that are too sparse to be useful
        nan_rates = df[check_cols].isna().mean()
        sparse_cols = nan_rates[nan_rates > self.MAX_COL_NAN_RATE].index.tolist()
        sparse_feat_cols = [c for c in sparse_cols if c != "label"]
        if sparse_feat_cols:
            logger.warning(
                f"Dropping {len(sparse_feat_cols)} sparse feature column(s) "
                f"(>{self.MAX_COL_NAN_RATE:.0%} NaN): {sparse_feat_cols}"
            )
            df = df.drop(columns=sparse_feat_cols)
            check_cols = [c for c in check_cols if c not in sparse_feat_cols]

        # Drop rows where any remaining feature or label is NaN
        before = len(df)
        df = df.dropna(subset=check_cols)
        dropped = before - len(df)
        logger.debug(f"Trimmed {dropped:,} warmup/NaN rows — {len(df):,} rows remain")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _output_path(self, symbol: str) -> Path:
        broker_dir = self.output_dir / self.broker / str(self.timeframe)
        broker_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"H{self.horizon}_W{self.default_window}"
        return broker_dir / f"{symbol}_{suffix}.parquet"