import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd


class FeatureStore:
    """
    Feature caching system using Parquet storage.
    Caches features by symbol, timeframe, and version to avoid recomputation.
    Examples
    --------
    >>> store = FeatureStore(cache_dir='cache/features')
    >>>
    >>> # Check cache
    >>> if store.exists('US500.r', '5min', 'v3'):
    ...     features = store.load('US500.r', '5min', 'v3')
    ... else:
    ...     features = compute_features(df)
    ...     store.save(features, 'US500.r', '5min', 'v3')
    """

    def __init__(self, cache_dir: str = "cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, symbol: str, timeframe: str, version: str) -> str:
        # Clean inputs
        symbol_clean = symbol.replace("/", "_").replace(".", "_")
        timeframe_clean = timeframe.replace("/", "_").replace(".", "_")
        version_clean = version.replace("/", "_").replace(".", "_")

        return f"{symbol_clean}_{timeframe_clean}_{version_clean}"

    def _get_cache_path(self, symbol: str, timeframe: str, version: str) -> Path:
        """Get path to cache file."""
        cache_key = self._get_cache_key(symbol, timeframe, version)
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_metadata_path(self, symbol: str, timeframe: str, version: str) -> Path:
        """Get path to metadata file."""
        cache_key = self._get_cache_key(symbol, timeframe, version)
        return self.cache_dir / f"{cache_key}_metadata.json"

    def exists(self, symbol: str, timeframe: str, version: str) -> bool:
        cache_path = self._get_cache_path(symbol, timeframe, version)
        return cache_path.exists()

    def load(self, symbol: str, timeframe: str, version: str) -> pd.DataFrame:
        cache_path = self._get_cache_path(symbol, timeframe, version)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found for {symbol} {timeframe} {version}. "
                f"Expected at: {cache_path}"
            )
        return pd.read_parquet(cache_path)

    def save(self, features: pd.DataFrame, symbol: str, timeframe: str, version: str, metadata: Optional[Dict[str, Any]] = None):
        cache_path = self._get_cache_path(symbol, timeframe, version)
        metadata_path = self._get_metadata_path(symbol, timeframe, version)

        # Save features
        features.to_parquet(cache_path, index=True)

        # Save metadata
        metadata_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "n_features": len(features.columns),
            "n_samples": len(features),
            "feature_names": list(features.columns),
            "cached_at": datetime.now().isoformat(),
            "data_hash": self._hash_dataframe(features),
        }

        if metadata:
            metadata_dict.update(metadata)

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def get_metadata(self, symbol: str, timeframe: str, version: str) -> Dict[str, Any]:
        metadata_path = self._get_metadata_path(symbol, timeframe, version)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def invalidate(self, symbol: str, timeframe: str, version: str):
        cache_path = self._get_cache_path(symbol, timeframe, version)
        metadata_path = self._get_metadata_path(symbol, timeframe, version)

        if cache_path.exists():
            cache_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

    def list_cached(self) -> pd.DataFrame:
        cached = []

        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                cached.append(metadata)

        if not cached:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "timeframe",
                    "version",
                    "n_features",
                    "n_samples",
                    "cached_at",
                ]
            )

        return pd.DataFrame(cached)

    def clear_all(self):
        """Clear all cached features."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        for file in self.cache_dir.glob("*_metadata.json"):
            file.unlink()

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        # Content-aware fingerprint: shape + columns + index boundary values
        # + sampled numeric statistics to detect different data with same schema.
        import numpy as np
        parts = [str(df.shape), str(list(df.columns))]
        if len(df) > 0:
            parts.append(str(df.index[0]))
            parts.append(str(df.index[-1]))
            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                # Use column sums and means of first/last 5 rows as a cheap fingerprint
                sample = pd.concat([numeric.head(5), numeric.tail(5)])
                parts.append(str(round(float(sample.values.sum()), 6)))
                parts.append(str(round(float(sample.values.mean()), 6)))
        data_str = "|".join(parts)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def get_or_compute(self, symbol: str, timeframe: str, version: str, compute_fn: Callable[[], pd.DataFrame],
                       force_recompute: bool = False, **compute_kwargs) -> pd.DataFrame:
        """
        Get features from cache or compute if not exists.
        Convenience method that handles cache checking and computation.

        Examples
        --------
        >>> def compute_features(df, lookback=20):
        ...     # Feature engineering logic
        ...     return features
        >>>
        >>> features = store.get_or_compute(
        ...     'US500.r', '5min', 'v3',
        ...     compute_fn=lambda: compute_features(df, lookback=20)
        ... )
        """
        if not force_recompute and self.exists(symbol, timeframe, version):
            print(f"Loading features from cache: {symbol} {timeframe} {version}")
            return self.load(symbol, timeframe, version)

        print(f"Computing features: {symbol} {timeframe} {version}")
        features = compute_fn(**compute_kwargs)

        print(f"Saving to cache: {symbol} {timeframe} {version}")
        self.save(features, symbol, timeframe, version)

        return features


def get_default_feature_store() -> FeatureStore:
    return FeatureStore(cache_dir="cache/features")
