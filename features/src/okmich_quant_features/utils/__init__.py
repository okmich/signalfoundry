import numpy as np
import pandas as pd

from .binning import partition, validate_partition
from .lag_features import create_lag_features
from .resample import resample_ohlcv
from .transform import (
    # Stateless functions (for EDA)
    logit_transform,
    log_transform,
    standardize,
    # Stateful transformers (for ML pipelines)
    YeoJohnsonTransformer,
    BoxCoxTransformer,
    LogitTransformer,
    LogTransformer,
    get_transformer,
    apply_transformation_recommendations,
)
from .rolling_transforms import rolling_zscore, rolling_percentile_rank, rolling_volatility_scale, rolling_slope, \
    rolling_persistence, tanh_compress, sigmoid_compress

from .transform_pipeline import (
    # Config export/load
    export_transformation_config,
    load_transformation_config,
    # Pipeline builder
    build_pipeline_from_config,
    # Artifact management
    save_pipeline_artifacts,
    load_pipeline_artifacts,
)


# Backward compatibility
def logit_transformation(series):
    """Deprecated: Use logit_transform instead."""
    return logit_transform(series)


def ensure_numpy_types(open_prices, high_prices, low_prices, close_prices):
    # Check if inputs are pandas Series
    is_pandas = isinstance(open_prices, pd.Series)

    index = None
    # Store index if pandas Series
    if is_pandas:
        index = open_prices.index
        # Convert to numpy arrays for processing
        open_prices = open_prices.values
        high_prices = high_prices.values
        low_prices = low_prices.values
        close_prices = close_prices.values

    # Ensure numpy arrays
    open_prices = np.asarray(open_prices, dtype=np.float64)
    high_prices = np.asarray(high_prices, dtype=np.float64)
    low_prices = np.asarray(low_prices, dtype=np.float64)
    close_prices = np.asarray(close_prices, dtype=np.float64)

    return index, open_prices, high_prices, low_prices, close_prices


def ensure_numpy_types_for_series(prices):
    # Check if inputs are pandas Series
    is_pandas = isinstance(prices, pd.Series)

    index = None
    # Store index if pandas Series
    if is_pandas:
        index = prices.index
        prices = prices.values

    # Ensure numpy arrays
    prices = np.asarray(prices, dtype=np.float64)
    return index, prices
