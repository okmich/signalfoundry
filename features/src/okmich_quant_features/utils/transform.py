"""
Feature transformations for machine learning pipelines.

Provides both:
1. Stateless functions for EDA/exploration
2. Stateful transformer classes for training pipelines (fit/transform pattern)

The stateful transformers can be saved with joblib and loaded during inference
to ensure consistent transformations between training and production.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from typing import Union, Tuple, Optional


def logit_transform(series: Union[np.ndarray, pd.Series], epsilon: float = 1e-9) -> Union[np.ndarray, pd.Series]:
    clipped = np.clip(series, epsilon, 1 - epsilon)
    result = np.log(clipped / (1 - clipped))

    # Preserve pandas metadata
    if isinstance(series, pd.Series):
        result = pd.Series(result, index=series.index, name=series.name)

    return result


def log_transform(series: Union[np.ndarray, pd.Series], offset: Optional[float] = None) -> Union[np.ndarray, pd.Series]:
    values = np.asarray(series)
    if offset is None:
        min_val = np.min(values)
        offset = 0 if min_val > 0 else abs(min_val) + 1

    result = np.log(values + offset)
    if isinstance(series, pd.Series):
        result = pd.Series(result, index=series.index, name=series.name)
    return result


def standardize(series: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    values = np.asarray(series)
    mean = np.mean(values)
    std = np.std(values)

    result = (values - mean) / std if std > 0 else np.zeros_like(values)
    if isinstance(series, pd.Series):
        result = pd.Series(result, index=series.index, name=series.name)

    return result


# ============================================================================
# Stateful Transformer Classes (for ML pipelines)
# ============================================================================

class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """
    Yeo-Johnson power transformation (works with positive and negative values).

    Fits transformation parameters on training data, applies same transformation
    to new data during inference.
    """

    def __init__(self, standardize: bool = False):
        self.standardize = standardize
        self._transformer = PowerTransformer(
            method='yeo-johnson',
            standardize=standardize
        )
        self._is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.Series], y=None) -> 'YeoJohnsonTransformer':
        values = self._to_2d(X)
        self._transformer.fit(values)
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")

        is_pandas = isinstance(X, pd.Series)
        values = self._to_2d(X)
        result = self._transformer.transform(values)

        # Return 2D for sklearn compatibility, or Series for pandas
        if is_pandas:
            result = pd.Series(result.ravel(), index=X.index, name=X.name)
        # Keep 2D shape for sklearn ColumnTransformer compatibility
        return result

    def fit_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before inverse_transform")

        is_pandas = isinstance(X, pd.Series)
        values = self._to_2d(X)
        result = self._transformer.inverse_transform(values)

        # Return 2D for sklearn compatibility, or Series for pandas
        if is_pandas:
            result = pd.Series(result.ravel(), index=X.index, name=X.name)
        # Keep 2D shape for sklearn compatibility
        return result

    @property
    def lambda_(self) -> float:
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted yet")
        return self._transformer.lambdas_[0]

    @staticmethod
    def _to_2d(X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        if isinstance(X, pd.Series):
            values = X.values
        else:
            values = np.asarray(X)
        return values.reshape(-1, 1)


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Box-Cox power transformation (only for strictly positive values).

    For data with zeros/negatives, use YeoJohnsonTransformer instead.
    """

    def __init__(self, standardize: bool = False):
        """
        Parameters
        ----------
        standardize : bool, default False
            If True, also standardize after transformation
        """
        self.standardize = standardize
        self._transformer = PowerTransformer(
            method='box-cox',
            standardize=standardize
        )
        self._is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.Series], y=None) -> 'BoxCoxTransformer':
        """Fit transformer to data."""
        values = self._to_2d(X)

        if np.any(values <= 0):
            raise ValueError(
                "Box-Cox requires strictly positive values. "
                "Use YeoJohnsonTransformer for data with zeros/negatives."
            )

        self._transformer.fit(values)
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Transform data using fitted parameters."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")

        is_pandas = isinstance(X, pd.Series)
        values = self._to_2d(X)

        result = self._transformer.transform(values)

        # Return 2D for sklearn compatibility, or Series for pandas
        if is_pandas:
            result = pd.Series(result.ravel(), index=X.index, name=X.name)
        # Keep 2D shape for sklearn ColumnTransformer compatibility
        return result

    def fit_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Inverse transformation."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before inverse_transform")

        is_pandas = isinstance(X, pd.Series)
        values = self._to_2d(X)

        result = self._transformer.inverse_transform(values)

        # Return 2D for sklearn compatibility, or Series for pandas
        if is_pandas:
            result = pd.Series(result.ravel(), index=X.index, name=X.name)
        # Keep 2D shape for sklearn compatibility
        return result

    @property
    def lambda_(self) -> float:
        """Fitted lambda parameter."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted yet")
        return self._transformer.lambdas_[0]

    @staticmethod
    def _to_2d(X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert to 2D array for sklearn."""
        if isinstance(X, pd.Series):
            values = X.values
        else:
            values = np.asarray(X)
        return values.reshape(-1, 1)


class LogitTransformer(BaseEstimator, TransformerMixin):
    """
    Logit transformation with fitted bounds.

    Fits clipping bounds on training data, applies same bounds during inference.
    Useful for bounded [0,1] features.
    """

    def __init__(self, epsilon: float = 1e-9):
        """
        Parameters
        ----------
        epsilon : float, default 1e-9
            Small value for clipping to avoid log(0)
        """
        self.epsilon = epsilon
        self._is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.Series], y=None) -> 'LogitTransformer':
        """Fit (no-op for logit, but maintains interface)."""
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Transform data."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")

        is_pandas = isinstance(X, pd.Series)
        result = logit_transform(X, epsilon=self.epsilon)

        # Ensure 2D for sklearn ColumnTransformer compatibility
        if not is_pandas and isinstance(result, np.ndarray) and result.ndim == 1:
            result = result.reshape(-1, 1)

        return result

    def fit_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log transformation with fitted offset.

    Computes offset on training data, applies same offset during inference.
    """

    def __init__(self):
        self.offset_ = None
        self._is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.Series], y=None) -> 'LogTransformer':
        """Fit transformer - compute offset from training data."""
        values = np.asarray(X)
        min_val = np.min(values)
        self.offset_ = 0 if min_val > 0 else abs(min_val) + 1
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Transform using fitted offset."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")

        is_pandas = isinstance(X, pd.Series)
        values = np.asarray(X)

        result = np.log(values + self.offset_)

        if is_pandas:
            result = pd.Series(result, index=X.index, name=X.name)
        elif result.ndim == 1:
            # Ensure 2D for sklearn ColumnTransformer compatibility
            result = result.reshape(-1, 1)

        return result

    def fit_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.Series], y=None) -> Union[np.ndarray, pd.Series]:
        """Inverse transformation."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before inverse_transform")

        is_pandas = isinstance(X, pd.Series)
        values = np.asarray(X)

        result = np.exp(values) - self.offset_

        if is_pandas:
            result = pd.Series(result, index=X.index, name=X.name)
        elif isinstance(result, np.ndarray) and result.ndim == 1:
            # Ensure 2D for sklearn compatibility
            result = result.reshape(-1, 1)

        return result


# ============================================================================
# Helper Functions
# ============================================================================

def get_transformer(transformation_type: str, **kwargs):
    transformers = {
        'yeo-johnson': YeoJohnsonTransformer,
        'box-cox': BoxCoxTransformer,
        'logit': LogitTransformer,
        'log': LogTransformer,
    }

    if transformation_type not in transformers:
        raise ValueError(
            f"Unknown transformation: {transformation_type}. "
            f"Valid options: {list(transformers.keys())}"
        )

    return transformers[transformation_type](**kwargs)


def apply_transformation_recommendations(df: pd.DataFrame, transformation_df: pd.DataFrame, replace_original: bool = True) -> pd.DataFrame:
    """
    Apply transformation recommendations from FeatureEDA.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
    transformation_df : pd.DataFrame
        Transformation recommendations from FeatureEDA.recommend_transformations()
    replace_original : bool, default True
        If True, replaces original columns with transformed values (in-place).
        If False, creates new columns with suffix (e.g., 'feature_yeojohnson').

    Returns
    -------
    df_transformed : pd.DataFrame
        DataFrame with transformed features (replaced or added based on replace_original)

    Examples
    --------
    # Replace original columns (default)
    >>> df_transformed = apply_transformation_recommendations(df, recommendations)

    # Keep original + add new transformed columns
    >>> df_transformed = apply_transformation_recommendations(df, recommendations, replace_original=False)
    """
    df_result = df.copy()
    fitted_transformers = {}

    for idx, row in transformation_df.iterrows():
        feature = row['feature']
        trans_str = row['transformations']

        if pd.isna(trans_str) or trans_str == 'none' or 'standardize' in trans_str:
            continue

        # Extract primary transformation (first in comma-separated list)
        primary_trans = trans_str.split(',')[0].strip()

        # Map to transformer type
        if primary_trans in ['yeo-johnson', 'box-cox', 'logit', 'log']:
            try:
                transformer = get_transformer(primary_trans)
                transformed = transformer.fit_transform(df[feature])

                if replace_original:
                    # Replace original column in-place
                    df_result[feature] = transformed
                else:
                    # Add new column with transformation suffix
                    new_name = f"{feature}_{primary_trans.replace('-', '')}"
                    df_result[new_name] = transformed

            except Exception as e:
                print(f"Warning: Failed to transform {feature} with {primary_trans}: {e}")
    return df_result
