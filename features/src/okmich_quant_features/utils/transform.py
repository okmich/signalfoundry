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
from typing import Dict, Union, Tuple, Optional


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

# Emitted verbatim by FeatureEDA.recommend_transformations for heavy-tailed features.
QUANTILE_OR_RANK = 'quantile/rank'

# Transformations get_transformer can build. 'standardize' is deliberately absent: scaling is
# the pipeline scaler's job, not this function's, and callers rely on it being skipped here.
FITTABLE_TRANSFORMATIONS = ('yeo-johnson', 'box-cox', 'logit', 'log')


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


def primary_transformation(transformation_str) -> Optional[str]:
    """
    First fittable transformation named in a recommendation string, or None.

    Recommendation strings are comma separated and already ordered by the recommender's own
    priority, e.g. ``'box-cox, standardize'``. The previous implementation skipped the entire
    row whenever 'standardize' appeared anywhere in the string, so that example silently
    discarded the box-cox and passed the feature through untransformed -- and FeatureEDA emits
    exactly that pairing routinely, appending 'standardize' for high-CV features and 'box-cox'
    for non-normal positive ones.

    Parameters
    ----------
    transformation_str : str or NaN
        Comma-separated recommendation, e.g. 'logit, standardize'.

    Returns
    -------
    Optional[str]
        A member of FITTABLE_TRANSFORMATIONS, or None when the row names nothing fittable
        (including 'none' and a bare 'standardize').
    """
    if transformation_str is None:
        return None
    if not isinstance(transformation_str, str):
        try:
            if pd.isna(transformation_str):
                return None
        except (TypeError, ValueError):
            pass
        transformation_str = str(transformation_str)

    for token in transformation_str.split(','):
        token = token.strip()
        if token in FITTABLE_TRANSFORMATIONS:
            return token
    return None


def fit_transformation_recommendations(df: pd.DataFrame, transformation_df: pd.DataFrame, verbose: bool = True,
                                       **kwargs) -> Dict[str, object]:
    """
    Fit one transformer per recommended feature and return them for reuse.

    Call this ONCE, on the training window. Persist the result with joblib and hand it to
    :func:`apply_transformation_recommendations` at inference time, so no parameter is ever
    re-estimated from serving data.

    Parameters
    ----------
    df : pd.DataFrame
        Training feature frame. Every transformer's parameters come from this frame only.
    transformation_df : pd.DataFrame
        Recommendations from ``FeatureEDA.recommend_transformations()``.
    verbose : bool, default True
        Print a per-feature warning when a transformer cannot be fitted.
    **kwargs
        Forwarded to the transformer constructors (e.g. ``standardize=True``, ``epsilon=1e-9``).

    Returns
    -------
    Dict[str, object]
        Feature name -> fitted transformer. joblib-serialisable.

    Examples
    --------
    >>> fitted = fit_transformation_recommendations(train_df, recommendations)
    >>> joblib.dump(fitted, 'transformers.pkl')
    >>> # ... later, in the serving process ...
    >>> fitted = joblib.load('transformers.pkl')
    >>> X = apply_transformation_recommendations(live_df, recommendations, fitted_transformers=fitted)
    """
    fitted_transformers = {}

    for _, row in transformation_df.iterrows():
        feature = row['feature']
        primary_trans = primary_transformation(row.get('transformations'))
        if primary_trans is None:
            continue

        if feature not in df.columns:
            if verbose:
                print(f"Warning: Failed to fit {feature} with {primary_trans}: column not in dataframe")
            continue

        try:
            transformer = get_transformer(primary_trans, **kwargs)
            transformer.fit(df[feature])
            fitted_transformers[feature] = transformer
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to fit {feature} with {primary_trans}: {e}")

    return fitted_transformers


def apply_transformation_recommendations(df: pd.DataFrame, transformation_df: pd.DataFrame,
                                         replace_original: bool = True,
                                         fitted_transformers: Optional[Dict[str, object]] = None) -> pd.DataFrame:
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
    fitted_transformers : Dict[str, object], optional
        Transformers already fitted by :func:`fit_transformation_recommendations`. When
        supplied, NOTHING is refitted -- this is the inference-safe path. A recommended
        feature missing from the mapping raises instead of quietly falling back to a refit,
        because a silent per-column refit on serving data is precisely the train/serve skew
        this argument exists to prevent.

        When None (the default), each transformer is fitted on ``df``. That is correct for a
        training frame and unsafe for anything else: the Box-Cox lambda and log offset would
        be estimated from whatever batch you passed, so the same bar transforms differently
        depending on what else is in the batch.

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

    # Inference: reuse the training parameters, refitting nothing
    >>> fitted = fit_transformation_recommendations(train_df, recommendations)
    >>> df_transformed = apply_transformation_recommendations(live_df, recommendations,
    ...                                                       fitted_transformers=fitted)
    """
    df_result = df.copy()
    use_fitted = fitted_transformers is not None

    for _, row in transformation_df.iterrows():
        feature = row['feature']
        primary_trans = primary_transformation(row.get('transformations'))
        if primary_trans is None:
            continue

        if feature not in df.columns:
            print(f"Warning: Failed to transform {feature} with {primary_trans}: column not in dataframe")
            continue

        # Deliberately outside the try below: falling through to a refit here would reintroduce
        # the exact bug this path guards against, and doing so silently is worse than failing.
        if use_fitted and feature not in fitted_transformers:
            raise KeyError(
                f"No fitted transformer for '{feature}', which the recommendations require "
                f"('{primary_trans}'). Refitting it on this frame would re-estimate its parameters from "
                f"serving data. Fit it with fit_transformation_recommendations on the training frame, or "
                f"drop it from transformation_df."
            )

        try:
            if use_fitted:
                transformed = fitted_transformers[feature].transform(df[feature])
            else:
                transformed = get_transformer(primary_trans).fit_transform(df[feature])

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
