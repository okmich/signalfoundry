from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def prepare_meta_dataset(features: pd.DataFrame, labels: pd.DataFrame, target_col: str = "label") -> \
        tuple[pd.DataFrame, pd.Series]:
    """
    Align features with target labels at event timestamps.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix indexed by timestamp (continuous, covers full history).
        Should contain all features you want to use for prediction.
    labels : pd.DataFrame
        Labels indexed by event timestamp (sparse, only at signal times).
        Output from any labeler (TBM, regime, custom).
    target_col : str
        Column name in labels to use as target. Default: "label"

    Returns
    -------
    X : pd.DataFrame
        Feature matrix at event times only (rows aligned to labels)
    y : pd.Series
        Target series aligned to event times

    Raises
    ------
    ValueError
        If target_col not in labels, or no overlapping timestamps

    Examples
    --------
    >>> features = pd.DataFrame({"rsi": rsi_values, "atr": atr_values}, index=prices.index)
    >>> tbm_labels = apply_tbm(prices, events, config)
    >>> X, y = prepare_meta_dataset(features, tbm_labels, target_col="label")
    """
    if target_col not in labels.columns:
        raise ValueError(f"target_col '{target_col}' not found in labels. "
                         f"Available columns: {list(labels.columns)}")

    # Find overlapping timestamps
    common_idx = features.index.intersection(labels.index)

    if len(common_idx) == 0:
        raise ValueError("No overlapping timestamps between features and labels")

    # Align to label timestamps
    X = features.loc[common_idx].copy()
    y = labels.loc[common_idx, target_col].copy()

    # Drop rows with NaN in features or target
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


@dataclass
class MetaModelConfig:
    """Configuration for MetaModel behavior."""

    model_type: Literal["classifier", "regressor"] = "classifier"
    threshold: float = 0.5  # Probability threshold for should_trade()


class MetaModel:
    """
    Wrapper for sklearn-compatible estimators for meta-labeling.

    Provides a consistent interface for training, optimization, and inference
    with proper handling of classification vs regression tasks.

    Parameters
    ----------
    config : MetaModelConfig
        Configuration for model behavior
    estimator : Any
        Any sklearn-compatible estimator (XGBClassifier, RandomForest, etc.)

    Attributes
    ----------
    config : MetaModelConfig
        Model configuration
    estimator : Any
        The underlying sklearn estimator
    is_fitted : bool
        Whether the model has been fitted
    feature_names : list
        Feature names from training data
    classes_ : ndarray
        Class labels (classifier only)

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> config = MetaModelConfig(model_type="classifier", threshold=0.6)
    >>> model = MetaModel(config, XGBClassifier(n_estimators=100))
    >>> model.fit(X_train, y_train)
    >>> probs = model.predict_proba(X_test)
    >>> should_act = model.should_trade(X_test)
    """

    def __init__(self, config: MetaModelConfig, estimator: Any):
        self.config = config
        self.estimator = estimator
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaModel":
        self.feature_names = list(X.columns)

        # Fit the estimator
        self.estimator.fit(X, y)
        self.is_fitted = True

        # Store classes for classifier
        if self.config.model_type == "classifier" and hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_

        return self

    def optimize(self, X: pd.DataFrame, y: pd.Series, param_space: dict, n_splits: int = 5, scoring: str = "accuracy",
                 n_iter: int = 50, random_state: int = 42) -> "MetaModel":
        """
        Find best hyperparameters using walk-forward cross-validation.

        Uses RandomizedSearchCV with TimeSeriesSplit for proper temporal CV.
        Updates self.estimator with the best model after search.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target series
        param_space : dict
            Hyperparameter search space (sklearn-style)
        n_splits : int
            Number of walk-forward splits. Default: 5
        scoring : str
            Metric to optimize (e.g., "accuracy", "f1_weighted", "roc_auc").
            Default: "accuracy"
        n_iter : int
            Number of random search iterations. Default: 50
        random_state : int
            Random seed for reproducibility. Default: 42

        Returns
        -------
        self : MetaModel
            Model with optimized estimator

        Examples
        --------
        >>> param_space = {
        ...     "n_estimators": [50, 100, 200],
        ...     "max_depth": [3, 5, 7],
        ... }
        >>> model.optimize(X, y, param_space, n_splits=5, scoring="f1_weighted")
        """
        self.feature_names = list(X.columns)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Randomized search
        search = RandomizedSearchCV(estimator=self.estimator, param_distributions=param_space, n_iter=n_iter,
                                    scoring=scoring, cv=tscv, random_state=random_state, n_jobs=-1)
        search.fit(X, y)

        # Update estimator with best model
        self.estimator = search.best_estimator_
        self.is_fitted = True

        if self.config.model_type == "classifier" and hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_

        # Store optimization results
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.cv_results_ = search.cv_results_

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        X = self._check_features(X)

        predictions = self.estimator.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict probability of positive outcome (classifier only).

        For binary classification, returns P(positive class).
        For multiclass, returns probability of the highest class.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.Series
            Probability values [0, 1], indexed same as X

        Raises
        ------
        ValueError
            If model is a regressor (no probability support)
        """
        self._check_fitted()
        X = self._check_features(X)

        if self.config.model_type != "classifier":
            raise ValueError("predict_proba is only available for classifiers")

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(f"Estimator {type(self.estimator).__name__} "
                             "does not support predict_proba")

        proba = self.estimator.predict_proba(X)

        # Handle different class configurations
        if self.classes_ is not None and len(self.classes_) == 2:
            # Binary: return probability of positive class (class 1)
            # Find index of class 1 (or highest class if no explicit 1)
            if 1 in self.classes_:
                pos_idx = list(self.classes_).index(1)
            else:
                pos_idx = -1  # Highest class
            prob_positive = proba[:, pos_idx]
        else:
            # Multiclass: return max probability
            prob_positive = proba.max(axis=1)

        return pd.Series(prob_positive, index=X.index, name="probability")

    def should_trade(self, X: pd.DataFrame) -> pd.Series:
        """
        Determine whether to act on signal based on probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.Series
            Boolean series: True if prob >= threshold
        """
        proba = self.predict_proba(X)
        return pd.Series(proba >= self.config.threshold, index=X.index, name="should_trade")

    def _check_fitted(self):
        """Verify model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def _check_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Verify feature columns match training data and return X with correct column selection/order."""
        if self.feature_names is None:
            return X

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Select and reorder to exactly match training schema (drops extras, enforces order).
        return X[self.feature_names]

    def get_feature_importance(self) -> Optional[pd.Series]:
        self._check_fitted()

        if hasattr(self.estimator, "feature_importances_"):
            return pd.Series(
                self.estimator.feature_importances_,
                index=self.feature_names,
                name="importance"
            ).sort_values(ascending=False)

        return None
