"""
RegimeClassificationTrainer — walk-forward trainer for regime classification pipeline.

Supports all four labelling strategies (A/B/B2/C) via the RegimeLabelGenerator interface. The key difference from
SupervisedTrainer is that labels are generated *per fold* inside the walk-forward loop (required for B/B2/A), or
once up-front for Strategy C (fully causal, no leakage).

For Strategy A (HmmDirectStrategy): the HMM is fitted per fold and is the deployed model — no downstream classifier.

For Strategies B/B2/C: a supervised sklearn classifier is trained per fold on the generated labels.
"""

import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from okmich_quant_labelling.regime.label_generator import HmmDirectStrategy, RegimeLabelGenerator
from okmich_quant_ml.hmm import InferenceMode

from .supervised_trainer import SupervisedTrainedModel, SupervisedModelMetadata, WalkForwardResult

SKLEARN_CLASSIFIERS = {
    "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "LogisticRegression": "sklearn.linear_model",
    "SVC": "sklearn.svm",
    "XGBClassifier": "xgboost",
    "LGBMClassifier": "lightgbm",
}


class RegimeClassificationTrainer:
    """
    Walk-forward trainer for the regime classification pipeline.

    Parameters
    ----------
    label_generator : RegimeLabelGenerator
        Strategy A/B/B2/C label generator instance.
    random_state : int, default=42
    scale_features : bool, default=True
    """

    def __init__(self, label_generator: RegimeLabelGenerator, random_state: int = 42, scale_features: bool = True):
        self.label_generator = label_generator
        self.random_state = random_state
        self.scale_features = scale_features

    def walk_forward_train(self, features_df: pd.DataFrame, feature_cols: List[str], price_col: str = "close",
                           return_col: str = "return", train_period: int = 5000, test_period: int = 1000,
                           step_period: int = 500, anchored: bool = True, max_train_bars: Optional[int] = None,
                           embargo_bars: int = 0, model_configs: Optional[List[Dict[str, Any]]] = None) -> List[SupervisedTrainedModel]:
        """
        Walk-forward training for regime classification.

        For Strategy A: fits HMM per fold and returns HMM-backed models.
        For B/B2/C: generates labels per fold, trains sklearn classifiers.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full dataset with features and OHLCV columns.
        feature_cols : list of str
            Feature columns used as classifier inputs.
        price_col, return_col : str
            Forwarded to label_generator.generate_labels().
        train_period, test_period, step_period : int
            Walk-forward window parameters.
        anchored : bool
            Whether training window starts from origin (True = expanding).
        max_train_bars : int, optional
            Cap on training window (capped-expanding WFA).
        embargo_bars : int
            Bars to skip between train end and test start.
            Should be >= label_generator.warmup_bars.
        model_configs : list of dict, optional
            sklearn model configurations.  Ignored for Strategy A.
            Each dict: {'name': str, 'algorithm': str, 'hyperparameters': dict}.
        """
        print("=" * 80)
        print("REGIME CLASSIFICATION — WALK-FORWARD TRAINING")
        print(f"  strategy: {self.label_generator.__class__.__name__}  "
              f"leaks_future={self.label_generator.leaks_future}")
        print("=" * 80)

        windows = self._calculate_windows(
            len(features_df), train_period, test_period, step_period,
            anchored, max_train_bars, embargo_bars,
        )
        print(f"  {len(windows)} folds | embargo={embargo_bars} | "
              f"warmup_bars={self.label_generator.warmup_bars}")

        if not windows:
            min_required = train_period + embargo_bars + test_period
            raise ValueError(
                f"No walk-forward windows could be generated. "
                f"Dataset has {len(features_df)} bars but needs at least "
                f"{min_required} (train={train_period} + embargo={embargo_bars} "
                f"+ test={test_period}). Reduce window sizes or provide more data."
            )

        is_strategy_a = isinstance(self.label_generator, HmmDirectStrategy)
        if is_strategy_a:
            return self._train_strategy_a(features_df, feature_cols, windows, price_col, return_col)

        if not model_configs:
            raise ValueError("model_configs required for strategies B/B2/C.")
        return self._train_classifiers(features_df, feature_cols, windows, price_col, return_col, model_configs)

    # ------------------------------------------------------------------
    # Strategy A — HMM direct
    # ------------------------------------------------------------------

    def _train_strategy_a(self, features_df: pd.DataFrame, feature_cols: List[str],
                          windows: List[Tuple[int, int, int, int]], price_col: str, return_col: str) -> List[SupervisedTrainedModel]:
        """Fit HMM per fold; the HMM IS the prediction model."""
        all_results = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"  [fold {fold_idx}] train={train_start}:{train_end}  "
                  f"test={test_start}:{test_end}")

            train_df = features_df.iloc[train_start:train_end]
            test_df = features_df.iloc[test_start:test_end]
            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values

            # generate_labels fits HMM + runs Viterbi on training fold
            labels = self.label_generator.generate_labels(train_df, X=X_train, price_col=price_col, return_col=return_col)

            # Live inference: filtering mode on test fold
            hmm = self.label_generator.fitted_hmm
            original_mode = hmm.inference_mode
            hmm.inference_mode = InferenceMode.FILTERING
            try:
                predictions = hmm.predict(X_test)
            finally:
                hmm.inference_mode = original_mode

            true_labels = test_df[labels.name if labels.name in test_df.columns else "regime_label"].values \
                if labels.name in test_df.columns else np.full(len(test_df), np.nan)

            result = WalkForwardResult(
                window_idx=fold_idx,
                train_start=str(train_df.index[0]),
                train_end=str(train_df.index[-1]),
                test_start=str(test_df.index[0]),
                test_end=str(test_df.index[-1]),
                model_name=f"hmm_direct_{fold_idx}",
                model=hmm,
                scaler=None,
                predictions=predictions,
                true_labels=true_labels,
                probabilities=None,
                metrics={},
                hyperparameters={},
                feature_names=feature_cols,
                n_train_samples=len(train_df),
                n_test_samples=len(test_df),
            )
            all_results.append(result)

        fitted_hmm = self.label_generator.fitted_hmm
        if fitted_hmm is None:
            raise RuntimeError("Strategy-A training completed but fitted_hmm is None. No folds ran or HMM fitting failed.")

        metadata = SupervisedModelMetadata(
            model_name="hmm_direct",
            model_type="hmm",
            algorithm=self.label_generator._hmm.__class__.__name__,
            task_type="classification",
            hyperparameters={},
            feature_names=feature_cols,
            n_train_samples=windows[-1][1] - windows[-1][0] if windows else 0,
            n_classes=self.label_generator._hmm.n_states,
        )
        return [SupervisedTrainedModel(model=fitted_hmm, metadata=metadata, walk_forward_results=all_results)]

    # ------------------------------------------------------------------
    # Strategies B / B2 / C — label generation + classifier
    # ------------------------------------------------------------------

    def _train_classifiers(self, features_df: pd.DataFrame, feature_cols: List[str],
                           windows: List[Tuple[int, int, int, int]], price_col: str, return_col: str,
                           model_configs: List[Dict[str, Any]]) -> List[SupervisedTrainedModel]:
        from sklearn.metrics import accuracy_score, f1_score

        trained_models = []

        for model_config in model_configs:
            model_name = model_config["name"]
            algorithm = model_config["algorithm"]
            params = model_config.get("hyperparameters", {})
            if isinstance(params, dict) and any(isinstance(v, list) for v in params.values()):
                # Single set of params (not a grid), pick first value of any list
                params = {k: v[0] if isinstance(v, list) else v for k, v in params.items()}

            print(f"\n[{model_name}] {algorithm}")
            all_results = []

            for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
                train_df = features_df.iloc[train_start:train_end]
                test_df = features_df.iloc[test_start:test_end]
                X_train_raw = train_df[feature_cols].values
                X_test_raw = test_df[feature_cols].values

                # Generate labels on training fold
                labels = self.label_generator.generate_labels(
                    train_df, X=X_train_raw, price_col=price_col, return_col=return_col
                )

                # Drop warmup NaNs
                valid_mask = ~labels.isna()
                valid_idx = np.where(valid_mask)[0]
                if len(valid_idx) < 10:
                    print(f"  [fold {fold_idx}] skipped — too few valid labels ({valid_mask.sum()})")
                    continue

                X_train = X_train_raw[valid_idx]
                y_train = labels.values[valid_idx].astype(float)

                # Scale
                scaler = None
                if self.scale_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test_raw)
                else:
                    X_test = X_test_raw

                # Train classifier
                model = self._create_sklearn_model(algorithm, params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # Metrics against test labels (best effort — use predictions vs mode of y_train as baseline)
                metrics = {
                    "n_labels": int(len(y_train)),
                    "label_classes": sorted(np.unique(y_train).tolist()),
                }
                # If test fold has ground-truth labels (Strategy C can use causal labels on test fold)
                if not self.label_generator.leaks_future:
                    test_labels = self.label_generator.generate_labels(
                        test_df, price_col=price_col, return_col=return_col
                    )
                    valid_test = ~test_labels.isna()
                    if valid_test.sum() >= 5:
                        y_test = test_labels.values[valid_test]
                        y_pred_valid = predictions[valid_test]
                        metrics["val_accuracy"] = float(accuracy_score(y_test, y_pred_valid))
                        metrics["val_f1_weighted"] = float(f1_score(y_test, y_pred_valid, average="weighted", zero_division=0))

                result = WalkForwardResult(
                    window_idx=fold_idx,
                    train_start=str(train_df.index[0]),
                    train_end=str(train_df.index[-1]),
                    test_start=str(test_df.index[0]),
                    test_end=str(test_df.index[-1]),
                    model_name=f"{model_name}_{fold_idx}",
                    model=model,
                    scaler=scaler,
                    predictions=predictions,
                    true_labels=np.full(len(test_df), np.nan),
                    probabilities=probabilities,
                    metrics=metrics,
                    hyperparameters=params,
                    feature_names=feature_cols,
                    n_train_samples=len(X_train),
                    n_test_samples=len(X_test),
                )
                all_results.append(result)

            if not all_results:
                print(f"  [{model_name}] no folds completed — skipping")
                continue

            metadata = SupervisedModelMetadata(
                model_name=model_name,
                model_type="sklearn",
                algorithm=algorithm,
                task_type="classification",
                hyperparameters=params,
                feature_names=feature_cols,
                n_train_samples=windows[-1][1] - windows[-1][0] if windows else 0,
                n_classes=None,
            )
            trained_models.append(SupervisedTrainedModel(
                model=all_results[-1].model,
                metadata=metadata,
                scaler=all_results[-1].scaler,
                walk_forward_results=all_results,
            ))
            print(f"  [{model_name}] {len(all_results)} folds completed")

        return trained_models

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_windows(self, total_length: int, train_period: int, test_period: int, step_period: int,
                           anchored: bool, max_train_bars: Optional[int], embargo_bars: int) -> List[Tuple[int, int, int, int]]:
        windows = []
        n_windows = (total_length - train_period - test_period - embargo_bars) // step_period + 1

        for i in range(n_windows):
            if anchored:
                raw_start = 0
                raw_end = train_period + (i * step_period)
                if max_train_bars is not None and (raw_end - raw_start) > max_train_bars:
                    train_end_idx = raw_end
                    train_start_idx = train_end_idx - max_train_bars
                else:
                    train_start_idx, train_end_idx = raw_start, raw_end
            else:
                train_start_idx = i * step_period
                train_end_idx = train_start_idx + train_period

            test_start_idx = train_end_idx + embargo_bars
            test_end_idx = test_start_idx + test_period

            if test_end_idx > total_length:
                break

            windows.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        return windows

    def _create_sklearn_model(self, algorithm: str, params: Dict[str, Any]):
        if algorithm not in SKLEARN_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier '{algorithm}'. "
                f"Available: {list(SKLEARN_CLASSIFIERS)}"
            )
        module = importlib.import_module(SKLEARN_CLASSIFIERS[algorithm])
        cls = getattr(module, algorithm)
        import inspect
        sig = inspect.signature(cls)
        model_params = params.copy()
        if "random_state" in sig.parameters:
            model_params.setdefault("random_state", self.random_state)
        return cls(**model_params)