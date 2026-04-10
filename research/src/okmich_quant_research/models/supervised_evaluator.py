"""
Supervised Learning Evaluator.

This module provides evaluation infrastructure for supervised learning models, aggregating metrics from walk-forward
validation and generating evaluation reports.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, \
    mean_absolute_percentage_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from .supervised_trainer import SupervisedTrainedModel, WalkForwardResult


def _convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    return obj


@dataclass
class SupervisedEvaluationResult:
    """
    Results from supervised model evaluation.

    Attributes
    ----------
    metrics_summary : dict
        Summary metrics per model (model_name -> metrics)
    per_window_metrics : pd.DataFrame
        Detailed metrics for each walk-forward window
    aggregate_predictions : pd.Series
        All predictions concatenated across windows
    aggregate_true_labels : pd.Series
        All true labels concatenated across windows
    aggregate_probabilities : pd.Series, optional
        All probabilities concatenated (classification only)
    confusion_matrices : dict, optional
        Confusion matrices per model (classification only)
    """

    metrics_summary: Dict[str, Dict[str, float]]
    per_window_metrics: pd.DataFrame
    aggregate_predictions: Optional[pd.Series] = None
    aggregate_true_labels: Optional[pd.Series] = None
    aggregate_probabilities: Optional[pd.Series] = None
    confusion_matrices: Optional[Dict[str, np.ndarray]] = None

    def save(self, output_dir: str):
        """Save evaluation results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics summary
        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(_convert_to_json_serializable(self.metrics_summary), f, indent=2)

        # Save per-window metrics
        if not self.per_window_metrics.empty:
            self.per_window_metrics.to_parquet(output_dir / "per_window_metrics.parquet", index=False)
            self.per_window_metrics.to_csv(output_dir / "per_window_metrics.csv", index=False)

        # Save confusion matrices
        if self.confusion_matrices:
            cm_path = output_dir / "confusion_matrices.json"
            with open(cm_path, "w") as f:
                json.dump(
                    {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in self.confusion_matrices.items()
                    },
                    f,
                    indent=2,
                )

        # Save predictions if available
        if self.aggregate_predictions is not None:
            predictions_df = pd.DataFrame(
                {
                    "predictions": self.aggregate_predictions,
                    "true_labels": self.aggregate_true_labels,
                }
            )
            if self.aggregate_probabilities is not None:
                predictions_df["probabilities"] = self.aggregate_probabilities
            predictions_df.to_parquet(output_dir / "aggregate_predictions.parquet", index=True)

    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get metrics for a specific model."""
        return self.metrics_summary.get(model_name, {})

    def __repr__(self):
        n_models = len(self.metrics_summary)
        n_windows = len(self.per_window_metrics) if not self.per_window_metrics.empty else 0
        has_cm = self.confusion_matrices is not None

        return (
            f"SupervisedEvaluationResult(\n"
            f"  models_evaluated={n_models},\n"
            f"  walk_forward_windows={n_windows},\n"
            f"  confusion_matrices={has_cm}\n"
            f")"
        )


class SupervisedEvaluator:
    """
    Evaluates supervised learning models.

    Supports both classification and regression metrics.

    Examples
    --------
    >>> evaluator = SupervisedEvaluator()
    >>>
    >>> # Evaluate classification models
    >>> result = evaluator.evaluate(trained_models, task_type='classification')
    >>>
    >>> # Get metrics for specific model
    >>> metrics = result.get_model_metrics('random_forest')
    """

    def __init__(self):
        """Initialize supervised evaluator."""
        pass

    def evaluate(self, trained_models: List[SupervisedTrainedModel], task_type: str = "classification") \
            -> SupervisedEvaluationResult:
        """
        Evaluate trained supervised models.

        Parameters
        ----------
        trained_models : list of SupervisedTrainedModel
            Models to evaluate (with walk-forward results)
        task_type : str
            'classification' or 'regression'

        Returns
        -------
        SupervisedEvaluationResult
            Evaluation results
        """
        print("=" * 80)
        print("SUPERVISED MODEL EVALUATION")
        print("=" * 80)
        if task_type == "classification":
            return self._evaluate_classification(trained_models)
        else:
            return self._evaluate_regression(trained_models)

    def _evaluate_classification(self, trained_models: List[SupervisedTrainedModel]) -> SupervisedEvaluationResult:
        """Evaluate classification models."""
        print("\nEvaluating classification models...")

        metrics_summary = {}
        per_window_records = []
        confusion_matrices = {}

        for model in trained_models:
            model_name = model.metadata.model_name
            print(f"\n[{model_name}]")

            # Aggregate predictions across windows
            all_predictions = []
            all_true_labels = []
            all_probabilities = []

            for result in model.walk_forward_results:
                all_predictions.extend(result.predictions.tolist())
                all_true_labels.extend(result.true_labels.tolist())
                if result.probabilities is not None:
                    # 1-D: already binary class-1 probabilities
                    # 2-D with 2 cols: binary full matrix — take col 1
                    # 2-D with >2 cols: multiclass — keep full rows so downstream
                    #   AUC can use the full probability matrix (OVR)
                    proba = result.probabilities
                    if proba.ndim == 1:
                        all_probabilities.extend(proba.tolist())
                    elif proba.shape[1] == 2:
                        all_probabilities.extend(proba[:, 1].tolist())
                    else:
                        all_probabilities.extend(proba.tolist())

                # Add per-window record
                per_window_records.append(
                    {
                        "model_name": model_name,
                        "window_idx": result.window_idx,
                        "train_start": result.train_start,
                        "train_end": result.train_end,
                        "test_start": result.test_start,
                        "test_end": result.test_end,
                        "n_train": result.n_train_samples,
                        "n_test": result.n_test_samples,
                        **result.metrics,
                    }
                )

            # Calculate aggregate metrics
            y_true = np.array(all_true_labels)
            y_pred = np.array(all_predictions)
            y_proba = np.array(all_probabilities) if all_probabilities else None

            aggregate_metrics = self._calculate_classification_metrics(y_true, y_pred, y_proba)

            # Add model's stored aggregate metrics
            aggregate_metrics.update(model.aggregate_metrics)
            metrics_summary[model_name] = aggregate_metrics

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[model_name] = cm

            # Print summary
            print(f"   Accuracy: {aggregate_metrics.get('accuracy', 0):.4f}")
            print(f"   Precision: {aggregate_metrics.get('precision', 0):.4f}")
            print(f"   Recall: {aggregate_metrics.get('recall', 0):.4f}")
            print(f"   F1 Score: {aggregate_metrics.get('f1_score', 0):.4f}")
            if 'auc_roc' in aggregate_metrics and not np.isnan(aggregate_metrics['auc_roc']):
                print(f"   AUC-ROC: {aggregate_metrics['auc_roc']:.4f}")

        per_window_df = pd.DataFrame(per_window_records)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        return SupervisedEvaluationResult(metrics_summary=metrics_summary,
                                          per_window_metrics=per_window_df,
                                          confusion_matrices=confusion_matrices)

    def _evaluate_regression(self, trained_models: List[SupervisedTrainedModel]) -> SupervisedEvaluationResult:
        """Evaluate regression models."""
        print("\nEvaluating regression models...")

        metrics_summary = {}
        per_window_records = []

        for model in trained_models:
            model_name = model.metadata.model_name
            print(f"\n[{model_name}]")

            # Aggregate predictions across windows
            all_predictions = []
            all_true_labels = []

            for result in model.walk_forward_results:
                all_predictions.extend(result.predictions.tolist())
                all_true_labels.extend(result.true_labels.tolist())

                # Add per-window record
                per_window_records.append(
                    {
                        "model_name": model_name,
                        "window_idx": result.window_idx,
                        "train_start": result.train_start,
                        "train_end": result.train_end,
                        "test_start": result.test_start,
                        "test_end": result.test_end,
                        "n_train": result.n_train_samples,
                        "n_test": result.n_test_samples,
                        **result.metrics,
                    }
                )

            # Calculate aggregate metrics
            y_true = np.array(all_true_labels)
            y_pred = np.array(all_predictions)

            aggregate_metrics = self._calculate_regression_metrics(y_true, y_pred)
            aggregate_metrics.update(self._compute_ic_ir(model.walk_forward_results))

            # Add model's stored aggregate metrics
            aggregate_metrics.update(model.aggregate_metrics)

            metrics_summary[model_name] = aggregate_metrics

            # Print summary
            print(f"   MSE: {aggregate_metrics.get('mse', 0):.6f}")
            print(f"   RMSE: {aggregate_metrics.get('rmse', 0):.6f}")
            print(f"   MAE: {aggregate_metrics.get('mae', 0):.6f}")
            print(f"   R2: {aggregate_metrics.get('r2', 0):.4f}")
            print(f"   IC: {aggregate_metrics.get('ic', float('nan')):.4f}  IC-IR: {aggregate_metrics.get('ic_ir', float('nan')):.4f}")
            print(f"   Ljung-Box p: {aggregate_metrics.get('ljung_box_pvalue', float('nan')):.4f}")

        per_window_df = pd.DataFrame(per_window_records)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)

        return SupervisedEvaluationResult(
            metrics_summary=metrics_summary,
            per_window_metrics=per_window_df,
        )

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          y_proba: Optional[np.ndarray] = None, average: str = "weighted") -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            ),
        }

        # AUC-ROC
        if y_proba is not None and len(y_proba) > 0:
            try:
                n_classes = len(np.unique(y_true))
                y_proba_arr = np.array(y_proba)
                if n_classes == 2:
                    # Binary: y_proba is 1-D class-1 probabilities
                    score_input = y_proba_arr if y_proba_arr.ndim == 1 else y_proba_arr[:, 1]
                    metrics["auc_roc"] = float(roc_auc_score(y_true, score_input))
                else:
                    # Multiclass: y_proba is 2-D (N, K); use OVR macro-average
                    if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == n_classes:
                        metrics["auc_roc"] = float(
                            roc_auc_score(y_true, y_proba_arr, multi_class="ovr", average="macro")
                        )
                    else:
                        metrics["auc_roc"] = np.nan
            except ValueError:
                metrics["auc_roc"] = np.nan
        else:
            metrics["auc_roc"] = np.nan

        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics including IC, IC-IR, and Ljung-Box."""

        mse = mean_squared_error(y_true, y_pred)

        # IC — Spearman correlation between predictions and realised values
        ic, ic_pvalue = scipy_stats.spearmanr(y_pred, y_true)

        # Ljung-Box test on prediction errors — adaptive lag to avoid crash on short series
        errors = y_true - y_pred
        max_safe_lag = max(1, len(errors) // 2 - 1)
        lb_lag = min(10, max_safe_lag)
        if len(errors) < 4 or lb_lag < 1:
            lb_stat, lb_pvalue = np.nan, np.nan
        else:
            lb_result = acorr_ljungbox(errors, lags=lb_lag, return_df=True)
            lb_stat = float(lb_result["lb_stat"].iloc[-1])
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])

        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            "ic": float(ic) if not np.isnan(ic) else np.nan,
            "ic_pvalue": float(ic_pvalue) if not np.isnan(ic_pvalue) else np.nan,
            "ljung_box_stat": lb_stat,
            "ljung_box_pvalue": lb_pvalue,
        }

    def _compute_ic_ir(self, window_results: List[WalkForwardResult]) -> Dict[str, float]:
        """Compute IC-IR from per-fold IC values."""
        ic_values = []
        for r in window_results:
            if len(r.predictions) >= 5:
                ic, _ = scipy_stats.spearmanr(r.predictions, r.true_labels)
                if not np.isnan(ic):
                    ic_values.append(ic)
        if len(ic_values) < 2:
            return {"ic_mean": np.nan, "ic_ir": np.nan, "ic_std": np.nan}
        ic_arr = np.array(ic_values)
        ic_std = float(np.std(ic_arr, ddof=1))
        return {
            "ic_mean": float(np.mean(ic_arr)),
            "ic_std": ic_std,
            "ic_ir": float(np.mean(ic_arr) / ic_std) if ic_std > 0 else np.nan,
        }

    def compare_models(self, trained_models: List[SupervisedTrainedModel], task_type: str = "classification") -> pd.DataFrame:
        result = self.evaluate(trained_models, task_type)

        comparison_data = []
        for model_name, metrics in result.metrics_summary.items():
            row = {"model": model_name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by primary metric
        if task_type == "classification":
            sort_col = "f1_score" if "f1_score" in df.columns else "accuracy"
            df = df.sort_values(sort_col, ascending=False)
        else:
            sort_col = "r2" if "r2" in df.columns else "mse"
            ascending = sort_col != "r2"
            df = df.sort_values(sort_col, ascending=ascending)

        return df.reset_index(drop=True)
