import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from . import RegimeEvaluationResult
from .experiment_runner import ExperimentResult
from .model_trainer import TrainedModel
from .supervised_trainer import SupervisedTrainedModel
from .objectives import ObjectiveScore, ModelRanking


class ExperimentTracker:
    """
    Track and manage experiment history.

    Examples
    --------
    >>> tracker = ExperimentTracker()
    >>> # Save experiment
    >>> tracker.save_experiment(result, tags=['hmm', 'directional'])
    >>> # Load experiment
    >>> result = tracker.load_experiment('experiment_id_123')
    >>> # List experiments
    >>> experiments = tracker.list_experiments()
    """

    def __init__(self, experiments_root: str = "experiments/research"):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        experiments_root : str, default="experiments/research"
            Root directory for all experiments
        """
        self.experiments_root = Path(experiments_root)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

        # Create index file if it doesn't exist
        self.index_file = self.experiments_root / "experiments_index.json"
        if not self.index_file.exists():
            self._save_index({})

    def save_experiment(self, result: ExperimentResult, tags: Optional[List[str]] = None,
                        notes: Optional[str] = None) -> str:
        # Generate experiment ID from output directory name
        output_path = Path(result.output_dir)
        experiment_id = output_path.name

        # Load current index
        index = self._load_index()

        # Create experiment entry
        best_model = result.get_best_model()

        entry = {
            "experiment_id": experiment_id,
            "experiment_name": result.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_path.absolute()),
            "tags": tags or [],
            "notes": notes or "",
            "best_model": best_model.model_name if best_model else None,
            "best_score": float(best_model.composite_score) if best_model else 0.0,
            "n_models": len(result.trained_models),
            "n_features": len(result.selected_features),
            "config": result.config,
        }
        index[experiment_id] = entry
        self._save_index(index)
        print(f"Experiment saved: {experiment_id}")
        return experiment_id

    def load_experiment(self, experiment_id: str) -> ExperimentResult:
        # Load index
        index = self._load_index()

        if experiment_id not in index:
            raise ValueError(f"Experiment not found: {experiment_id}")

        entry = index[experiment_id]
        output_dir = Path(entry["output_dir"])

        if not output_dir.exists():
            raise ValueError(f"Experiment directory not found: {output_dir}")

        print(f"Loading experiment: {experiment_id}")

        # Load features
        features_path = output_dir / "features" / "selected_features.json"
        with open(features_path, "r") as f:
            features_data = json.load(f)
            selected_features = features_data["selected_features"]

        # Load models — branch on experiment type so artifact format is matched
        models_dir = output_dir / "models"
        model_files = list(models_dir.glob("*.joblib"))
        _supervised_types = {
            "supervised_classification",
            "supervised_regression",
            "regime_classification",
        }
        _exp_model_type = entry.get("config", {}).get("model", {}).get("type", "clustering")
        _use_supervised_loader = _exp_model_type in _supervised_types

        trained_models = []
        for model_file in model_files:
            if model_file.stem.endswith("_metadata") or model_file.stem.endswith("_scaler"):
                continue

            model_name = model_file.stem
            try:
                if _use_supervised_loader:
                    trained_model = SupervisedTrainedModel.load(model_name, str(models_dir))
                else:
                    trained_model = TrainedModel.load(model_name, str(models_dir))
                trained_models.append(trained_model)
            except Exception as e:
                print(f"   Warning: Failed to load {model_name}: {e}")

        # Load evaluation results — branch on model type
        metrics_dir = output_dir / "metrics"
        model_type = entry.get("config", {}).get("model", {}).get("type", "clustering")
        supervised_types = {
            "supervised_classification",
            "supervised_regression",
            "regime_classification",
        }

        if model_type in supervised_types:
            from okmich_quant_research.models.supervised_evaluator import (
                SupervisedEvaluationResult,
            )

            metrics_summary = {}
            metrics_summary_path = metrics_dir / "metrics_summary.json"
            if metrics_summary_path.exists():
                with open(metrics_summary_path, "r") as f:
                    metrics_summary = json.load(f)

            per_window_metrics = pd.DataFrame()
            per_window_path = metrics_dir / "per_window_metrics.parquet"
            if per_window_path.exists():
                per_window_metrics = pd.read_parquet(per_window_path)

            evaluation_result = SupervisedEvaluationResult(
                metrics_summary=metrics_summary,
                per_window_metrics=per_window_metrics,
            )
        else:
            # Load path structure stats (may not exist if no models were trained)
            path_stats = pd.DataFrame()
            path_stats_path = metrics_dir / "path_structure_stats.parquet"
            if path_stats_path.exists():
                path_stats = pd.read_parquet(path_stats_path)

            regime_stats = None
            regime_stats_path = metrics_dir / "regime_returns_stats.parquet"
            if regime_stats_path.exists():
                regime_stats = pd.read_parquet(regime_stats_path)

            label_mapping = None
            label_mapping_path = metrics_dir / "label_mapping.json"
            if label_mapping_path.exists():
                with open(label_mapping_path, "r") as f:
                    label_mapping = json.load(f)

            # Load metrics summary (may not exist if no models were trained)
            metrics_summary = {}
            metrics_summary_path = metrics_dir / "metrics_summary.json"
            if metrics_summary_path.exists():
                with open(metrics_summary_path, "r") as f:
                    metrics_summary = json.load(f)

            evaluation_result = RegimeEvaluationResult(
                path_structure_stats=path_stats,
                regime_returns_stats=regime_stats,
                label_mapping=label_mapping,
                metrics_summary=metrics_summary,
            )

        rankings_path = metrics_dir / "rankings.json"
        rankings = []
        if not rankings_path.exists():
            print(f"   Warning: rankings.json not found in {metrics_dir}; skipping rankings.")
        else:
            with open(rankings_path, "r") as f:
                rankings_data = json.load(f)
        for rank_dict in rankings_data.get("rankings", []) if rankings_path.exists() else []:
            objective_scores = []
            for obj_name, obj_data in rank_dict["objectives"].items():
                obj_score = ObjectiveScore(
                    name=obj_name,
                    value=(
                        obj_data["value"] if obj_data["value"] is not None else np.nan
                    ),
                    normalized_score=obj_data["normalized_score"],
                    weight=obj_data["weight"],
                    achieved=obj_data["achieved"],
                )
                objective_scores.append(obj_score)

            ranking = ModelRanking(
                model_name=rank_dict["model_name"],
                composite_score=rank_dict["composite_score"],
                objective_scores=objective_scores,
                rank=rank_dict["rank"],
            )
            rankings.append(ranking)

        # Create mock features_df (not saved, would need to reconstruct)
        # For now, create empty with correct index
        features_df = pd.DataFrame()

        # Create result
        result = ExperimentResult(
            experiment_name=entry["experiment_name"],
            config=entry["config"],
            features_df=features_df,
            selected_features=selected_features,
            trained_models=trained_models,
            evaluation_result=evaluation_result,
            rankings=rankings,
            output_dir=str(output_dir),
        )
        print(f"   [OK] Loaded {len(trained_models)} models")
        best_model = result.get_best_model()
        if best_model:
            print(f"   [OK] Best model: {best_model.model_name}")
        else:
            print(f"   [OK] No models available")
        return result

    def list_experiments(self, tags: Optional[List[str]] = None, limit: int = 10) -> pd.DataFrame:
        index = self._load_index()
        if not index:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(index, orient="index")
        # Filter by tags if provided
        if tags:
            df = df[df["tags"].apply(lambda x: any(tag in x for tag in tags))]

        # Sort by timestamp (newest first)
        df = df.sort_values("timestamp", ascending=False)

        # Limit results
        df = df.head(limit)
        # Select key columns
        display_cols = [
            "experiment_id",
            "experiment_name",
            "timestamp",
            "best_model",
            "best_score",
            "n_models",
            "tags",
        ]
        available_cols = [col for col in display_cols if col in df.columns]
        return df[available_cols].reset_index(drop=True)

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        index = self._load_index()

        comparison = []
        for exp_id in experiment_ids:
            if exp_id not in index:
                print(f"Warning: Experiment {exp_id} not found")
                continue

            entry = index[exp_id]
            comparison.append(
                {
                    "experiment_id": exp_id,
                    "name": entry["experiment_name"],
                    "timestamp": entry["timestamp"],
                    "best_model": entry["best_model"],
                    "best_score": entry["best_score"],
                    "n_models": entry["n_models"],
                    "n_features": entry["n_features"],
                }
            )

        return pd.DataFrame(comparison)

    def delete_experiment(self, experiment_id: str, delete_files: bool = False):
        index = self._load_index()

        if experiment_id not in index:
            raise ValueError(f"Experiment not found: {experiment_id}")

        entry = index[experiment_id]

        # Delete files if requested
        if delete_files:
            output_dir = Path(entry["output_dir"])
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"Deleted files: {output_dir}")

        # Remove from index
        del index[experiment_id]
        self._save_index(index)

        print(f"Deleted experiment: {experiment_id}")

    def _load_index(self) -> Dict[str, Any]:
        """Load experiments index."""
        if not self.index_file.exists():
            return {}

        with open(self.index_file, "r") as f:
            return json.load(f)

    def _save_index(self, index: Dict[str, Any]):
        """Save experiments index."""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2, default=str)


def load_experiment(experiment_id: str, experiments_root: str = "experiments/research") -> ExperimentResult:
    """
    Load a saved experiment (convenience function).

    Examples
    --------
    >>> result = load_experiment('hmm_directional_20241210_123456')
    >>> labels = result.get_labels_df()
    """
    tracker = ExperimentTracker(experiments_root=experiments_root)
    return tracker.load_experiment(experiment_id)
