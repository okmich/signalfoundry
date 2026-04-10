"""
ModelRegistry — lightweight JSON-backed registry of trained model versions.

Tracks experiments, their model versions, and the currently promoted (live) model per experiment.
Provides promote/demote/compare operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ModelRegistry:
    """
    JSON-backed registry of trained model versions.

    Parameters
    ----------
    registry_path : str or Path
        Path to the JSON registry file (created if absent).
    """

    def __init__(self, registry_path: str = "experiments/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def register(self, experiment_name: str, model_name: str, metrics: Dict[str, Any], model_path: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a new model version.

        Parameters
        ----------
        experiment_name : str
            Logical grouping (e.g. 'btc_direction_rf').
        model_name : str
            Unique model identifier within the experiment.
        metrics : dict
            Evaluation metrics dict (ic, ic_ir, val_accuracy, etc.).
        model_path : str, optional
            Path to the saved model artefact.
        tags : dict, optional
            Free-form key-value tags (strategy, yardstick, framework, …).

        Returns
        -------
        str
            The registered version ID ('{experiment_name}:{model_name}').
        """
        if experiment_name not in self._data:
            self._data[experiment_name] = {"versions": {}, "promoted": None}

        version_id = model_name
        self._data[experiment_name]["versions"][version_id] = {
            "model_name": model_name,
            "metrics": metrics,
            "model_path": model_path,
            "tags": tags or {},
            "registered_at": datetime.now().isoformat(),
            "promoted": False,
        }
        self._save()
        return f"{experiment_name}:{version_id}"

    def promote(self, experiment_name: str, model_name: str) -> None:
        """Mark a version as the live (promoted) model for an experiment."""
        self._check_exists(experiment_name, model_name)
        # Demote any previously promoted version
        for v in self._data[experiment_name]["versions"].values():
            v["promoted"] = False
        self._data[experiment_name]["versions"][model_name]["promoted"] = True
        self._data[experiment_name]["promoted"] = model_name
        self._save()

    def get_promoted(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Return the promoted model version dict, or None if none promoted."""
        exp = self._data.get(experiment_name)
        if exp is None or exp["promoted"] is None:
            return None
        return exp["versions"].get(exp["promoted"])

    def get_best(self, experiment_name: str, metric: str = "ic_ir", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Return the version with the best value for the given metric."""
        exp = self._data.get(experiment_name)
        if exp is None:
            return None
        import math
        versions = [
            v for v in exp["versions"].values()
            if metric in v.get("metrics", {})
            and v["metrics"][metric] is not None
            and not (isinstance(v["metrics"][metric], float) and math.isnan(v["metrics"][metric]))
        ]
        if not versions:
            return None
        return max(versions, key=lambda v: v["metrics"][metric]) if higher_is_better \
            else min(versions, key=lambda v: v["metrics"][metric])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[str]:
        return list(self._data.keys())

    def list_versions(self, experiment_name: str) -> List[str]:
        exp = self._data.get(experiment_name, {})
        return list(exp.get("versions", {}).keys())

    def summary(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """DataFrame summary of all (or one experiment's) registered versions."""
        rows = []
        experiments = [experiment_name] if experiment_name else self.list_experiments()
        for exp_name in experiments:
            exp = self._data.get(exp_name, {})
            for version_id, v in exp.get("versions", {}).items():
                row = {
                    "experiment": exp_name,
                    "version_id": version_id,
                    "promoted": v.get("promoted", False),
                    "registered_at": v.get("registered_at"),
                    "model_path": v.get("model_path"),
                }
                row.update(v.get("metrics", {}))
                row.update({f"tag_{k}": val for k, val in v.get("tags", {}).items()})
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def _save(self) -> None:
        self.registry_path.write_text(json.dumps(self._data, indent=2, default=str))

    def _check_exists(self, experiment_name: str, model_name: str) -> None:
        if experiment_name not in self._data:
            raise KeyError(f"Experiment '{experiment_name}' not in registry.")
        if model_name not in self._data[experiment_name]["versions"]:
            raise KeyError(f"Model '{model_name}' not in experiment '{experiment_name}'.")