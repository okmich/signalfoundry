"""
PredictionStore — lightweight parquet-based store for walk-forward predictions.

Persists per-fold predictions so that post-hoc analysis, monitoring, and drift detection can be run without re-running the full experiment.
"""

import json
from pathlib import Path

import pandas as pd


class PredictionStore:
    """
    Stores and retrieves walk-forward predictions keyed by experiment name.

    Directory layout::

        store_root/
          <experiment_name>/
            predictions.parquet   — all folds concatenated
            metadata.json         — column schema and creation time

    Parameters
    ----------
    store_root : str or Path
        Root directory for the store (created if absent).
    """

    def __init__(self, store_root: str = "experiments/prediction_store"):
        self.store_root = Path(store_root)
        self.store_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, experiment_name: str, predictions_df: pd.DataFrame, overwrite: bool = False) -> Path:
        """
        Persist predictions for an experiment.

        Parameters
        ----------
        experiment_name : str
            Unique identifier for the experiment.
        predictions_df : pd.DataFrame
            Must contain at minimum: ``fold_idx``, ``predicted``.
            Recommended additional columns: ``actual``, ``timestamp``,
            ``model_name``, ``regime_label``.
        overwrite : bool, default=False
            If False, appends to existing parquet file.

        Returns
        -------
        Path
            Path to the parquet file written.
        """
        exp_dir = self.store_root / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = exp_dir / "predictions.parquet"
        if parquet_path.exists() and not overwrite:
            existing = pd.read_parquet(parquet_path)
            predictions_df = pd.concat([existing, predictions_df], ignore_index=False)

        predictions_df.to_parquet(parquet_path)
        # Save metadata
        meta = {
            "experiment_name": experiment_name,
            "columns": predictions_df.columns.tolist(),
            "n_rows": len(predictions_df),
        }
        (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return parquet_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, experiment_name: str) -> pd.DataFrame:
        """Load all stored predictions for an experiment."""
        parquet_path = self.store_root / experiment_name / "predictions.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No predictions found for experiment '{experiment_name}'. "
                f"Expected: {parquet_path}"
            )
        return pd.read_parquet(parquet_path)

    def list_experiments(self) -> list:
        """Return names of all stored experiments."""
        return [p.name for p in self.store_root.iterdir() if p.is_dir()]

    def summary(self) -> pd.DataFrame:
        """One-row-per-experiment summary DataFrame."""
        rows = []
        for name in self.list_experiments():
            meta_path = self.store_root / name / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                rows.append({"experiment": name, **meta})
        return pd.DataFrame(rows)