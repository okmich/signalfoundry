import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


class ObjectiveScore:

    def __init__(self, name: str, value: float, normalized_score: float, weight: float, achieved: bool = True):
        self.name = name
        self.value = value
        self.normalized_score = normalized_score
        self.weight = weight
        self.weighted_score = normalized_score * weight
        self.achieved = achieved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": float(self.value) if not np.isnan(self.value) else None,
            "normalized_score": float(self.normalized_score),
            "weight": float(self.weight),
            "weighted_score": float(self.weighted_score),
            "achieved": self.achieved,
        }


class ModelRanking:

    def __init__(self, model_name: str, composite_score: float, objective_scores: List[ObjectiveScore],
                 rank: Optional[int] = None):
        self.model_name = model_name
        self.composite_score = composite_score
        self.objective_scores = objective_scores
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "rank": self.rank,
            "composite_score": float(self.composite_score),
            "objectives": {
                score.name: score.to_dict() for score in self.objective_scores
            },
        }

    def __repr__(self):
        return f"ModelRanking(model={self.model_name}, rank={self.rank}, score={self.composite_score:.3f})"


class ObjectivesEngine:
    """
    Engine for scoring and ranking models based on objectives.

    Examples
    --------
    >>> objectives_config = [
    ...     {'name': 'regime_discriminability', 'target': 'maximize', 'weight': 0.35},
    ...     {'name': 'mean_duration', 'target': 'range', 'min': 5, 'max': 30, 'weight': 0.25},
    ... ]
    >>>
    >>> engine = ObjectivesEngine(objectives_config)
    >>> rankings = engine.rank_models(metrics_df)
    >>> best_model = rankings[0]
    """

    def __init__(self, objectives: List[Dict[str, Any]]):
        """
        Initialize objectives engine.

        Parameters
        ----------
        objectives : list of dict
            List of objective configurations. Each dict should have:
            - name: str - Metric name
            - target: str - 'maximize', 'minimize', 'range', or 'separation'
            - weight: float - Objective weight (should sum to 1.0)
            - Additional params based on target type:
              - For 'range': min, max
              - For 'maximize'/'minimize': threshold (optional)
              - For 'separation': calculation (optional)
        """
        self.objectives = objectives
        self._validate_objectives()

    def _validate_objectives(self):
        """Validate objectives configuration."""
        if not self.objectives:
            raise ValueError("At least one objective required")

        total_weight = sum(obj.get("weight", 0) for obj in self.objectives)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            print(f"Warning: Objective weights sum to {total_weight:.3f}, not 1.0")

        valid_targets = ["maximize", "minimize", "range", "separation"]
        for obj in self.objectives:
            if "name" not in obj:
                raise ValueError(f"Objective missing 'name': {obj}")
            if "target" not in obj:
                raise ValueError(f"Objective '{obj['name']}' missing 'target'")
            if obj["target"] not in valid_targets:
                raise ValueError(
                    f"Invalid target '{obj['target']}' for objective '{obj['name']}'. "
                    f"Must be one of: {valid_targets}"
                )
            if obj["target"] == "range":
                if "min" not in obj or "max" not in obj:
                    raise ValueError(
                        f"Range objective '{obj['name']}' must have 'min' and 'max'"
                    )

    def rank_models(self, metrics_summary: Dict[str, Dict[str, float]],
                    regime_stats_df: Optional[pd.DataFrame] = None) -> List[ModelRanking]:
        """
        Rank models based on objectives.

        Parameters
        ----------
        metrics_summary : dict
            Dictionary mapping model names to metrics
        regime_stats_df : pd.DataFrame, optional
            Regime statistics (for separation objectives)

        Returns
        -------
        list of ModelRanking
            Models ranked by composite score (best first)
        """
        print("=" * 80)
        print("MODEL RANKING")
        print("=" * 80)

        model_rankings = []

        for model_name, metrics in metrics_summary.items():
            print(f"\nEvaluating: {model_name}")

            objective_scores = []
            total_weighted_score = 0.0

            for obj_config in self.objectives:
                score = self._score_objective(
                    obj_config=obj_config,
                    metrics=metrics,
                    model_name=model_name,
                    regime_stats_df=regime_stats_df,
                )

                objective_scores.append(score)
                total_weighted_score += score.weighted_score

                status = "[OK]" if score.achieved else "[WARN]"
                print(
                    f"  {status} {score.name}: {score.value:.4f} "
                    f"(normalized: {score.normalized_score:.3f}, "
                    f"weighted: {score.weighted_score:.3f})"
                )

            composite_score = total_weighted_score
            print(f"  Composite Score: {composite_score:.4f}")

            ranking = ModelRanking(
                model_name=model_name,
                composite_score=composite_score,
                objective_scores=objective_scores,
            )
            model_rankings.append(ranking)

        # Sort by composite score (descending)
        model_rankings.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign ranks
        for rank, ranking in enumerate(model_rankings, start=1):
            ranking.rank = rank

        print("\n" + "=" * 80)
        print("RANKING COMPLETE")
        print("=" * 80)
        print(f"\nTop 3 Models:")
        for ranking in model_rankings[:3]:
            print(
                f"  {ranking.rank}. {ranking.model_name}: {ranking.composite_score:.4f}"
            )

        return model_rankings

    def _score_objective(self, obj_config: Dict[str, Any], metrics: Dict[str, float], model_name: str,
                         regime_stats_df: Optional[pd.DataFrame] = None) -> ObjectiveScore:
        """Score a single objective for a model."""
        obj_name = obj_config["name"]
        target = obj_config["target"]
        weight = obj_config.get("weight", 1.0)

        # Get metric value
        if target == "separation" and regime_stats_df is not None:
            value = self._compute_separation(obj_config, model_name, regime_stats_df)
        elif "calculation" in obj_config:
            raise ValueError(
                f"Objective '{obj_name}' specifies a 'calculation' expression "
                f"({obj_config['calculation']!r}), but expression-based objective "
                "evaluation is not yet implemented. Remove the 'calculation' key and "
                "use a direct metric name or a 'separation' target instead."
            )
        else:
            value = metrics.get(obj_name, np.nan)

        # Enforce scalar contract — dict/list metrics (e.g. regime_conditioned_sharpe)
        # cannot be used as objective values; treat as missing.
        if not isinstance(value, (int, float, np.floating, np.integer)):
            value = np.nan

        # Handle NaN
        if np.isnan(value):
            return ObjectiveScore(
                name=obj_name,
                value=value,
                normalized_score=0.0,
                weight=weight,
                achieved=False,
            )

        # Normalize score based on target type
        if target == "maximize":
            normalized_score, achieved = self._normalize_maximize(value, obj_config)
        elif target == "minimize":
            normalized_score, achieved = self._normalize_minimize(value, obj_config)
        elif target == "range":
            normalized_score, achieved = self._normalize_range(value, obj_config)
        elif target == "separation":
            normalized_score, achieved = self._normalize_maximize(value, obj_config)
        else:
            normalized_score = 0.0
            achieved = False

        return ObjectiveScore(
            name=obj_name,
            value=value,
            normalized_score=normalized_score,
            weight=weight,
            achieved=achieved)

    def _normalize_maximize(self, value: float, obj_config: Dict[str, Any]) -> tuple[float, bool]:
        """Normalize maximize objective."""
        threshold = obj_config.get("threshold", 0.0)

        # Achieved if above threshold
        achieved = value >= threshold

        # Normalize: score increases linearly from 0 at threshold
        if value <= threshold:
            normalized = max(0.0, value / threshold if threshold > 0 else 0.5)
        else:
            # Score = 0.5 at threshold, approaches 1.0 as value increases
            normalized = 0.5 + 0.5 * min(
                1.0, (value - threshold) / (1.0 - threshold if threshold < 1 else value)
            )
        return min(1.0, max(0.0, normalized)), achieved

    def _normalize_minimize(self, value: float, obj_config: Dict[str, Any]) -> tuple[float, bool]:
        """Normalize minimize objective."""
        threshold = obj_config.get("threshold")  # None when not configured

        if threshold is None:
            # No hard target: use a scale-free reciprocal mapping so that
            # smaller values score higher without a degenerate inf comparison.
            #   value=0   → 1.0  (best possible)
            #   value=1   → 0.5
            #   value→∞   → 0.0
            achieved = True  # no constraint to enforce
            normalized = 1.0 / (1.0 + max(0.0, value))
        else:
            # Achieved if below threshold
            achieved = value <= threshold

            # Normalize: score decreases as value increases
            if value >= threshold:
                normalized = max(0.0, 1.0 - (value - threshold) / threshold if threshold > 0 else 0.0)
            else:
                normalized = 0.5 + 0.5 * (1.0 - value / threshold if threshold > 0 else 0.5)
        return min(1.0, max(0.0, normalized)), achieved

    def _normalize_range(self, value: float, obj_config: Dict[str, Any]) -> tuple[float, bool]:
        """Normalize range objective."""
        min_val = obj_config["min"]
        max_val = obj_config["max"]

        # Achieved if within range
        achieved = min_val <= value <= max_val

        if value < min_val:
            # Below range: score decreases as distance increases
            distance = min_val - value
            normalized = max(0.0, 1.0 - distance / min_val if min_val > 0 else 0.0)
        elif value > max_val:
            # Above range: score decreases as distance increases
            distance = value - max_val
            normalized = max(0.0, 1.0 - distance / max_val if max_val > 0 else 0.0)
        else:
            # Within range: perfect score
            normalized = 1.0

        return normalized, achieved

    def _compute_separation(self, obj_config: Dict[str, Any], model_name: str, regime_stats_df: pd.DataFrame) -> float:
        base_metric = obj_config["name"].replace("_separation", "")
        calculation = obj_config.get("calculation", "range")
        model_stats = regime_stats_df[regime_stats_df["algo"] == model_name]
        if len(model_stats) == 0 or base_metric not in model_stats.columns:
            return 0.0

        values = model_stats[base_metric].dropna()
        if len(values) == 0:
            return 0.0

        if "max" in calculation and "min" in calculation:
            # Range-based separation
            return float(values.max() - values.min())
        elif "std" in calculation:
            # Standard deviation-based
            return float(values.std())
        elif "/" in calculation:
            # Ratio-based
            max_val = values.max()
            min_val = values.min()
            if min_val > 0:
                return float(max_val / min_val)
            return 0.0
        else:
            # Default: range
            return float(values.max() - values.min())

    def save_rankings(self, rankings: List[ModelRanking], output_path: str):
        """
        Save rankings to JSON file.

        Parameters
        ----------
        rankings : list of ModelRanking
            Model rankings
        output_path : str
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rankings_data = {
            "rankings": [r.to_dict() for r in rankings],
            "best_model": rankings[0].model_name if rankings else None,
            "objectives": self.objectives,
        }
        with open(output_path, "w") as f:
            json.dump(rankings_data, f, indent=2)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ObjectivesEngine":
        """
        Create engine from configuration.

        Parameters
        ----------
        config : dict
            Configuration with 'objectives' or 'primary' key

        Returns
        -------
        ObjectivesEngine
        """
        if "objectives" in config:
            objectives = config["objectives"]
        elif "primary" in config:
            objectives = config["primary"]
        else:
            raise ValueError("Config must have 'objectives' or 'primary' key")

        # Handle both list and dict formats
        if isinstance(objectives, dict):
            objectives = objectives.get("primary", [])
        return cls(objectives=objectives)
