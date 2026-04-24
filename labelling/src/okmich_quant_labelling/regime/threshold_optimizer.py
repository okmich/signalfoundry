from __future__ import annotations

import json
import math
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import ks_2samp, kruskal
from sklearn.cluster import KMeans


class MarketPropertyType(StrEnum):
    DIRECTION = "direction"
    MOMENTUM = "momentum"
    DIRECTIONLESS_MOMENTUM = "directionless_momentum"
    VOLATILITY = "volatility"
    PATH_STRUCTURE = "path_structure"
    LIQUIDITY = "liquidity"


class ThresholdMethod(StrEnum):
    QUANTILE = "quantile"
    KMEANS_1D = "kmeans_1d"
    SUPERVISED_GRID = "supervised_grid"
    SUPERVISED_DE = "supervised_de"


class ClipMethod(StrEnum):
    NONE = "none"
    PERCENTILE = "percentile"
    IQR = "iqr"
    MAD_ZSCORE = "mad_zscore"


class ObjectiveType(StrEnum):
    EDGE = "edge"
    SEPARATION = "separation"


def _series(values: pd.Series | np.ndarray | list[float], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(np.asarray(values, dtype=float), name=name)


def _align_to_index(
        values: pd.Series | np.ndarray | list[float],
        name: str,
        index: pd.Index,
) -> pd.Series:
    target_index = pd.Index(index)
    if isinstance(values, pd.Series):
        series = values.astype(float)
        if len(series) != len(target_index):
            raise ValueError(
                f"{name} length ({len(series)}) must match feature length ({len(target_index)})."
            )
        if series.index.equals(target_index):
            return series
        return pd.Series(series.to_numpy(dtype=float), index=target_index, name=name)

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape={array.shape}.")
    if len(array) != len(target_index):
        raise ValueError(
            f"{name} length ({len(array)}) must match feature length ({len(target_index)})."
        )
    return pd.Series(array, index=target_index, name=name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value


class DirectFeatureThresholdOptimizer:
    """
    Direct univariate feature threshold optimizer.

    Train on one observed feature and produce class thresholds for live inference.
    This optimizer does not require a price series or metric function wrapper.
    """

    leaks_future: bool = False

    def __init__(self, n_classes: int = 3, market_property_type: MarketPropertyType = MarketPropertyType.DIRECTION,
                 threshold_method: ThresholdMethod = ThresholdMethod.QUANTILE, clip_method: ClipMethod = ClipMethod.PERCENTILE,
                 objective_type: ObjectiveType = ObjectiveType.EDGE, min_class_support: int = 30,
                 max_class_imbalance: float = 15.0, min_class_entropy: float = 0.20, hysteresis: float = 0.005,
                 min_persistence: int = 2, clip_lower_pct: float = 0.005, clip_upper_pct: float = 0.995,
                 iqr_multiplier: float = 3.0, mad_zscore: float = 6.0, clip_before_threshold_fit: bool = True,
                 clip_on_predict: bool = True, supervised_grid_size: int = 25, max_supervised_combinations: int = 50000,
                 de_max_iter: int = 120, de_popsize: int = 20, de_min_quantile_gap: float = 0.02, turnover_penalty: float = 0.0,
                 whipsaw_penalty: float = 0.0, psi_bins: int = 20, drift_psi_alert: float = 0.20,
                 drift_psi_refit: float = 0.30, drift_ks_alpha: float = 0.01, drift_clip_rate_multiplier: float = 2.5, random_state: int = 42):
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if min_class_support < 1:
            raise ValueError(f"min_class_support must be >= 1, got {min_class_support}")
        if max_class_imbalance < 1.0:
            raise ValueError(f"max_class_imbalance must be >= 1.0, got {max_class_imbalance}")
        if not 0.0 <= min_class_entropy <= 1.0:
            raise ValueError(f"min_class_entropy must be in [0,1], got {min_class_entropy}")
        if min_persistence < 1:
            raise ValueError(f"min_persistence must be >= 1, got {min_persistence}")
        if hysteresis < 0.0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        if not 0.0 < clip_lower_pct < clip_upper_pct < 1.0:
            raise ValueError(
                f"clip percentiles must satisfy 0 < lower < upper < 1, got lower={clip_lower_pct}, upper={clip_upper_pct}"
            )
        if supervised_grid_size < 5:
            raise ValueError(f"supervised_grid_size must be >= 5, got {supervised_grid_size}")
        if de_min_quantile_gap <= 0.0:
            raise ValueError(f"de_min_quantile_gap must be > 0, got {de_min_quantile_gap}")
        if psi_bins < 5:
            raise ValueError(f"psi_bins must be >= 5, got {psi_bins}")
        if not 0.0 < drift_ks_alpha < 1.0:
            raise ValueError(f"drift_ks_alpha must be in (0,1), got {drift_ks_alpha}")
        if drift_clip_rate_multiplier < 1.0:
            raise ValueError(f"drift_clip_rate_multiplier must be >= 1, got {drift_clip_rate_multiplier}")

        self.market_property_type = MarketPropertyType(market_property_type)
        self.n_classes = n_classes
        self.threshold_method = ThresholdMethod(threshold_method)
        self.clip_method = ClipMethod(clip_method)
        self.objective_type = ObjectiveType(objective_type)
        self.min_class_support = min_class_support
        self.max_class_imbalance = max_class_imbalance
        self.min_class_entropy = min_class_entropy
        self.hysteresis = hysteresis
        self.min_persistence = min_persistence
        self.clip_lower_pct = clip_lower_pct
        self.clip_upper_pct = clip_upper_pct
        self.iqr_multiplier = iqr_multiplier
        self.mad_zscore = mad_zscore
        self.clip_before_threshold_fit = clip_before_threshold_fit
        self.clip_on_predict = clip_on_predict
        self.supervised_grid_size = supervised_grid_size
        self.max_supervised_combinations = max_supervised_combinations
        self.de_max_iter = de_max_iter
        self.de_popsize = de_popsize
        self.de_min_quantile_gap = de_min_quantile_gap
        self.turnover_penalty = turnover_penalty
        self.whipsaw_penalty = whipsaw_penalty
        self.psi_bins = psi_bins
        self.drift_psi_alert = drift_psi_alert
        self.drift_psi_refit = drift_psi_refit
        self.drift_ks_alpha = drift_ks_alpha
        self.drift_clip_rate_multiplier = drift_clip_rate_multiplier
        self.random_state = random_state

        supervised_methods = {ThresholdMethod.SUPERVISED_GRID, ThresholdMethod.SUPERVISED_DE}
        directional_types = {MarketPropertyType.DIRECTION, MarketPropertyType.MOMENTUM}
        if self.threshold_method in supervised_methods and self.market_property_type in directional_types and self.objective_type != ObjectiveType.EDGE:
            raise ValueError(
                "For DIRECTION and MOMENTUM, supervised threshold methods must optimize ObjectiveType.EDGE "
                "on forward returns."
            )

        self.is_fitted_: bool = False
        self.thresholds_: Optional[np.ndarray] = None
        self.clip_bounds_: tuple[float, float] = (float("-inf"), float("inf"))
        self.class_signals_: Optional[np.ndarray] = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.training_reference_: dict[str, Any] = {}

    @property
    def n_thresholds(self) -> int:
        return self.n_classes - 1

    def _reset_fitted_state(self) -> None:
        self.is_fitted_ = False
        self.thresholds_ = None
        self.clip_bounds_ = (float("-inf"), float("inf"))
        self.class_signals_ = None
        self.fit_diagnostics_ = {}
        self.training_reference_ = {}

    def fit(self, feature: pd.Series | np.ndarray | list[float],
            forward_returns: Optional[pd.Series | np.ndarray | list[float]] = None) -> "DirectFeatureThresholdOptimizer":
        self._reset_fitted_state()
        feat = _series(feature, "feature")
        if len(feat) < self.n_classes * self.min_class_support:
            raise ValueError(
                f"Insufficient data length ({len(feat)}) for n_classes={self.n_classes} and "
                f"min_class_support={self.min_class_support}."
            )

        fwd = None
        if forward_returns is not None:
            fwd = _align_to_index(forward_returns, "forward_return", feat.index)

        try:
            values = feat.to_numpy(dtype=float)
            self.clip_bounds_ = self._fit_clip_bounds(values)

            fit_values = self._apply_clipping(values, apply_clip=self.clip_before_threshold_fit)[0]
            valid_fit = fit_values[np.isfinite(fit_values)]
            if len(valid_fit) < self.n_classes * self.min_class_support:
                raise ValueError(
                    f"Too few finite values after clipping ({len(valid_fit)}). "
                    f"Need at least {self.n_classes * self.min_class_support}."
                )

            thresholds = self._fit_thresholds(fit_values, fwd)
            self._validate_thresholds(thresholds)
            self.thresholds_ = thresholds
            self.class_signals_ = np.linspace(-1.0, 1.0, self.n_classes)
            self.is_fitted_ = True

            labels, diag = self.predict(feat, return_diagnostics=True)
            class_stats = self._compute_class_stats(labels)
            self._validate_class_constraints(class_stats)

            clip_rate = float((diag["is_clipped_low"] | diag["is_clipped_high"]).mean())
            self.training_reference_ = self._build_training_reference(values=values, labels=labels, clip_rate=clip_rate)
            self.fit_diagnostics_ = {
                "n_samples": int(len(feat)),
                "n_finite_feature": int(np.isfinite(values).sum()),
                "clip_bounds": [float(self.clip_bounds_[0]), float(self.clip_bounds_[1])],
                "clip_rate": clip_rate,
                "thresholds": thresholds.tolist(),
                "class_counts": class_stats["counts"],
                "class_imbalance_ratio": class_stats["imbalance_ratio"],
                "class_entropy": class_stats["entropy"],
                "market_property_type": self.market_property_type.value,
            }
            return self
        except Exception:
            self._reset_fitted_state()
            raise

    def predict(self, feature: pd.Series | np.ndarray | list[float],
                return_diagnostics: bool = False) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
        self._assert_fitted()
        feat = _series(feature, "feature")
        values = feat.to_numpy(dtype=float)
        clipped_values, clipped_low, clipped_high = self._apply_clipping(values, apply_clip=self.clip_on_predict)

        raw_labels = self._assign_labels(clipped_values, self.thresholds_)
        smooth_labels, candidate_state, consecutive_count = self._apply_state_machine(
            raw_labels,
            clipped_values,
            thresholds=self.thresholds_,
        )

        label_series = pd.Series(smooth_labels, index=feat.index, name="label").astype("Float64").astype("Int16")
        if not return_diagnostics:
            return label_series

        diagnostics = pd.DataFrame(
            {
                "feature_value": values,
                "feature_clipped": clipped_values,
                "is_clipped_low": clipped_low,
                "is_clipped_high": clipped_high,
                "raw_label": raw_labels,
                "candidate_state": candidate_state,
                "consecutive_count": consecutive_count,
                "label": smooth_labels,
            },
            index=feat.index,
        )
        return label_series, diagnostics

    def fit_predict(self, feature: pd.Series | np.ndarray | list[float],
                    forward_returns: Optional[pd.Series | np.ndarray | list[float]] = None,
                    return_diagnostics: bool = False) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
        self.fit(feature=feature, forward_returns=forward_returns)
        return self.predict(feature=feature, return_diagnostics=return_diagnostics)

    def evaluate_forward_blocks(self, feature: pd.Series | np.ndarray | list[float],
                                forward_returns: Optional[pd.Series | np.ndarray | list[float]],
                                block_size: int = 1000) -> pd.DataFrame:
        if block_size < 20:
            raise ValueError(f"block_size must be >= 20, got {block_size}")

        labels, diag = self.predict(feature, return_diagnostics=True)
        frame = pd.DataFrame({"label": labels}, index=labels.index).join(diag[["is_clipped_low", "is_clipped_high"]])
        if forward_returns is not None:
            frame["forward_return"] = _align_to_index(forward_returns, "forward_return", labels.index)
        frame["block_id"] = np.arange(len(frame)) // block_size

        directional_types = {MarketPropertyType.DIRECTION, MarketPropertyType.MOMENTUM}
        is_directional = self.market_property_type in directional_types

        rows: list[dict[str, Any]] = []
        for block_id, blk in frame.groupby("block_id", sort=True):
            labels_blk = blk["label"].dropna()
            turnover = self._label_turnover(labels_blk)
            mean_dwell = self._mean_dwell(labels_blk)
            clip_rate = float((blk["is_clipped_low"] | blk["is_clipped_high"]).mean())

            row: dict[str, Any] = {
                "block_id": int(block_id),
                "n_samples": int(len(blk)),
                "coverage": float(labels_blk.shape[0] / len(blk)) if len(blk) > 0 else np.nan,
                "turnover": turnover,
                "mean_dwell": mean_dwell,
                "clip_rate": clip_rate,
            }
            counts = labels_blk.value_counts().to_dict()
            for k in range(self.n_classes):
                row[f"class_{k}_count"] = int(counts.get(k, 0))

            # Edge/hit-rate rely on signals = linspace(-1, 1, n_classes), which maps ordered
            # classes to directional intensity. That mapping is meaningless for volatility,
            # liquidity, or path-structure regimes, so we only emit edge columns when
            # market_property_type is directional.
            if is_directional and "forward_return" in blk.columns:
                valid = blk["label"].notna() & blk["forward_return"].notna()
                if valid.any():
                    labels_arr = blk.loc[valid, "label"].to_numpy(dtype=int)
                    returns_arr = blk.loc[valid, "forward_return"].to_numpy(dtype=float)
                    signals = self.class_signals_[labels_arr]
                    edge = float(np.mean(signals * returns_arr))
                    nz = signals != 0
                    hit_rate = float(np.mean(np.sign(returns_arr[nz]) == np.sign(signals[nz]))) if nz.any() else np.nan
                    row["edge"] = edge
                    row["edge_bps"] = edge * 10000.0
                    row["hit_rate"] = hit_rate
                else:
                    row["edge"] = np.nan
                    row["edge_bps"] = np.nan
                    row["hit_rate"] = np.nan

            rows.append(row)
        return pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)

    def evaluate_drift(self, feature: pd.Series | np.ndarray | list[float]) -> dict[str, Any]:
        """
        Compare current feature distribution versus training distribution.

        Returns a dictionary with PSI/KS/clip-rate/class-occupancy drift diagnostics and refit trigger flags.
        """
        self._assert_fitted()
        if not self.training_reference_:
            raise RuntimeError("Training reference is unavailable. Re-fit the optimizer.")

        feat = _series(feature, "feature")
        values = feat.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise ValueError("Current feature has no finite values.")

        _, diag = self.predict(feat, return_diagnostics=True)
        current_clip_rate = float((diag["is_clipped_low"] | diag["is_clipped_high"]).mean())
        current_class_probs = self._class_probabilities(diag["label"])
        ref_class_probs = np.asarray(self.training_reference_["class_probabilities"], dtype=float)
        class_prob_l1 = float(np.sum(np.abs(current_class_probs - ref_class_probs)))

        ref_edges = np.asarray(self.training_reference_["psi_bin_edges"], dtype=float)
        ref_hist = np.asarray(self.training_reference_["psi_base_probs"], dtype=float)
        cur_hist = self._hist_probs(finite, ref_edges)
        psi_value = self._population_stability_index(ref_hist, cur_hist)

        ref_sample = np.asarray(self.training_reference_["sample_values"], dtype=float)
        ks_stat, ks_pvalue = ks_2samp(ref_sample, finite)
        ks_stat = float(ks_stat)
        ks_pvalue = float(ks_pvalue)

        ref_clip_rate = float(self.training_reference_["clip_rate"])
        clip_threshold = max(ref_clip_rate * self.drift_clip_rate_multiplier, ref_clip_rate + 0.05)
        clip_rate_trigger = current_clip_rate > clip_threshold
        psi_alert_trigger = psi_value >= self.drift_psi_alert
        psi_refit_trigger = psi_value >= self.drift_psi_refit
        ks_trigger = ks_pvalue < self.drift_ks_alpha
        occupancy_trigger = class_prob_l1 > 0.35

        alert_trigger = psi_alert_trigger or ks_trigger or clip_rate_trigger or occupancy_trigger
        refit_trigger = psi_refit_trigger or (psi_alert_trigger and occupancy_trigger) or (ks_trigger and clip_rate_trigger)

        return {
            "psi": psi_value,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "current_clip_rate": current_clip_rate,
            "reference_clip_rate": ref_clip_rate,
            "class_prob_l1_shift": class_prob_l1,
            "alert_trigger": alert_trigger,
            "refit_trigger": refit_trigger,
            "trigger_breakdown": {
                "psi_alert_trigger": psi_alert_trigger,
                "psi_refit_trigger": psi_refit_trigger,
                "ks_trigger": ks_trigger,
                "clip_rate_trigger": clip_rate_trigger,
                "occupancy_trigger": occupancy_trigger,
            },
        }

    def evaluate_acceptance(
            self,
            feature: pd.Series | np.ndarray | list[float],
            forward_returns: Optional[pd.Series | np.ndarray | list[float]],
            block_size: int = 1000,
            n_perm: int = 200,
            gates: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate pass/fail acceptance gates with defaults driven by MarketPropertyType.
        """
        self._assert_fitted()
        blocks = self.evaluate_forward_blocks(feature=feature, forward_returns=forward_returns, block_size=block_size)
        labels = self.predict(feature)
        class_stats = self._compute_class_stats(labels)

        summary: dict[str, float] = {
            "coverage_mean": float(blocks["coverage"].mean()),
            "turnover_mean": float(blocks["turnover"].mean()),
            "mean_dwell_mean": float(blocks["mean_dwell"].mean()),
            "clip_rate_mean": float(blocks["clip_rate"].mean()),
            "class_entropy": float(class_stats["entropy"]),
            "class_imbalance_ratio": float(class_stats["imbalance_ratio"]),
            "min_class_count": float(min(class_stats["counts"].values())),
        }

        applied = self._default_acceptance_gates()
        if gates is not None:
            applied.update(gates)

        violated: list[str] = []
        if not np.isfinite(summary["coverage_mean"]):
            violated.append("coverage_mean_non_finite")
        elif summary["coverage_mean"] < applied["min_coverage"]:
            violated.append(f"coverage_mean<{applied['min_coverage']}")
        if not np.isfinite(summary["turnover_mean"]):
            violated.append("turnover_mean_non_finite")
        elif summary["turnover_mean"] > applied["max_turnover"]:
            violated.append(f"turnover_mean>{applied['max_turnover']}")
        if not np.isfinite(summary["mean_dwell_mean"]):
            violated.append("mean_dwell_mean_non_finite")
        elif summary["mean_dwell_mean"] < applied["min_mean_dwell"]:
            violated.append(f"mean_dwell_mean<{applied['min_mean_dwell']}")
        if not np.isfinite(summary["clip_rate_mean"]):
            violated.append("clip_rate_mean_non_finite")
        elif summary["clip_rate_mean"] > applied["max_clip_rate"]:
            violated.append(f"clip_rate_mean>{applied['max_clip_rate']}")
        if not np.isfinite(summary["class_entropy"]):
            violated.append("class_entropy_non_finite")
        elif summary["class_entropy"] < applied["min_class_entropy"]:
            violated.append(f"class_entropy<{applied['min_class_entropy']}")
        if not np.isfinite(summary["class_imbalance_ratio"]):
            violated.append("class_imbalance_ratio_non_finite")
        elif summary["class_imbalance_ratio"] > applied["max_class_imbalance"]:
            violated.append(f"class_imbalance_ratio>{applied['max_class_imbalance']}")
        if not np.isfinite(summary["min_class_count"]):
            violated.append("min_class_count_non_finite")
        elif summary["min_class_count"] < applied["min_class_support"]:
            violated.append(f"min_class_count<{applied['min_class_support']}")

        directional_types = {MarketPropertyType.DIRECTION, MarketPropertyType.MOMENTUM}
        if self.market_property_type in directional_types:
            if forward_returns is None:
                raise ValueError("forward_returns are required for acceptance evaluation of directional properties.")
            edge_metrics = self._directional_acceptance_metrics(
                feature=feature,
                forward_returns=forward_returns,
                n_perm=n_perm,
                blocks=blocks,
            )
            summary.update(edge_metrics)
            if not np.isfinite(summary["edge"]):
                violated.append("edge_non_finite")
            elif summary["edge"] < applied["min_edge"]:
                violated.append(f"edge<{applied['min_edge']}")
            if not np.isfinite(summary["hit_rate"]):
                violated.append("hit_rate_non_finite")
            elif summary["hit_rate"] < applied["min_hit_rate"]:
                violated.append(f"hit_rate<{applied['min_hit_rate']}")
            if not np.isfinite(summary["perm_pvalue"]):
                violated.append("perm_pvalue_non_finite")
            elif summary["perm_pvalue"] > applied["max_perm_pvalue"]:
                violated.append(f"perm_pvalue>{applied['max_perm_pvalue']}")
            if not np.isfinite(summary["positive_edge_block_fraction"]):
                violated.append("positive_edge_block_fraction_non_finite")
            elif summary["positive_edge_block_fraction"] < applied["min_positive_edge_block_fraction"]:
                violated.append(f"positive_edge_block_fraction<{applied['min_positive_edge_block_fraction']}")
        else:
            sep_metrics = self._non_directional_acceptance_metrics(feature=feature, forward_returns=forward_returns)
            summary.update(sep_metrics)
            if not np.isfinite(summary["separation_score"]):
                violated.append("separation_score_non_finite")
            elif summary["separation_score"] < applied["min_separation_score"]:
                violated.append(f"separation_score<{applied['min_separation_score']}")
            if not np.isfinite(summary["kruskal_pvalue"]):
                violated.append("kruskal_pvalue_non_finite")
            elif summary["kruskal_pvalue"] > applied["max_kruskal_pvalue"]:
                violated.append(f"kruskal_pvalue>{applied['max_kruskal_pvalue']}")

        return {
            "pass": len(violated) == 0,
            "violated_gates": violated,
            "gates": applied,
            "summary": summary,
            "block_metrics": blocks,
        }

    def save(self, path: str | Path) -> None:
        self._assert_fitted()
        payload = {
            "version": 1,
            "config": {
                "n_classes": self.n_classes,
                "market_property_type": self.market_property_type.value,
                "threshold_method": self.threshold_method.value,
                "clip_method": self.clip_method.value,
                "objective_type": self.objective_type.value,
                "min_class_support": self.min_class_support,
                "max_class_imbalance": self.max_class_imbalance,
                "min_class_entropy": self.min_class_entropy,
                "hysteresis": self.hysteresis,
                "min_persistence": self.min_persistence,
                "clip_lower_pct": self.clip_lower_pct,
                "clip_upper_pct": self.clip_upper_pct,
                "iqr_multiplier": self.iqr_multiplier,
                "mad_zscore": self.mad_zscore,
                "clip_before_threshold_fit": self.clip_before_threshold_fit,
                "clip_on_predict": self.clip_on_predict,
                "supervised_grid_size": self.supervised_grid_size,
                "max_supervised_combinations": self.max_supervised_combinations,
                "de_max_iter": self.de_max_iter,
                "de_popsize": self.de_popsize,
                "de_min_quantile_gap": self.de_min_quantile_gap,
                "turnover_penalty": self.turnover_penalty,
                "whipsaw_penalty": self.whipsaw_penalty,
                "psi_bins": self.psi_bins,
                "drift_psi_alert": self.drift_psi_alert,
                "drift_psi_refit": self.drift_psi_refit,
                "drift_ks_alpha": self.drift_ks_alpha,
                "drift_clip_rate_multiplier": self.drift_clip_rate_multiplier,
                "random_state": self.random_state,
            },
            "state": {
                "is_fitted": self.is_fitted_,
                "thresholds": self.thresholds_.tolist(),
                "clip_bounds": [float(self.clip_bounds_[0]), float(self.clip_bounds_[1])],
                "class_signals": self.class_signals_.tolist(),
                "fit_diagnostics": _json_safe(self.fit_diagnostics_),
                "training_reference": _json_safe(self.training_reference_),
            },
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DirectFeatureThresholdOptimizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(**payload["config"])
        state = payload["state"]
        obj.is_fitted_ = bool(state["is_fitted"])
        obj.thresholds_ = np.asarray(state["thresholds"], dtype=float)
        obj.clip_bounds_ = (float(state["clip_bounds"][0]), float(state["clip_bounds"][1]))
        obj.class_signals_ = np.asarray(state["class_signals"], dtype=float)
        obj.fit_diagnostics_ = state.get("fit_diagnostics", {})
        obj.training_reference_ = state.get("training_reference", {})
        return obj

    def _assert_fitted(self) -> None:
        if not self.is_fitted_ or self.thresholds_ is None or self.class_signals_ is None:
            raise RuntimeError("Optimizer is not fitted. Call fit() before predict/evaluate/save.")

    def _fit_clip_bounds(self, values: np.ndarray) -> tuple[float, float]:
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            raise ValueError("Feature has no finite values.")

        if self.clip_method == ClipMethod.NONE:
            return float("-inf"), float("inf")
        if self.clip_method == ClipMethod.PERCENTILE:
            lo = float(np.nanquantile(valid, self.clip_lower_pct))
            hi = float(np.nanquantile(valid, self.clip_upper_pct))
            return lo, hi
        if self.clip_method == ClipMethod.IQR:
            q1 = float(np.nanquantile(valid, 0.25))
            q3 = float(np.nanquantile(valid, 0.75))
            iqr = q3 - q1
            return q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr
        if self.clip_method == ClipMethod.MAD_ZSCORE:
            med = float(np.nanmedian(valid))
            mad = float(np.nanmedian(np.abs(valid - med)))
            scaled = 1.4826 * mad
            if scaled < 1e-12:
                lo = float(np.nanquantile(valid, self.clip_lower_pct))
                hi = float(np.nanquantile(valid, self.clip_upper_pct))
                return lo, hi
            return med - self.mad_zscore * scaled, med + self.mad_zscore * scaled
        raise RuntimeError(f"Unsupported clip method: {self.clip_method}")  # pragma: no cover

    def _apply_clipping(self, values: np.ndarray, apply_clip: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = values.copy()
        if not apply_clip:
            no = np.zeros(len(values), dtype=bool)
            return out, no, no

        lower, upper = self.clip_bounds_
        finite = np.isfinite(out)
        clipped_low = finite & (out < lower)
        clipped_high = finite & (out > upper)
        out[clipped_low] = lower
        out[clipped_high] = upper
        return out, clipped_low, clipped_high

    def _fit_thresholds(self, fit_values: np.ndarray, fwd: Optional[pd.Series]) -> np.ndarray:
        if self.threshold_method == ThresholdMethod.QUANTILE:
            return self._fit_thresholds_quantile(fit_values)
        if self.threshold_method == ThresholdMethod.KMEANS_1D:
            return self._fit_thresholds_kmeans(fit_values)
        if self.threshold_method == ThresholdMethod.SUPERVISED_GRID:
            return self._fit_thresholds_supervised_grid(fit_values, fwd)
        if self.threshold_method == ThresholdMethod.SUPERVISED_DE:
            return self._fit_thresholds_supervised_de(fit_values, fwd)
        raise RuntimeError(f"Unsupported threshold method: {self.threshold_method}")  # pragma: no cover

    def _fit_thresholds_quantile(self, values: np.ndarray) -> np.ndarray:
        valid = values[np.isfinite(values)]
        probs = np.linspace(1.0 / self.n_classes, (self.n_classes - 1) / self.n_classes, self.n_classes - 1)
        return np.quantile(valid, probs).astype(float)

    def _fit_thresholds_kmeans(self, values: np.ndarray) -> np.ndarray:
        valid = values[np.isfinite(values)]
        km = KMeans(n_clusters=self.n_classes, random_state=self.random_state, n_init=10)
        km.fit(valid.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        return ((centers[:-1] + centers[1:]) / 2.0).astype(float)

    def _fit_thresholds_supervised_grid(self, values: np.ndarray, fwd: Optional[pd.Series]) -> np.ndarray:
        if fwd is None:
            raise ValueError("forward_returns are required for supervised threshold methods.")
        valid_values = values[np.isfinite(values)]
        candidate_probs = np.linspace(0.01, 0.99, self.supervised_grid_size)
        candidate_thresholds = np.unique(np.quantile(valid_values, candidate_probs))
        if len(candidate_thresholds) < self.n_thresholds:
            raise ValueError("Too few unique candidate thresholds for supervised grid search.")

        n_candidates = len(candidate_thresholds)
        total_combinations = math.comb(n_candidates, self.n_thresholds)
        if total_combinations <= self.max_supervised_combinations:
            combos_iter = combinations(range(n_candidates), self.n_thresholds)
        else:
            rng = np.random.default_rng(self.random_state)
            sampled = self._sample_unique_combinations(
                n_items=n_candidates,
                k=self.n_thresholds,
                n_samples=self.max_supervised_combinations,
                rng=rng,
            )
            combos_iter = sampled

        best_score = -np.inf
        best: Optional[np.ndarray] = None
        for combo in combos_iter:
            thresholds = candidate_thresholds[list(combo)]
            score = self._score_thresholds(values, thresholds, fwd)
            if score > best_score:
                best_score = score
                best = thresholds.astype(float)

        if best is None:
            raise ValueError("Could not find a feasible supervised-grid threshold set.")
        return best

    def _fit_thresholds_supervised_de(self, values: np.ndarray, fwd: Optional[pd.Series]) -> np.ndarray:
        if fwd is None:
            raise ValueError("forward_returns are required for supervised threshold methods.")
        valid_values = values[np.isfinite(values)]

        def objective(params: np.ndarray) -> float:
            probs = np.sort(params)
            if np.any(np.diff(probs) < self.de_min_quantile_gap):
                return 1e6
            thresholds = np.quantile(valid_values, probs)
            score = self._score_thresholds(values, thresholds, fwd)
            if not np.isfinite(score):
                return 1e6
            return -score

        bounds = [(0.01, 0.99) for _ in range(self.n_thresholds)]
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=self.random_state,
            maxiter=self.de_max_iter,
            popsize=self.de_popsize,
            polish=True,
            updating="deferred",
        )
        probs = np.sort(result.x)
        return np.quantile(valid_values, probs).astype(float)

    def _score_thresholds(self, values: np.ndarray, thresholds: np.ndarray, fwd: pd.Series) -> float:
        raw_labels = self._assign_labels(values, thresholds)
        labels, _, _ = self._apply_state_machine(raw_labels, values, thresholds=thresholds)
        label_series = pd.Series(labels)
        stats = self._compute_class_stats(label_series)
        if not self._is_class_constraints_ok(stats):
            return -np.inf

        clean_labels = label_series.dropna()
        turns = self._label_turnover(clean_labels)
        whipsaw = self._whipsaw_rate(clean_labels)

        if self.objective_type == ObjectiveType.EDGE:
            ret = fwd.to_numpy(dtype=float)
            valid = np.isfinite(labels) & np.isfinite(ret)
            if valid.sum() < max(50, self.n_classes * self.min_class_support):
                return -np.inf
            signals = np.linspace(-1.0, 1.0, self.n_classes)[labels[valid].astype(int)]
            edge = float(np.mean(signals * ret[valid]))
            return edge - self.turnover_penalty * turns - self.whipsaw_penalty * whipsaw

        if self.objective_type == ObjectiveType.SEPARATION:
            # Supervised fit paths require fwd (checked in _fit_thresholds_supervised_*),
            # so this branch is only reachable with a non-None fwd.
            target = fwd.to_numpy(dtype=float)
            valid = np.isfinite(labels) & np.isfinite(target)
            if valid.sum() < max(50, self.n_classes * self.min_class_support):
                return -np.inf
            separation = self._separation_score(target[valid], labels[valid].astype(int))
            return separation - self.turnover_penalty * turns - self.whipsaw_penalty * whipsaw

        raise RuntimeError(f"Unsupported objective type: {self.objective_type}")  # pragma: no cover

    @staticmethod
    def _separation_score(values: np.ndarray, labels: np.ndarray) -> float:
        if len(values) == 0:
            return -np.inf
        overall_mean = float(np.mean(values))
        between = 0.0
        within = 0.0
        n = float(len(values))
        for cls in np.unique(labels):
            group = values[labels == cls]
            if len(group) == 0:
                continue
            group_mean = float(np.mean(group))
            between += len(group) * ((group_mean - overall_mean) ** 2)
            within += float(np.sum((group - group_mean) ** 2))
        if within <= 0.0:
            return -np.inf
        return (between / n) / (within / n + 1e-12)

    def _validate_thresholds(self, thresholds: np.ndarray) -> None:
        if thresholds.shape[0] != self.n_thresholds:
            raise ValueError(
                f"Expected {self.n_thresholds} thresholds for n_classes={self.n_classes}, "
                f"got {thresholds.shape[0]}"
            )
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("Thresholds contain non-finite values.")
        if np.any(np.diff(thresholds) <= 0.0):
            raise ValueError(
                "Thresholds are not strictly increasing. Feature may be near-constant or clipping too aggressive."
            )

    @staticmethod
    def _assign_labels(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        labels = np.full(len(values), np.nan)
        finite = np.isfinite(values)
        labels[finite] = np.searchsorted(thresholds, values[finite], side="right").astype(float)
        return labels

    @staticmethod
    def _sample_unique_combinations(n_items: int, k: int, n_samples: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
        sampled: set[tuple[int, ...]] = set()
        max_attempts = max(1000, n_samples * 20)
        attempts = 0

        while len(sampled) < n_samples and attempts < max_attempts:
            combo = tuple(sorted(rng.choice(n_items, size=k, replace=False).tolist()))
            sampled.add(combo)
            attempts += 1

        if len(sampled) < n_samples:
            for combo in combinations(range(n_items), k):
                sampled.add(combo)
                if len(sampled) >= n_samples:
                    break

        return sorted(sampled)

    def _apply_state_machine(
            self,
            raw_labels: np.ndarray,
            clipped_values: np.ndarray,
            thresholds: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(raw_labels)
        labels = np.full(n, np.nan)
        candidate_states = np.full(n, np.nan)
        consecutive = np.zeros(n, dtype=int)

        current_state = np.nan
        candidate_state = np.nan
        count = 0

        for i in range(n):
            raw = raw_labels[i]
            x = clipped_values[i]
            if not np.isfinite(raw) or not np.isfinite(x):
                # Preserve last confirmed regime (current_state) across gaps but
                # reset candidate tracking — otherwise pre-gap consecutive counts
                # can complete a transition on the first post-gap tick, which is
                # not what "min_persistence" should mean across a data hole.
                candidate_state = np.nan
                count = 0
                continue
            raw_int = int(raw)

            cand = raw_int
            if np.isfinite(current_state) and cand != int(current_state):
                if self.hysteresis > 0.0 and not self._passes_hysteresis(
                        value=x,
                        current=int(current_state),
                        candidate=cand,
                        thresholds=thresholds,
                ):
                    cand = int(current_state)

            if not np.isfinite(current_state):
                current_state = float(cand)
                candidate_state = float(cand)
                count = 1
            elif cand != int(current_state):
                if not np.isfinite(candidate_state) or cand != int(candidate_state):
                    candidate_state = float(cand)
                    count = 1
                else:
                    count += 1
                if count >= self.min_persistence:
                    current_state = float(cand)
            else:
                candidate_state = float(cand)
                count = 1

            candidate_states[i] = candidate_state
            consecutive[i] = count
            labels[i] = current_state
        return labels, candidate_states, consecutive

    def _passes_hysteresis(
            self,
            value: float,
            current: int,
            candidate: int,
            thresholds: Optional[np.ndarray] = None,
    ) -> bool:
        active_thresholds = self.thresholds_ if thresholds is None else thresholds
        if candidate == current or active_thresholds is None or len(active_thresholds) == 0:
            return True
        # For skip transitions (|candidate - current| > 1), gate on the boundary
        # adjacent to the TARGET class, not to the current class. That is the
        # strictest band among the boundaries being crossed, so if it clears
        # all lower boundaries clear too (assuming hysteresis < threshold gaps).
        if candidate > current:
            boundary_idx = min(candidate - 1, len(active_thresholds) - 1)
            boundary = active_thresholds[boundary_idx]
            return value >= boundary + self.hysteresis
        boundary_idx = min(candidate, len(active_thresholds) - 1)
        boundary = active_thresholds[boundary_idx]
        return value <= boundary - self.hysteresis

    def _compute_class_stats(self, labels: pd.Series) -> dict[str, Any]:
        finite = labels.dropna().astype(int)
        counts_raw = finite.value_counts().to_dict()
        counts = {int(k): int(v) for k, v in counts_raw.items()}
        for cls in range(self.n_classes):
            counts.setdefault(cls, 0)
        counts = dict(sorted(counts.items()))

        vals = np.array(list(counts.values()), dtype=float)
        total = vals.sum()
        probs = vals / total if total > 0 else np.zeros_like(vals)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum() / np.log(self.n_classes))
        nonzero = vals[vals > 0]
        imbalance = float(nonzero.max() / nonzero.min()) if len(nonzero) > 0 else float("inf")
        return {"counts": counts, "entropy": entropy, "imbalance_ratio": imbalance}

    def _validate_class_constraints(self, stats: dict[str, Any]) -> None:
        if not self._is_class_constraints_ok(stats):
            raise ValueError(
                "Class constraints failed: "
                f"counts={stats['counts']}, "
                f"entropy={stats['entropy']:.4f}, "
                f"imbalance={stats['imbalance_ratio']:.4f}"
            )

    def _is_class_constraints_ok(self, stats: dict[str, Any]) -> bool:
        counts = stats["counts"]
        if any(counts[k] < self.min_class_support for k in counts):
            return False
        if stats["imbalance_ratio"] > self.max_class_imbalance:
            return False
        if stats["entropy"] < self.min_class_entropy:
            return False
        return True

    @staticmethod
    def _label_turnover(labels: pd.Series) -> float:
        if len(labels) < 2:
            return np.nan
        return float((labels != labels.shift(1)).dropna().mean())

    @staticmethod
    def _mean_dwell(labels: pd.Series) -> float:
        arr = labels.dropna().to_numpy()
        if len(arr) == 0:
            return np.nan
        runs = []
        run = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        return float(np.mean(runs))

    @staticmethod
    def _whipsaw_rate(labels: pd.Series) -> float:
        arr = labels.dropna().to_numpy()
        if len(arr) < 3:
            return 0.0
        whipsaw = 0
        for i in range(2, len(arr)):
            if arr[i] == arr[i - 2] and arr[i] != arr[i - 1]:
                whipsaw += 1
        return float(whipsaw / (len(arr) - 2))

    def _build_training_reference(self, values: np.ndarray, labels: pd.Series, clip_rate: float) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        edges = self._histogram_edges(finite, self.psi_bins)
        probs = self._hist_probs(finite, edges)
        rng = np.random.default_rng(self.random_state)
        sample_size = min(len(finite), 10000)
        sample_idx = rng.choice(len(finite), size=sample_size, replace=False)
        sample_values = finite[sample_idx]
        return {
            "clip_rate": float(clip_rate),
            "psi_bin_edges": edges.tolist(),
            "psi_base_probs": probs.tolist(),
            "class_probabilities": self._class_probabilities(labels).tolist(),
            "sample_values": sample_values.tolist(),
        }

    @staticmethod
    def _histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
        probs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(values, probs)
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Ensure monotonic increase even for repeated quantiles.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        return edges

    @staticmethod
    def _hist_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=edges)
        probs = counts.astype(float) / max(float(counts.sum()), 1.0)
        return probs

    @staticmethod
    def _population_stability_index(base_probs: np.ndarray, current_probs: np.ndarray) -> float:
        eps = 1e-8
        b = np.clip(base_probs, eps, None)
        c = np.clip(current_probs, eps, None)
        return float(np.sum((c - b) * np.log(c / b)))

    def _class_probabilities(self, labels: pd.Series) -> np.ndarray:
        clean = labels.dropna().astype(int)
        counts = np.zeros(self.n_classes, dtype=float)
        for k, v in clean.value_counts().to_dict().items():
            if 0 <= int(k) < self.n_classes:
                counts[int(k)] = float(v)
        total = counts.sum()
        if total <= 0.0:
            return np.zeros(self.n_classes, dtype=float)
        return counts / total

    def _default_acceptance_gates(self) -> dict[str, float]:
        common = {
            "min_coverage": 0.80,
            "max_turnover": 0.35,
            "min_mean_dwell": 2.0,
            "max_clip_rate": 0.25,
            "min_class_entropy": self.min_class_entropy,
            "max_class_imbalance": self.max_class_imbalance,
            "min_class_support": float(self.min_class_support),
        }
        directional_types = {MarketPropertyType.DIRECTION, MarketPropertyType.MOMENTUM}
        if self.market_property_type in directional_types:
            common.update(
                {
                    "min_edge": 0.0,
                    "min_hit_rate": 0.50,
                    "max_perm_pvalue": 0.10,
                    "min_positive_edge_block_fraction": 0.55,
                }
            )
        else:
            common.update(
                {
                    "min_separation_score": 0.01,
                    "max_kruskal_pvalue": 0.10,
                }
            )
        return common

    def _directional_acceptance_metrics(self, feature: pd.Series | np.ndarray | list[float],
                                        forward_returns: pd.Series | np.ndarray | list[float], n_perm: int = 200,
                                        blocks: Optional[pd.DataFrame] = None) -> dict[str, float]:
        labels = self.predict(feature)
        fwd = _align_to_index(forward_returns, "forward_return", labels.index)
        valid = labels.notna() & fwd.notna()
        if valid.sum() < max(50, self.n_classes * self.min_class_support):
            return {
                "edge": np.nan,
                "edge_bps": np.nan,
                "hit_rate": np.nan,
                "perm_pvalue": np.nan,
                "positive_edge_block_fraction": np.nan,
            }

        labels_arr = labels.loc[valid].to_numpy(dtype=int)
        returns_arr = fwd.loc[valid].to_numpy(dtype=float)
        signals = self.class_signals_[labels_arr]
        edge = float(np.mean(signals * returns_arr))
        nz = signals != 0
        hit_rate = float(np.mean(np.sign(returns_arr[nz]) == np.sign(signals[nz]))) if nz.any() else np.nan

        # Block permutation preserves within-block signal autocorrelation. IID
        # permutation breaks it entirely and produces optimistically small
        # p-values on serially correlated financial time series. Block size of
        # n^(1/3) is the standard Politis-Romano rule of thumb.
        n_obs = len(signals)
        perm_block_size = max(1, int(np.ceil(n_obs ** (1.0 / 3.0))))
        block_starts = np.arange(0, n_obs, perm_block_size)
        rng = np.random.default_rng(self.random_state)
        perm = np.empty(n_perm)
        for i in range(n_perm):
            shuffled_starts = rng.permutation(block_starts)
            perm_sig = np.concatenate([signals[s : s + perm_block_size] for s in shuffled_starts])[:n_obs]
            perm[i] = np.mean(perm_sig * returns_arr)
        perm_pvalue = float((np.sum(np.abs(perm) >= abs(edge)) + 1) / (n_perm + 1))

        if blocks is None:
            blocks = self.evaluate_forward_blocks(feature=feature, forward_returns=forward_returns, block_size=1000)
        if "edge" in blocks.columns and blocks["edge"].notna().any():
            positive_frac = float((blocks["edge"] > 0).mean())
        else:
            positive_frac = np.nan

        return {
            "edge": edge,
            "edge_bps": edge * 10000.0,
            "hit_rate": hit_rate,
            "perm_pvalue": perm_pvalue,
            "positive_edge_block_fraction": positive_frac,
        }

    def _non_directional_acceptance_metrics(self, feature: pd.Series | np.ndarray | list[float],
                                            forward_returns: Optional[pd.Series | np.ndarray | list[float]]) -> dict[str, float]:
        labels = self.predict(feature)
        feat = _align_to_index(feature, "feature", labels.index)
        target = feat
        if forward_returns is not None:
            target = _align_to_index(forward_returns, "forward_return", labels.index)

        valid = labels.notna() & target.notna()
        if valid.sum() < max(50, self.n_classes * self.min_class_support):
            return {"separation_score": np.nan, "kruskal_pvalue": np.nan}

        labels_arr = labels.loc[valid].to_numpy(dtype=int)
        target_arr = target.loc[valid].to_numpy(dtype=float)
        separation = self._separation_score(target_arr, labels_arr)

        groups = [target_arr[labels_arr == k] for k in range(self.n_classes) if np.any(labels_arr == k)]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                kw_p = float(kruskal(*groups).pvalue)
            except Exception:
                kw_p = np.nan
        else:
            kw_p = np.nan
        return {"separation_score": separation, "kruskal_pvalue": kw_p}

    def __repr__(self) -> str:
        return (
            f"DirectFeatureThresholdOptimizer("
            f"market_property_type={self.market_property_type.value!r}, "
            f"n_classes={self.n_classes}, "
            f"threshold_method={self.threshold_method.value!r}, "
            f"clip_method={self.clip_method.value!r}, "
            f"objective_type={self.objective_type.value!r}, "
            f"min_persistence={self.min_persistence}, "
            f"hysteresis={self.hysteresis})"
        )
