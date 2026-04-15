from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True)
class PolicyWindowMetrics:
    sharpe: float
    total_log_return: float
    max_drawdown: float
    turnover: int


class LabelAlignmentStatus(StrEnum):
    ALIGNED = "aligned"
    WEAKLY_ALIGNED = "weakly_aligned"
    MISALIGNED = "misaligned"


class PolicyQualityStatus(StrEnum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"


def scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def split_non_overlapping_windows(n_rows: int, target_windows: int, min_window_size: int) -> list[tuple[int, int]]:
    if n_rows <= 0:
        return []
    if min_window_size <= 0:
        raise ValueError(f"min_window_size must be > 0, got {min_window_size}")
    if target_windows <= 0:
        raise ValueError(f"target_windows must be > 0, got {target_windows}")

    max_by_size = max(1, n_rows // min_window_size)
    window_count = max(1, min(int(target_windows), max_by_size))

    base_size = n_rows // window_count
    remainder = n_rows % window_count
    windows: list[tuple[int, int]] = []
    start = 0
    for idx in range(window_count):
        extra = 1 if idx < remainder else 0
        end = start + base_size + extra
        if end > start:
            windows.append((start, end))
        start = end
    return windows


def build_forward_return_oracle_labels(ret_fwd_train: np.ndarray, ret_fwd_test: np.ndarray,
                                       neutral_quantile: float = 0.40) -> dict[str, np.ndarray | float]:
    if not 0.0 < neutral_quantile < 1.0:
        raise ValueError(f"neutral_quantile must be in (0,1), got {neutral_quantile}")

    train = np.asarray(ret_fwd_train, dtype=float)
    test = np.asarray(ret_fwd_test, dtype=float)
    abs_train = np.abs(train)
    threshold = float(np.quantile(abs_train, neutral_quantile))

    def _label(x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), 1, dtype=np.int64)
        out[x > threshold] = 2
        out[x < -threshold] = 0
        return out

    y_train = _label(train)
    y_test = _label(test)
    train_dist = np.bincount(y_train, minlength=3) / max(len(y_train), 1)
    test_dist = np.bincount(y_test, minlength=3) / max(len(y_test), 1)
    return {
        "threshold_abs_return": threshold,
        "y_train": y_train,
        "y_test": y_test,
        "train_distribution": train_dist,
        "test_distribution": test_dist,
    }


def learn_state_to_label_matrix(state_probs: np.ndarray, y_idx: np.ndarray, n_labels: int = 3) -> np.ndarray:
    probs = np.asarray(state_probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"state_probs must be 2D, got shape {probs.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
    if len(probs) != len(y):
        raise ValueError(f"state_probs rows must match y_idx length. Got {len(probs)} vs {len(y)}")

    n_states = int(probs.shape[1])
    counts = np.ones((n_states, n_labels), dtype=np.float64)
    for label in range(n_labels):
        mask = y == label
        if mask.any():
            counts[:, label] += np.asarray(probs[mask], dtype=np.float64).sum(axis=0)
    return counts / counts.sum(axis=1, keepdims=True)


def multi_class_log_loss(label_probs: np.ndarray, y_idx: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(label_probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"label_probs must be 2D, got shape {probs.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
    if len(probs) != len(y):
        raise ValueError(f"label_probs rows must match y_idx length. Got {len(probs)} vs {len(y)}")

    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def multi_class_accuracy(label_probs: np.ndarray, y_idx: np.ndarray) -> float:
    probs = np.asarray(label_probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y))


def expected_calibration_error_multiclass(label_probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(label_probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (float(mask.sum()) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_label_alignment_from_oracle(probs_train: np.ndarray, probs_test: np.ndarray,
                                         oracle_train: np.ndarray, oracle_test: np.ndarray,
                                         train_prior: np.ndarray) -> dict:
    mapping = learn_state_to_label_matrix(probs_train, oracle_train, n_labels=3)
    label_probs_test = np.asarray(probs_test, dtype=float) @ mapping
    baseline_probs_test = np.tile(np.asarray(train_prior, dtype=float), (len(oracle_test), 1))
    model_log_loss = multi_class_log_loss(label_probs_test, oracle_test)
    baseline_log_loss = multi_class_log_loss(baseline_probs_test, oracle_test)
    log_loss_delta = baseline_log_loss - model_log_loss
    if log_loss_delta >= 0.02:
        status = LabelAlignmentStatus.ALIGNED
    elif log_loss_delta >= 0.0:
        status = LabelAlignmentStatus.WEAKLY_ALIGNED
    else:
        status = LabelAlignmentStatus.MISALIGNED

    return {
        "status": status.value,
        "log_loss": float(model_log_loss),
        "baseline_log_loss": float(baseline_log_loss),
        "log_loss_delta_vs_baseline": float(log_loss_delta),
        "accuracy": float(multi_class_accuracy(label_probs_test, oracle_test)),
        "baseline_accuracy": float(multi_class_accuracy(baseline_probs_test, oracle_test)),
        "ece": float(expected_calibration_error_multiclass(label_probs_test, oracle_test, n_bins=10)),
        "mean_top_prob": float(np.mean(label_probs_test.max(axis=1))),
        "state_to_oracle_mapping": [[float(x) for x in row] for row in mapping],
    }


def evaluate_label_alignment(probs_train: np.ndarray, probs_test: np.ndarray, ret_fwd_train: np.ndarray,
                             ret_fwd_test: np.ndarray, neutral_quantile: float = 0.40) -> dict:
    oracle = build_forward_return_oracle_labels(ret_fwd_train, ret_fwd_test, neutral_quantile=neutral_quantile)
    result = evaluate_label_alignment_from_oracle(
        probs_train=probs_train,
        probs_test=probs_test,
        oracle_train=np.asarray(oracle["y_train"], dtype=np.int64),
        oracle_test=np.asarray(oracle["y_test"], dtype=np.int64),
        train_prior=np.asarray(oracle["train_distribution"], dtype=float),
    )
    result["oracle"] = {
        "neutral_quantile": float(neutral_quantile),
        "abs_return_threshold": float(oracle["threshold_abs_return"]),
        "train_label_distribution_bear_flat_bull": [
            float(x) for x in np.asarray(oracle["train_distribution"], dtype=float)
        ],
        "test_label_distribution_bear_flat_bull": [
            float(x) for x in np.asarray(oracle["test_distribution"], dtype=float)
        ],
    }
    return result


def summarize_walk_forward_policy_quality(baseline_windows: list[PolicyWindowMetrics],
                                          candidate_windows: list[PolicyWindowMetrics]) -> dict:
    if len(baseline_windows) == 0:
        raise ValueError("baseline_windows must contain at least one window.")
    if len(baseline_windows) != len(candidate_windows):
        raise ValueError(
            f"baseline_windows and candidate_windows must have the same length. "
            f"Got {len(baseline_windows)} vs {len(candidate_windows)}."
        )

    sharpe_deltas: list[float] = []
    total_return_deltas: list[float] = []
    drawdown_deltas: list[float] = []
    turnover_ratios: list[float] = []
    quality_scores: list[float] = []

    for baseline, candidate in zip(baseline_windows, candidate_windows):
        sharpe_delta = float(candidate.sharpe - baseline.sharpe)
        total_return_delta = float(candidate.total_log_return - baseline.total_log_return)
        drawdown_delta = float(candidate.max_drawdown - baseline.max_drawdown)
        base_turnover = max(int(baseline.turnover), 1)
        turnover_ratio = float(candidate.turnover / base_turnover)
        quality_score = sharpe_delta - (0.20 * max(0.0, turnover_ratio - 1.0)) - (0.05 * max(0.0, -drawdown_delta))

        sharpe_deltas.append(sharpe_delta)
        total_return_deltas.append(total_return_delta)
        drawdown_deltas.append(drawdown_delta)
        turnover_ratios.append(turnover_ratio)
        quality_scores.append(quality_score)

    mean_sharpe_delta = float(np.mean(sharpe_deltas))
    positive_sharpe_fraction = float(np.mean(np.asarray(sharpe_deltas) > 0.0))
    if mean_sharpe_delta >= 0.0 and positive_sharpe_fraction >= 0.5:
        status = PolicyQualityStatus.POSITIVE
    elif mean_sharpe_delta > -0.10:
        status = PolicyQualityStatus.MIXED
    else:
        status = PolicyQualityStatus.NEGATIVE

    return {
        "status": status.value,
        "pass": status != PolicyQualityStatus.NEGATIVE,
        "windows_total": len(baseline_windows),
        "positive_sharpe_window_fraction": positive_sharpe_fraction,
        "sharpe_delta": scalar_summary(sharpe_deltas),
        "total_log_return_delta": scalar_summary(total_return_deltas),
        "max_drawdown_delta": scalar_summary(drawdown_deltas),
        "turnover_ratio": scalar_summary(turnover_ratios),
        "quality_score": scalar_summary(quality_scores),
    }
