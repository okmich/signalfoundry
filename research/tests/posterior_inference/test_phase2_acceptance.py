import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_phase2_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "src" / "okmich_quant_research" / "posterior_inference" / "phase2_acceptance.py"
    spec = importlib.util.spec_from_file_location("phase2_acceptance", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_phase2 = _load_phase2_module()
PolicyWindowMetrics = _phase2.PolicyWindowMetrics
evaluate_label_alignment_from_oracle = _phase2.evaluate_label_alignment_from_oracle
split_non_overlapping_windows = _phase2.split_non_overlapping_windows
summarize_walk_forward_policy_quality = _phase2.summarize_walk_forward_policy_quality


def test_split_non_overlapping_windows_covers_full_range() -> None:
    windows = split_non_overlapping_windows(n_rows=103, target_windows=4, min_window_size=20)

    assert len(windows) == 4
    lengths = [end - start for start, end in windows]
    assert sum(lengths) == 103
    assert min(lengths) >= 25
    assert windows[0][0] == 0
    assert windows[-1][1] == 103


def test_evaluate_label_alignment_from_oracle_detects_signal_vs_prior() -> None:
    n_train = 300
    n_test = 180
    oracle_train = np.arange(n_train, dtype=np.int64) % 3
    oracle_test = (np.arange(n_test, dtype=np.int64) + 1) % 3

    probs_train = np.full((n_train, 3), 0.05, dtype=float)
    probs_train[np.arange(n_train), oracle_train] = 0.90
    probs_test = np.full((n_test, 3), 0.10, dtype=float)
    probs_test[np.arange(n_test), oracle_test] = 0.80

    train_prior = np.bincount(oracle_train, minlength=3) / float(n_train)
    result = evaluate_label_alignment_from_oracle(
        probs_train=probs_train,
        probs_test=probs_test,
        oracle_train=oracle_train,
        oracle_test=oracle_test,
        train_prior=train_prior,
    )

    assert result["log_loss_delta_vs_baseline"] > 0.0
    assert result["status"] in {"aligned", "weakly_aligned"}


def test_summarize_walk_forward_policy_quality_positive_status() -> None:
    baseline = [
        PolicyWindowMetrics(sharpe=0.30, total_log_return=0.08, max_drawdown=-0.18, turnover=100),
        PolicyWindowMetrics(sharpe=0.45, total_log_return=0.10, max_drawdown=-0.20, turnover=120),
        PolicyWindowMetrics(sharpe=0.25, total_log_return=0.05, max_drawdown=-0.16, turnover=90),
    ]
    candidate = [
        PolicyWindowMetrics(sharpe=0.40, total_log_return=0.09, max_drawdown=-0.15, turnover=105),
        PolicyWindowMetrics(sharpe=0.50, total_log_return=0.12, max_drawdown=-0.18, turnover=118),
        PolicyWindowMetrics(sharpe=0.31, total_log_return=0.06, max_drawdown=-0.14, turnover=92),
    ]

    result = summarize_walk_forward_policy_quality(baseline, candidate)

    assert result["status"] == "positive"
    assert result["pass"] is True
    assert result["windows_total"] == 3


def test_summarize_walk_forward_policy_quality_negative_status() -> None:
    baseline = [
        PolicyWindowMetrics(sharpe=0.60, total_log_return=0.15, max_drawdown=-0.10, turnover=80),
        PolicyWindowMetrics(sharpe=0.55, total_log_return=0.14, max_drawdown=-0.11, turnover=85),
        PolicyWindowMetrics(sharpe=0.50, total_log_return=0.12, max_drawdown=-0.09, turnover=82),
    ]
    candidate = [
        PolicyWindowMetrics(sharpe=0.20, total_log_return=0.05, max_drawdown=-0.20, turnover=160),
        PolicyWindowMetrics(sharpe=0.15, total_log_return=0.03, max_drawdown=-0.22, turnover=170),
        PolicyWindowMetrics(sharpe=0.18, total_log_return=0.04, max_drawdown=-0.19, turnover=150),
    ]

    result = summarize_walk_forward_policy_quality(baseline, candidate)

    assert result["status"] == "negative"
    assert result["pass"] is False
