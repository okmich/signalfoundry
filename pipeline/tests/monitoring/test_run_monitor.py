"""End-to-end test for run_monitor_for_symbol.

Builds a synthetic fixture tree (artefact dir with metadata.json + transform_pipeline.joblib,
OHLCV parquet, inference JSONL log) and runs the full per-symbol orchestration. Verifies the
cycle report is written, counter state is updated, and the orchestrator returns a well-shaped
MonitoringCycleReport.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from okmich_quant_core import InferenceLogRecord, JsonlInferenceLogger
from okmich_quant_ml.posterior_inference import (
    fit_loglik_drift_baselines,
    fit_posterior_health_baselines,
)
from okmich_quant_pipeline.monitoring import MonitorConfig, ViolationCounterState, run_monitor_for_symbol


# Test feature-engineering callable referenced via importable spec.
def _compute_features_for_tests(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Pass-through: the test OHLCV already contains the feature columns."""
    return df[feature_columns]


# Expose the callable at module level so it can be imported via the
# "pkg.module:func" spec used by the config.
import sys as _sys
_sys.modules[__name__].compute_features = _compute_features_for_tests


def _build_test_fixture(tmp_path: Path, symbol: str = "TESTSYM",
                        n_oos: int = 500, n_log: int = 300, n_states: int = 3) -> tuple[Path, MonitorConfig]:
    """Lay out a minimal fixture tree:

    tmp_path/
    ├── artefacts/<symbol>/<variant>/
    │   ├── metadata.json           (with monitoring_baselines + oos_window + feature_columns)
    │   └── transform_pipeline.joblib
    ├── raw/<symbol>.parquet         (OHLCV containing the feature columns directly)
    ├── logs/                        (one daily JSONL with n_log records)
    └── monitor_out/                 (empty; will receive cycle reports + counter state)
    """
    variant = "hmm_lambda_L3"
    feature_columns = ["tsi", "dbl_smoothed_log_rets"]
    rng = np.random.default_rng(42)

    # 1. Raw OHLCV parquet covering both OOS and log windows; index is UTC timestamps.
    n_raw = n_oos + 100
    raw_index = pd.date_range("2026-04-01T00:00:00Z", periods=n_raw, freq="5min")
    raw_features = rng.normal(size=(n_raw, len(feature_columns)))
    raw_df = pd.DataFrame(raw_features, index=raw_index, columns=feature_columns)
    raw_df["close"] = 100.0 + np.cumsum(rng.normal(0, 0.01, n_raw))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_df.to_parquet(raw_dir / f"{symbol}.parquet")

    # 2. Transform pipeline fitted on the OOS slice — same as production would do.
    oos_slice = raw_df.iloc[:n_oos]
    scaler = StandardScaler().fit(oos_slice[feature_columns].to_numpy())
    artefact_dir = tmp_path / "artefacts" / symbol / variant
    artefact_dir.mkdir(parents=True)
    joblib.dump(scaler, artefact_dir / "transform_pipeline.joblib")

    # 3. Compute baselines from the (scaled) OOS slice and embed in metadata.
    oos_X_scaled = scaler.transform(oos_slice[feature_columns].to_numpy())
    # Synthetic posteriors + logliks for the OOS slice — just dirichlet draws + normal.
    oos_posteriors = rng.dirichlet([1.0, 1.0, 1.0], size=n_oos)
    oos_logliks = rng.normal(loc=-2.0, scale=0.3, size=n_oos)
    posterior_b = fit_posterior_health_baselines(oos_posteriors, window=50)
    loglik_b = fit_loglik_drift_baselines(oos_logliks, window=50)
    metadata = {
        "model_type": "hmm", "symbol": symbol, "variant": variant,
        "feature_columns": feature_columns,
        "oos_window": {
            "start_ts": oos_slice.index[0].isoformat(),
            "end_ts": oos_slice.index[-1].isoformat(),
            "n_bars": n_oos,
        },
        "monitoring_baselines": {
            "posterior": posterior_b.to_dict(),
            "loglik": loglik_b.to_dict(),
        },
    }
    (artefact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 4. Inference log: write n_log records via the real JsonlInferenceLogger.
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    strategy_name = f"{symbol}_fl3_hmm"
    logger = JsonlInferenceLogger(log_dir, strategy_name=strategy_name, fsync=False)
    log_index = pd.date_range("2026-05-15T00:00:00Z", periods=n_log, freq="5min")
    live_posteriors = rng.dirichlet([1.0, 1.0, 1.0], size=n_log)
    live_logliks = rng.normal(loc=-2.0, scale=0.3, size=n_log)
    live_features = rng.normal(size=(n_log, len(feature_columns)))
    for i, ts in enumerate(log_index):
        logger.write(InferenceLogRecord(
            wall_clock_utc=pd.Timestamp(ts), asof_bar_ts=pd.Timestamp(ts),
            label_bar_ts=pd.Timestamp(ts), bar_close=100.0,
            features={col: float(live_features[i, j]) for j, col in enumerate(feature_columns)},
            direction=int(np.argmax(live_posteriors[i])) - 1,
            confidence=float(live_posteriors[i].max()),
            extras={"probs": [float(p) for p in live_posteriors[i]], "loglik": float(live_logliks[i])},
        ))
    logger.close()

    # 5. Config pointing at all the above.
    output_dir = tmp_path / "monitor_out"
    output_dir.mkdir()
    cfg = MonitorConfig(
        symbols=(symbol,),
        artifact_base_dir=tmp_path / "artefacts",
        variant_with_lag=variant,
        inference_log_base_dir=log_dir,
        strategy_name_template="{symbol}_fl3_hmm",
        raw_data_dir=raw_dir,
        output_dir=output_dir,
        feature_engineering_callable=f"{__name__}:compute_features",
        tail_n=200,
        violation_counter_threshold=3,
    )
    return tmp_path, cfg


def test_run_monitor_for_symbol_end_to_end(tmp_path: Path) -> None:
    _, cfg = _build_test_fixture(tmp_path)
    symbol = cfg.symbols[0]

    report = run_monitor_for_symbol(cfg, symbol, notifier=None)

    # Report shape
    assert report.log_window_n == cfg.tail_n
    assert report.posterior is not None
    assert report.feature is not None
    assert report.loglik is not None
    # Since live data is drawn from the same distribution as the OOS baseline,
    # all three gates should pass.
    assert report.overall_ok is True

    # Cycle report file written.
    cycle_files = list((cfg.output_dir / symbol).glob("monitor_cycles_*.jsonl"))
    assert len(cycle_files) == 1
    lines = cycle_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["overall_ok"] is True

    # Counter state initialised, counters at zero.
    counter_path = cfg.output_dir / symbol / "counter_state.json"
    assert counter_path.is_file()
    state = ViolationCounterState.load_or_init(counter_path)
    assert state.posterior_consecutive_failures == 0
    assert state.last_overall_ok is True


def test_run_monitor_for_symbol_persists_failure_counters_across_invocations(tmp_path: Path) -> None:
    """Force a failing cycle by passing an extreme threshold, run twice, verify counter persistence."""
    _, cfg = _build_test_fixture(tmp_path)
    # Override gate thresholds to absurdly tight so the posterior gate fails on noise.
    cfg = MonitorConfig(**{**cfg.__dict__, "max_entropy_abs_z": 0.001})
    symbol = cfg.symbols[0]

    report_a = run_monitor_for_symbol(cfg, symbol, notifier=None)
    assert report_a.posterior.overall_ok is False  # tight threshold definitely tripped

    state_after_first = ViolationCounterState.load_or_init(cfg.output_dir / symbol / "counter_state.json")
    assert state_after_first.posterior_consecutive_failures == 1

    report_b = run_monitor_for_symbol(cfg, symbol, notifier=None)
    state_after_second = ViolationCounterState.load_or_init(cfg.output_dir / symbol / "counter_state.json")
    assert state_after_second.posterior_consecutive_failures == 2


def test_run_monitor_for_symbol_missing_inference_logs_raises(tmp_path: Path) -> None:
    _, cfg = _build_test_fixture(tmp_path)
    # Wipe the log dir.
    for f in cfg.inference_log_base_dir.glob("*.jsonl"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match="no inference logs found"):
        run_monitor_for_symbol(cfg, cfg.symbols[0], notifier=None)


def test_run_monitor_for_symbol_missing_metadata_raises(tmp_path: Path) -> None:
    _, cfg = _build_test_fixture(tmp_path)
    # Delete metadata.json
    metadata = cfg.artifact_base_dir / cfg.symbols[0] / cfg.variant_with_lag / "metadata.json"
    metadata.unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json not found"):
        run_monitor_for_symbol(cfg, cfg.symbols[0], notifier=None)
