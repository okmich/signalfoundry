"""Streaming monitor — CLI entry point + per-symbol orchestration."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from okmich_quant_core import BaseNotifier
from okmich_quant_ml.posterior_inference import (
    MonitoringCycleReport,
    load_posterior_and_loglik_baselines_from_metadata,
    read_inference_log,
    run_streaming_gates,
)

from .config import MonitorConfig
from .counter_state import ViolationCounterState
from .feature_loader import derive_feature_baseline_from_oos_window, load_transform_pipeline


_log = logging.getLogger(__name__)


class StaleInferenceLogError(RuntimeError):
    """The most recent gate-eligible bar is older than ``MonitorConfig.max_log_age_minutes``.

    Raised before scoring so a stalled/broken live stream is never scored as healthy — which would
    reset the violation counters and mask the outage. Surfaced as a per-symbol failure by ``main()``.
    """


def _read_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"_read_metadata: metadata.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _glob_inference_logs(log_base: Path, strategy_name: str, symbol: str,
                         timeframe: int | None = None) -> list[Path]:
    """Return the OPS-path inference files for ONE logical system, sorted by date.

    Path layout (LOGGING_CONTRACT §10 / OPS §7):
    ``<log_base>/<strategy>/<symbol>/<timeframe>/inference/inference_<YYYYMMDD>.jsonl``. Sorted
    lexicographically, which is chronological for the zero-padded ``YYYYMMDD`` filename suffix.

    A logical system is ``strategy/symbol/timeframe`` and the drift baselines are cadence-specific,
    so this must resolve to a SINGLE timeframe. When ``timeframe`` is given, only that segment is
    read. When it is ``None`` the timeframe is auto-detected, but if logs exist under more than one
    timeframe a ``ValueError`` is raised rather than silently mixing M1/M5/H1 records into one frame.
    """
    base = Path(log_base) / strategy_name / symbol
    if timeframe is not None:
        return sorted((base / str(timeframe)).glob("inference/inference_*.jsonl"))
    tf_dirs_with_logs = sorted(
        d for d in base.glob("*") if d.is_dir() and any(d.glob("inference/inference_*.jsonl"))
    )
    if len(tf_dirs_with_logs) > 1:
        raise ValueError(
            f"_glob_inference_logs: {base} has inference logs under multiple timeframes "
            f"{[d.name for d in tf_dirs_with_logs]}; drift baselines are cadence-specific. "
            f"Set MonitorConfig.timeframe to disambiguate."
        )
    if not tf_dirs_with_logs:
        return []
    return sorted(tf_dirs_with_logs[0].glob("inference/inference_*.jsonl"))


def _serialise_cycle_report(report: MonitoringCycleReport) -> dict[str, Any]:
    """Flatten the report into a JSON-serialisable dict for the cycle log."""
    return {
        "overall_ok": bool(report.overall_ok),
        "log_window_n": int(report.log_window_n),
        "log_window_start_ts": str(report.log_window_start_ts),
        "log_window_end_ts": str(report.log_window_end_ts),
        "posterior": {
            "overall_ok": bool(report.posterior.overall_ok),
            "entropy_staleness_z": float(report.posterior.entropy_staleness_z),
            "entropy_staleness_ok": bool(report.posterior.entropy_staleness_ok),
            "occupancy_drift_l1": float(report.posterior.occupancy_drift_l1),
            "occupancy_drift_ok": bool(report.posterior.occupancy_drift_ok),
            "flip_rate_drift_signed": float(report.posterior.flip_rate_drift_signed),
            "flip_rate_drift_ok": bool(report.posterior.flip_rate_drift_ok),
        },
        "feature": {
            "overall_ok": bool(report.feature.overall_ok),
            "ks_statistics": [float(x) for x in report.feature.ks_statistics],
            "p_values": [float(x) for x in report.feature.p_values],
            "per_feature_ok": [bool(x) for x in report.feature.per_feature_ok],
            "feature_names": list(report.feature.feature_names),
        },
        "loglik": {
            "overall_ok": bool(report.loglik.overall_ok),
            "loglik_drift_z": float(report.loglik.loglik_drift_z),
            "loglik_drift_ok": bool(report.loglik.loglik_drift_ok),
        },
    }


def _append_cycle_report(output_dir: Path, symbol: str, report_dict: dict[str, Any]) -> Path:
    """Append cycle report to ``<output_dir>/<symbol>/monitor_cycles_<YYYYMMDD>.jsonl``."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    target_dir = output_dir / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"monitor_cycles_{today}.jsonl"
    payload = dict(report_dict)
    payload["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    return target


def _build_alert_message(symbol: str, tripped: list[str], report: MonitoringCycleReport,
                        threshold: int) -> str:
    bullets = []
    for gate in tripped:
        if gate == "posterior":
            bullets.append(
                f"posterior: entropy_z={report.posterior.entropy_staleness_z:.2f} "
                f"occupancy_l1={report.posterior.occupancy_drift_l1:.3f} "
                f"flip_rate_diff={report.posterior.flip_rate_drift_signed:+.3f}"
            )
        elif gate == "feature":
            failed = [report.feature.feature_names[i] for i in range(len(report.feature.feature_names))
                      if not bool(report.feature.per_feature_ok[i])]
            bullets.append(f"feature: KS-failed columns {failed}")
        elif gate == "loglik":
            bullets.append(f"loglik: z={report.loglik.loglik_drift_z:+.2f}")
    return (f"Drift alert [{symbol}]: {len(tripped)} gate(s) tripped after {threshold} consecutive "
            f"failing cycles — {'; '.join(bullets)}")


def run_monitor_for_symbol(cfg: MonitorConfig, symbol: str,
                           notifier: BaseNotifier | None = None) -> MonitoringCycleReport:
    """Run one monitor cycle for ``symbol``.

    End-to-end flow: read metadata → hydrate posterior+loglik baselines from cache →
    re-derive feature baselines from OOS window → glob inference logs → tail to
    ``cfg.tail_n`` → ``run_streaming_gates`` → persist cycle report → advance counter
    state → dispatch alerts on newly-tripped gates. Returns the cycle report so the
    caller can summarise across symbols.

    Raises on any setup failure (missing artefacts, empty OOS slice, no logs); the
    caller (``main``) wraps in try/except for per-symbol failure isolation.
    """
    artifact_dir = cfg.artifact_base_dir / symbol / cfg.variant_with_lag
    metadata = _read_metadata(artifact_dir)

    posterior_baselines, loglik_baselines = load_posterior_and_loglik_baselines_from_metadata(metadata)

    oos_window = metadata.get("oos_window")
    if not oos_window:
        raise ValueError(f"run_monitor_for_symbol: metadata.json at {artifact_dir} missing 'oos_window' block")
    feature_columns = list(metadata.get("feature_columns", []))
    if not feature_columns:
        raise ValueError(f"run_monitor_for_symbol: metadata.json at {artifact_dir} missing 'feature_columns'")

    transform_pipeline = load_transform_pipeline(artifact_dir)
    feature_baselines = derive_feature_baseline_from_oos_window(
        raw_data_dir=cfg.raw_data_dir, symbol=symbol,
        oos_start_ts=oos_window["start_ts"], oos_end_ts=oos_window["end_ts"],
        feature_columns=feature_columns,
        feature_engineering_callable=cfg.feature_engineering_callable,
        transform_pipeline=transform_pipeline,
    )

    strategy_name = cfg.strategy_name_template.format(symbol=symbol)
    log_files = _glob_inference_logs(cfg.inference_log_base_dir, strategy_name, symbol, cfg.timeframe)
    if not log_files:
        tf_seg = cfg.timeframe if cfg.timeframe is not None else "*"
        raise FileNotFoundError(
            f"run_monitor_for_symbol: no inference logs found at "
            f"{cfg.inference_log_base_dir}/{strategy_name}/{symbol}/{tf_seg}/inference/inference_*.jsonl"
        )

    frame = read_inference_log(log_files).tail(cfg.tail_n)

    # Freshness gate: refuse to score a stale stream. read_inference_log quarantines bad recent lines
    # but still returns older valid rows, so without this a broken/stalled live stream could be scored
    # as healthy and silently RESET the alert counters. Fail the cycle instead (counters untouched).
    if cfg.max_log_age_minutes is not None:
        last_ts = frame.label_timestamps[-1].to_pydatetime()
        age_minutes = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0
        if age_minutes > cfg.max_log_age_minutes:
            raise StaleInferenceLogError(
                f"run_monitor_for_symbol: {symbol} inference log is stale — last gate-eligible bar "
                f"{last_ts.isoformat()} is {age_minutes:.1f}min old "
                f"(> max_log_age_minutes={cfg.max_log_age_minutes}); refusing to score stale data."
            )

    # Delegate to run_streaming_gates with the caller-tuned thresholds. Doing the scoring here by hand
    # would bypass that function's pre-validation (feature-name alignment + minimum-window) — a
    # feature-name mismatch silently pairs the wrong KS columns, and an under-populated tail aborts
    # mid-eval. Forwarding the thresholds keeps both guards and removes the duplicated aggregation.
    report = run_streaming_gates(
        frame, posterior_baselines, feature_baselines, loglik_baselines,
        max_entropy_abs_z=cfg.max_entropy_abs_z,
        max_occupancy_drift_l1=cfg.max_occupancy_drift_l1,
        max_flip_rate_drift_abs=cfg.max_flip_rate_drift_abs,
        max_ks_statistic=cfg.max_ks_statistic, ks_alpha=cfg.ks_alpha,
        max_loglik_abs_z=cfg.max_loglik_abs_z,
    )

    _append_cycle_report(cfg.output_dir, symbol, _serialise_cycle_report(report))

    counter_path = cfg.output_dir / symbol / "counter_state.json"
    state = ViolationCounterState.load_or_init(counter_path)
    tripped = state.advance(report, cfg.violation_counter_threshold)
    state.save(counter_path)

    if tripped and notifier is not None:
        message = _build_alert_message(symbol, tripped, report, cfg.violation_counter_threshold)
        try:
            notifier.on_error(strategy_name=strategy_name, error_message=message,
                              context=_serialise_cycle_report(report))
        except Exception as exc:
            _log.exception(f"run_monitor_for_symbol: notifier.on_error raised for {symbol}: {exc}")

    return report


def _build_notifier(spec: dict[str, Any] | None) -> BaseNotifier | None:
    """Instantiate a notifier from the config spec, or return None.

    Currently supports ``{"type": "telegram", "bot_token": "...", "chat_id": "...",
    "strategy_name": "...", "broker": "..."}``. Returns None when spec is None or
    when the type is unknown (logs a warning).
    """
    if spec is None:
        return None
    notifier_type = spec.get("type", "").lower()
    if notifier_type == "telegram":
        from okmich_quant_core import TelegramNotifier
        return TelegramNotifier(
            bot_token=spec["bot_token"], chat_id=spec["chat_id"],
            strategy_name=spec.get("strategy_name", "monitor"),
            broker=spec.get("broker", ""),
        )
    _log.warning(f"_build_notifier: unknown notifier type {notifier_type!r}; alerts disabled.")
    return None


def main() -> int:
    """CLI entry point: parse config, iterate symbols, summarise outcomes.

    Per-symbol failures are caught and logged so one bad symbol does not kill
    the others. Returns a non-zero exit code if any symbol failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the streaming drift monitor for one or more symbols.")
    parser.add_argument("--config", required=True, type=str, help="Path to the monitor JSON config.")
    args = parser.parse_args()

    cfg = MonitorConfig.from_json(args.config)
    notifier = _build_notifier(cfg.notifier)

    successes: list[str] = []
    failures: dict[str, str] = {}
    for symbol in cfg.symbols:
        try:
            report = run_monitor_for_symbol(cfg, symbol, notifier=notifier)
            _log.info(
                f"[{symbol}] cycle ok={report.overall_ok} n_bars={report.log_window_n} "
                f"posterior_ok={report.posterior.overall_ok} feature_ok={report.feature.overall_ok} "
                f"loglik_ok={report.loglik.overall_ok}"
            )
            successes.append(symbol)
        except Exception as exc:
            _log.exception(f"[{symbol}] monitor cycle FAILED: {exc}")
            failures[symbol] = f"{type(exc).__name__}: {exc}"

    _log.info(
        f"Monitor summary: {len(successes)} ok, {len(failures)} failed. "
        f"ok={successes}; failed={list(failures)}"
    )
    if notifier is not None:
        notifier.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
