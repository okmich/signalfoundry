"""Streaming monitor — CLI entry point + per-symbol orchestration."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import asdict
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


def _read_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"_read_metadata: metadata.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _glob_inference_logs(log_dir: Path, strategy_name: str) -> list[Path]:
    """Return ``inference_<strategy>_<YYYYMMDD>.jsonl`` files sorted by date suffix."""
    pattern = f"inference_{strategy_name}_*.jsonl"
    files = sorted(log_dir.glob(pattern))
    return files


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
    log_files = _glob_inference_logs(cfg.inference_log_base_dir, strategy_name)
    if not log_files:
        raise FileNotFoundError(
            f"run_monitor_for_symbol: no inference logs found at "
            f"{cfg.inference_log_base_dir}/inference_{strategy_name}_*.jsonl"
        )

    frame = read_inference_log(log_files).tail(cfg.tail_n)

    # Override gate thresholds with caller-tuned values.
    from okmich_quant_ml.posterior_inference import (
        score_feature_health, score_loglik_health, score_posterior_health,
    )
    posterior_report = score_posterior_health(
        frame.posteriors, posterior_baselines,
        max_entropy_abs_z=cfg.max_entropy_abs_z,
        max_occupancy_drift_l1=cfg.max_occupancy_drift_l1,
        max_flip_rate_drift_abs=cfg.max_flip_rate_drift_abs,
    )
    feature_report = score_feature_health(
        frame.features, feature_baselines,
        max_ks_statistic=cfg.max_ks_statistic, alpha=cfg.ks_alpha,
    )
    loglik_report = score_loglik_health(
        frame.logliks, loglik_baselines, max_abs_z=cfg.max_loglik_abs_z,
    )
    overall_ok = bool(posterior_report.overall_ok and feature_report.overall_ok and loglik_report.overall_ok)
    report = MonitoringCycleReport(
        overall_ok=overall_ok, posterior=posterior_report, feature=feature_report, loglik=loglik_report,
        log_window_n=frame.n_bars, log_window_start_ts=frame.label_timestamps[0],
        log_window_end_ts=frame.label_timestamps[-1],
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
