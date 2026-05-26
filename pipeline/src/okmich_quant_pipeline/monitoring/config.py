"""Frozen configuration dataclass for the streaming monitor + JSON loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonitorConfig:
    """All knobs the streaming monitor needs to run.

    Loaded from a JSON file via :meth:`from_json`; the typical deployment
    is one JSON per broker/timeframe pair, committed alongside other ops
    configs. Paths are resolved relative to the config file's parent on
    load so configs are portable.

    Field reference:

    - ``symbols`` — list of symbol names to monitor in this invocation.
      One run iterates them serially; per-symbol failure is isolated.
    - ``artifact_base_dir`` — directory containing
      ``<symbol>/<variant_with_lag>/metadata.json`` per the
      ``generate_all_001.py`` layout.
    - ``variant_with_lag`` — subfolder name (e.g. ``hmm_lambda_L3``);
      same across all symbols in this config.
    - ``inference_log_base_dir`` — where the trader writes
      ``inference_<strategy>_<YYYYMMDD>.jsonl`` (one strategy per symbol).
    - ``strategy_name_template`` — format string with ``{symbol}`` placeholder
      used to derive the per-symbol strategy name for log-file globbing.
    - ``raw_data_dir`` — root of the OHLCV parquet lake; expects
      ``<raw_data_dir>/<symbol>.parquet``.
    - ``output_dir`` — where the monitor writes per-symbol cycle reports
      and counter state.
    - ``feature_engineering_callable`` — importable path
      ``pkg.module:func_name`` to the feature-engineering function used at
      fit time. Must accept ``(df, feature_columns)`` and return a DataFrame.
    - ``tail_n`` — how many most-recent log rows to feed the gates. Must be
      >= max(posterior_window, loglik_window) from baselines.
    - ``violation_counter_threshold`` — number of consecutive cycle failures
      per gate before an alert is dispatched.
    - Gate thresholds: ``max_entropy_abs_z``, ``max_occupancy_drift_l1``,
      ``max_flip_rate_drift_abs``, ``max_ks_statistic``, ``ks_alpha``,
      ``max_loglik_abs_z``. Default to the values from monitoring.py docstrings.
    - ``notifier`` — optional dict with at least ``{"type": "..."}``; the
      orchestrator instantiates the named notifier and dispatches alerts
      via :meth:`on_error`. ``None`` disables alerting (cycle reports + counter
      state still persist).
    """
    symbols: tuple[str, ...]
    artifact_base_dir: Path
    variant_with_lag: str
    inference_log_base_dir: Path
    strategy_name_template: str
    raw_data_dir: Path
    output_dir: Path
    feature_engineering_callable: str
    tail_n: int
    violation_counter_threshold: int
    max_entropy_abs_z: float = 3.0
    max_occupancy_drift_l1: float = 0.2
    max_flip_rate_drift_abs: float = 0.1
    max_ks_statistic: float = 0.1
    ks_alpha: float = 0.01
    max_loglik_abs_z: float = 3.0
    notifier: dict[str, Any] | None = field(default=None)

    @classmethod
    def from_json(cls, path: str | Path) -> MonitorConfig:
        """Load + validate a monitor config from JSON. Paths are resolved
        relative to ``path``'s parent so configs are portable across machines.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"MonitorConfig.from_json: config file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        base = p.parent

        required = (
            "symbols", "artifact_base_dir", "variant_with_lag", "inference_log_base_dir",
            "strategy_name_template", "raw_data_dir", "output_dir", "feature_engineering_callable",
            "tail_n", "violation_counter_threshold",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"MonitorConfig.from_json: missing required keys {missing} in {p}")

        symbols = tuple(payload["symbols"])
        if not symbols:
            raise ValueError(f"MonitorConfig.from_json: 'symbols' must be non-empty")
        if "{symbol}" not in payload["strategy_name_template"]:
            raise ValueError(
                f"MonitorConfig.from_json: 'strategy_name_template' must contain "
                f"'{{symbol}}' placeholder, got {payload['strategy_name_template']!r}"
            )
        if int(payload["tail_n"]) < 1:
            raise ValueError(f"MonitorConfig.from_json: 'tail_n' must be >= 1, got {payload['tail_n']}")
        if int(payload["violation_counter_threshold"]) < 1:
            raise ValueError(
                f"MonitorConfig.from_json: 'violation_counter_threshold' must be >= 1, "
                f"got {payload['violation_counter_threshold']}"
            )
        if ":" not in payload["feature_engineering_callable"]:
            raise ValueError(
                f"MonitorConfig.from_json: 'feature_engineering_callable' must be of the form "
                f"'pkg.module:func_name', got {payload['feature_engineering_callable']!r}"
            )

        def _resolve(rel: str) -> Path:
            return (base / rel).resolve()

        return cls(
            symbols=symbols,
            artifact_base_dir=_resolve(payload["artifact_base_dir"]),
            variant_with_lag=str(payload["variant_with_lag"]),
            inference_log_base_dir=_resolve(payload["inference_log_base_dir"]),
            strategy_name_template=str(payload["strategy_name_template"]),
            raw_data_dir=_resolve(payload["raw_data_dir"]),
            output_dir=_resolve(payload["output_dir"]),
            feature_engineering_callable=str(payload["feature_engineering_callable"]),
            tail_n=int(payload["tail_n"]),
            violation_counter_threshold=int(payload["violation_counter_threshold"]),
            max_entropy_abs_z=float(payload.get("max_entropy_abs_z", 3.0)),
            max_occupancy_drift_l1=float(payload.get("max_occupancy_drift_l1", 0.2)),
            max_flip_rate_drift_abs=float(payload.get("max_flip_rate_drift_abs", 0.1)),
            max_ks_statistic=float(payload.get("max_ks_statistic", 0.1)),
            ks_alpha=float(payload.get("ks_alpha", 0.01)),
            max_loglik_abs_z=float(payload.get("max_loglik_abs_z", 3.0)),
            notifier=payload.get("notifier"),
        )
