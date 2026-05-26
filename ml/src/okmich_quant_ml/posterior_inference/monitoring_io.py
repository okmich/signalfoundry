"""Log-based I/O primitives for the streaming monitoring pipeline.

The live trader writes one JSONL line per inference cycle via
:class:`okmich_quant_core.JsonlInferenceLogger`; the schema is the universal
:class:`okmich_quant_core.InferenceLogRecord` with HMM-specific fields nested
under ``extras``. The monitor process (running on its own cadence, out-of-band)
reads those files via :func:`read_inference_log`, hydrates the cached
baselines from ``metadata.json`` via
:func:`load_posterior_and_loglik_baselines_from_metadata`, and evaluates the
three streaming gates via :func:`run_streaming_gates`. The trader has no
dependency on this module; the monitor has no dependency on the trader.

HMM-runner contract (what the reader expects at each line):

* Top-level: ``label_bar_ts`` (ISO string or null) — bar the prediction is for
* Top-level: ``features`` (dict[str, float]) — emission feature values
* ``extras.probs`` (list[float] of length n_states) — matured posterior
* ``extras.loglik`` (float) — per-bar predictive log-likelihood

Lines where ``label_bar_ts`` is null, or where ``extras.probs`` / ``extras.loglik``
is missing or null (typically warmup rows where the fixed-lag posterior is not
yet matured), are silently dropped during read — the gates operate only on the
fully-populated tail.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .monitoring import FeatureHealthBaselines, FeatureHealthReport, LoglikDriftBaselines, LoglikDriftReport,\
    PosteriorHealthBaselines, PosteriorHealthReport, score_feature_health, score_loglik_health,\
    score_posterior_health


@dataclass(frozen=True)
class InferenceLogFrame:
    """Typed, time-aligned view of per-bar inference log records.

    All arrays share a common length ``T`` and are aligned to
    ``label_timestamps[i]`` (the bar the prediction at row ``i`` is *for*).
    ``features`` columns are aligned to ``feature_names``; ``posteriors``
    rows are aligned to the state ordering used by the model that emitted
    the log. Built by :func:`read_inference_log` from JSONL files written
    by the trader; warmup rows (where ``probs`` or ``loglik`` were ``null``)
    are dropped upstream so every row here is gate-ready.

    Read-only arrays in ``__post_init__`` (``setflags(write=False)``) extend
    the frozen-dataclass immutability through ndarray contents. Length
    consistency across all four fields is asserted on construction.

    Named ``InferenceLogFrame`` (rather than the naive ``InferenceLog``) to
    avoid collision with the per-record :class:`okmich_quant_core.InferenceLogRecord`
    that the writer emits — this is the *tabular* / columnar projection of
    many such records.
    """
    label_timestamps: pd.DatetimeIndex
    features: NDArray
    feature_names: tuple[str, ...]
    posteriors: NDArray
    logliks: NDArray

    def __post_init__(self) -> None:
        T = self.posteriors.shape[0]
        if not (self.features.shape[0] == T and self.logliks.shape[0] == T and len(self.label_timestamps) == T):
            raise ValueError(
                f"InferenceLogFrame: array lengths inconsistent — "
                f"posteriors={self.posteriors.shape[0]}, features={self.features.shape[0]}, "
                f"logliks={self.logliks.shape[0]}, label_timestamps={len(self.label_timestamps)}"
            )
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"InferenceLogFrame: features has {self.features.shape[1]} columns but feature_names has "
                f"{len(self.feature_names)} entries"
            )
        self.features.setflags(write=False)
        self.posteriors.setflags(write=False)
        self.logliks.setflags(write=False)

    @property
    def n_bars(self) -> int:
        return int(self.posteriors.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.posteriors.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def tail(self, n: int) -> InferenceLogFrame:
        """Return a new InferenceLogFrame with only the last ``n`` rows.

        Useful for the monitor process to slice down to the last day or two
        of logs before invoking :func:`run_streaming_gates`.
        """
        if n < 0:
            raise ValueError(f"InferenceLogFrame.tail: n must be >= 0, got {n}")
        if n >= self.n_bars:
            return self
        return InferenceLogFrame(
            label_timestamps=self.label_timestamps[-n:],
            features=self.features[-n:].copy(),
            feature_names=self.feature_names,
            posteriors=self.posteriors[-n:].copy(),
            logliks=self.logliks[-n:].copy(),
        )


def _coerce_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def read_inference_log(paths: str | Path | Iterable[str | Path]) -> InferenceLogFrame:
    """Parse one or more inference JSONL files into a typed :class:`InferenceLogFrame`.

    Each line must conform to :class:`okmich_quant_core.InferenceLogRecord`'s
    serialised form: top-level ``label_bar_ts`` (ISO string or null),
    ``features`` (dict[str, float]), plus an ``extras`` sub-block carrying
    HMM-specific fields ``probs`` (list of floats) and ``loglik`` (float).

    ``paths`` may be a single file path or an iterable of paths; files are
    concatenated in the order they are given (caller is responsible for
    sorting if chronological order is required — typically by filename when
    the trader uses daily-rotated log files).

    Lines where ``label_bar_ts`` is null, or where ``extras.probs`` /
    ``extras.loglik`` is missing or null, are silently dropped — these are
    warmup rows the gates cannot evaluate. Lines where required top-level
    keys are missing entirely raise ``ValueError``. The feature-name set
    and ``n_states`` must be consistent across all retained rows, otherwise
    a ``ValueError`` is raised — schema changes (variant swap, n_states
    change) must rotate to a new log file rather than appending.

    **Truncated tail tolerance:** if the *last* line of a file fails to
    parse as JSON, it is treated as a torn write (trader killed mid-flush)
    and skipped with a warning rather than raising. Earlier invalid lines
    still raise — they indicate genuine corruption, not crash truncation.

    All timestamps are coerced to UTC (``utc=True``) at read time, so logs
    written by traders in different timezones / DST regimes compare cleanly
    on the canonical UTC axis.
    """
    resolved = _coerce_paths(paths)
    if not resolved:
        raise ValueError("read_inference_log: paths is empty")

    label_ts: list[pd.Timestamp] = []
    feature_rows: list[list[float]] = []
    posterior_rows: list[list[float]] = []
    loglik_values: list[float] = []
    feature_names: tuple[str, ...] | None = None
    n_states: int | None = None

    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"read_inference_log: log file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw_lines = fh.readlines()
        for line_no, raw in enumerate(raw_lines, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                # Last-line failure → likely torn write from trader killed mid-flush.
                # Skip with a warning so the monitor sees the rest of the file.
                if line_no == len(raw_lines):
                    import warnings
                    warnings.warn(
                        f"read_inference_log: ignoring truncated/invalid last line at {path}:{line_no}: {exc}",
                        RuntimeWarning,
                    )
                    continue
                raise ValueError(f"read_inference_log: invalid JSON at {path}:{line_no}: {exc}") from exc

            required = ("label_bar_ts", "features", "extras")
            missing = [k for k in required if k not in record]
            if missing:
                raise ValueError(
                    f"read_inference_log: missing required keys {missing} at {path}:{line_no}"
                )

            extras = record["extras"]
            if not isinstance(extras, Mapping):
                raise ValueError(
                    f"read_inference_log: 'extras' must be a mapping at {path}:{line_no}, "
                    f"got {type(extras).__name__}"
                )

            # Warmup / null-decision rows: any of the gate inputs missing → drop silently.
            if record["label_bar_ts"] is None:
                continue
            probs = extras.get("probs")
            loglik = extras.get("loglik")
            if probs is None or loglik is None:
                continue

            features = record["features"]
            if not isinstance(features, Mapping):
                raise ValueError(
                    f"read_inference_log: 'features' must be a mapping at {path}:{line_no}, "
                    f"got {type(features).__name__}"
                )

            row_feature_names = tuple(features.keys())
            if feature_names is None:
                feature_names = row_feature_names
            elif row_feature_names != feature_names:
                raise ValueError(
                    f"read_inference_log: feature-name set changed at {path}:{line_no}: "
                    f"expected {feature_names}, got {row_feature_names}"
                )

            if not isinstance(probs, list):
                raise ValueError(
                    f"read_inference_log: extras['probs'] must be a list at {path}:{line_no}, "
                    f"got {type(probs).__name__}"
                )
            row_n_states = len(probs)
            if n_states is None:
                n_states = row_n_states
            elif row_n_states != n_states:
                raise ValueError(
                    f"read_inference_log: posterior length changed at {path}:{line_no}: "
                    f"expected {n_states}, got {row_n_states}"
                )

            label_ts.append(pd.Timestamp(record["label_bar_ts"], tz="UTC")
                            if pd.Timestamp(record["label_bar_ts"]).tz is None
                            else pd.Timestamp(record["label_bar_ts"]).tz_convert("UTC"))
            feature_rows.append([float(features[name]) for name in feature_names])
            posterior_rows.append([float(p) for p in probs])
            loglik_values.append(float(loglik))

    if not label_ts:
        raise ValueError(
            f"read_inference_log: no fully-populated rows found across {len(resolved)} file(s) "
            f"— every line either was empty, was a warmup row, or had null probs/loglik."
        )

    return InferenceLogFrame(label_timestamps=pd.DatetimeIndex(label_ts), features=np.asarray(feature_rows, dtype=float),
                        feature_names=feature_names, posteriors=np.asarray(posterior_rows, dtype=float),
                        logliks=np.asarray(loglik_values, dtype=float))


@dataclass(frozen=True)
class MonitoringCycleReport:
    """Aggregate verdict from one monitor cycle.

    Carries the three component gate reports plus the trailing-window
    metadata (n bars, first and last label timestamps) so the consumer
    can write a structured cycle report (JSONL) for the alert pipeline
    and the audit log. ``overall_ok`` is the strict AND across all three
    component gates — any single gate failure fails the cycle.
    """
    overall_ok: bool
    posterior: PosteriorHealthReport
    feature: FeatureHealthReport
    loglik: LoglikDriftReport
    log_window_n: int
    log_window_start_ts: pd.Timestamp
    log_window_end_ts: pd.Timestamp


def run_streaming_gates(log: InferenceLogFrame, posterior_baselines: PosteriorHealthBaselines,
                        feature_baselines: FeatureHealthBaselines,
                        loglik_baselines: LoglikDriftBaselines) -> MonitoringCycleReport:
    """Evaluate all three streaming gates against ``log`` and aggregate.

    Each gate is run against the entire log slice the caller passes:

      * :func:`score_posterior_health` reads ``log.posteriors`` (final-row gate)
      * :func:`score_feature_health`   reads ``log.features`` (KS over the whole slice)
      * :func:`score_loglik_health`    reads ``log.logliks`` (final-row gate)

    Caller is responsible for sizing the log slice — typical pattern is
    ``log.tail(max(2 * posterior_baselines.window, 2 * loglik_baselines.window))``
    so the posterior/loglik gates are well past warmup and the feature gate
    has enough samples for KS power. Mismatched shapes between log and
    baselines (n_states, n_features) raise from the underlying gate.

    ``overall_ok`` is the strict AND across the three component ``overall_ok``
    fields. The cycle window metadata (start/end timestamps and bar count)
    is captured so the consumer can timestamp the cycle report.

    Pre-validates the log against the baselines before invoking any gate, so
    one under-populated gate cannot abort the cycle mid-eval and discard the
    other two gates' findings.
    """
    if log.n_bars == 0:
        raise ValueError("run_streaming_gates: log is empty")
    if log.feature_names != feature_baselines.feature_names:
        raise ValueError(
            f"run_streaming_gates: feature-name mismatch between log and baseline. "
            f"Log={log.feature_names}; baseline={feature_baselines.feature_names}. "
            f"KS comparison would silently pair wrong columns. Reorder log columns to match the baseline "
            f"or refit the baseline against the current feature ordering."
        )
    required_n = max(posterior_baselines.window, loglik_baselines.window)
    if log.n_bars < required_n:
        raise ValueError(
            f"run_streaming_gates: log has {log.n_bars} bars but the posterior/loglik gates need at least "
            f"{required_n} bars (max baseline window) to evaluate post-warmup. Wait for the log to grow "
            f"or pass a wider trailing slice."
        )

    posterior_report = score_posterior_health(log.posteriors, posterior_baselines)
    feature_report = score_feature_health(log.features, feature_baselines)
    loglik_report = score_loglik_health(log.logliks, loglik_baselines)

    overall_ok = bool(posterior_report.overall_ok and feature_report.overall_ok and loglik_report.overall_ok)
    return MonitoringCycleReport(overall_ok=overall_ok, posterior=posterior_report, feature=feature_report,
                                 loglik=loglik_report, log_window_n=log.n_bars,
                                 log_window_start_ts=pd.Timestamp(log.label_timestamps[0]),
                                 log_window_end_ts=pd.Timestamp(log.label_timestamps[-1]))


def load_posterior_and_loglik_baselines_from_metadata(
        metadata: Mapping) -> tuple[PosteriorHealthBaselines, LoglikDriftBaselines]:
    """Hydrate the cached scalar baselines from a parsed ``metadata.json``.

    Reads ``metadata["monitoring_baselines"]["posterior"]`` and
    ``metadata["monitoring_baselines"]["loglik"]`` via the dataclasses'
    ``from_dict`` constructors. The feature baseline is intentionally NOT
    loaded here — by design, feature baselines are re-derived at monitor
    runtime from the OOS window (via ``metadata["oos_window"]``), the
    persisted ``transform_pipeline.joblib``, and the OHLCV data lake. This
    keeps the (large) feature samples out of ``metadata.json`` while still
    locking the (small, probabilistic) scalar baselines as a deployment
    audit record.

    Raises ``ValueError`` if the ``monitoring_baselines`` block is missing
    or malformed.
    """
    if "monitoring_baselines" not in metadata:
        raise ValueError(
            "load_posterior_and_loglik_baselines_from_metadata: metadata is missing 'monitoring_baselines' block "
            "— was the artefact produced by generate_all_001.py post-2026-05-25?"
        )
    block = metadata["monitoring_baselines"]
    if not isinstance(block, Mapping):
        raise ValueError(
            f"load_posterior_and_loglik_baselines_from_metadata: 'monitoring_baselines' must be a mapping, "
            f"got {type(block).__name__}"
        )
    for required in ("posterior", "loglik"):
        if required not in block:
            raise ValueError(
                f"load_posterior_and_loglik_baselines_from_metadata: 'monitoring_baselines' missing '{required}' sub-block"
            )
    posterior = PosteriorHealthBaselines.from_dict(block["posterior"])
    loglik = LoglikDriftBaselines.from_dict(block["loglik"])
    return posterior, loglik
