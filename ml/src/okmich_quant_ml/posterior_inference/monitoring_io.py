"""Log-based I/O primitives for the streaming monitoring pipeline.

The live trader writes one LOGGING_CONTRACT v1.0.0 record per inference cycle via
:class:`okmich_quant_core.JsonlEventLogger`. The monitor process (running on its own
cadence, out-of-band) reads those files via :func:`read_inference_log`, hydrates the
cached baselines from ``metadata.json`` via
:func:`load_posterior_and_loglik_baselines_from_metadata`, and evaluates the
three streaming gates via :func:`run_streaming_gates`. The trader has no
dependency on this module; the monitor has no dependency on the trader.

HMM-runner contract (what the reader pulls from each ``event == "bar"`` record):

* ``label_bar_ts`` (ISO string or null) — bar the prediction is for
* ``features`` (dict[str, float]) — emission feature values
* ``extras.probs`` (list[float] of length n_states) — matured posterior
* ``extras.loglik`` (float) — per-bar predictive log-likelihood

Lines where ``label_bar_ts`` is null, or where ``extras.probs`` / ``extras.loglik``
is missing or null (typically warmup rows where the fixed-lag posterior is not
yet matured), are silently dropped during read — the gates operate only on the
fully-populated tail.
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from okmich_quant_core.logging import LOG_SCHEMA_VERSION, BarOutcome, LogEventType

_SUPPORTED_MAJOR = int(LOG_SCHEMA_VERSION.split(".")[0])

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
    avoid collision with the per-record :class:`okmich_quant_core.BarRecord`
    that the writer emits — this is the *tabular* / columnar projection of
    many such ``bar`` records.
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


def _to_utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def read_inference_log(paths: str | Path | Iterable[str | Path],
                       on_quarantine: Callable[[str, int, str], None] | None = None) -> InferenceLogFrame:
    """Parse one or more inference JSONL files into a typed :class:`InferenceLogFrame`.

    Reads the LOGGING_CONTRACT v1.0.0 event-typed schema (each line is a record with an ``event``
    discriminator). Only ``event == "bar"`` records with ``outcome == "ok"`` feed the gates;
    ``startup`` / ``shutdown`` / breaker records and ``error`` / ``skipped_disabled`` bars are
    skipped (a failed/partial cycle must not leak a stale posterior into the baseline comparison).
    From each gate-eligible ``bar`` this pulls Tier 1 ``label_bar_ts`` + ``features`` and the HMM
    ``extras.probs`` / ``extras.loglik`` the gates read.

    ``paths`` may be a single path or an iterable; files are concatenated in the given order
    (caller sorts — typically by filename for daily-rotated logs).

    **Warmup rows** — a ``bar`` where ``label_bar_ts`` is null, or ``extras.probs`` /
    ``extras.loglik`` is missing/null — are silently dropped; the gates run on the populated tail.

    **Quarantine-and-continue (LOGGING_CONTRACT §13).** A line that fails to parse, carries an
    unknown major ``log_schema_version``, or is a structurally invalid ``bar`` (bad ``extras`` /
    ``features`` type, feature-name drift, ``n_states`` drift) is **rejected, recorded, and an
    alert-worthy warning raised — then parsing continues**. One corrupt line never blinds the
    consumer to the healthy heartbeats after it. ``on_quarantine(path, line_no, reason)`` is invoked
    per offending line (a side channel for the supervisor); a ``RuntimeWarning`` is always raised.

    Raises only for whole-input problems: empty ``paths``, a missing file, or no fully-populated
    ``bar`` rows across all files (every line empty / non-bar / warmup / quarantined).

    All timestamps are coerced to UTC at read time.
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
    quarantined_total = 0

    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"read_inference_log: log file not found: {path}")
        # Per-file quarantine accounting. We deliberately do NOT warn per line: the default
        # warnings filter de-dups by call site, so a systematically-bad file would surface only its
        # first quarantine and silently hide the rest. Instead, the per-line detail goes to the
        # on_quarantine side channel, and one aggregate alert-worthy warning is raised per file (§13).
        file_quarantined: list[tuple[int, str]] = []

        def _quarantine(line_no: int, reason: str, _p: Path = path) -> None:
            file_quarantined.append((line_no, reason))
            if on_quarantine is not None:
                on_quarantine(str(_p), line_no, reason)

        with open(path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):  # stream — never materialise the whole file
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    _quarantine(line_no, f"invalid JSON: {exc}")
                    continue

                if not isinstance(record, Mapping) or "event" not in record:
                    _quarantine(line_no, "missing 'event' discriminator")
                    continue

                version = record.get("log_schema_version")
                if version is None:
                    _quarantine(line_no, "missing required 'log_schema_version'")
                    continue
                try:
                    if int(str(version).split(".")[0]) > _SUPPORTED_MAJOR:
                        _quarantine(line_no, f"unknown major log_schema_version {version!r} "
                                             f"(consumer supports major {_SUPPORTED_MAJOR})")
                        continue
                except (ValueError, AttributeError):
                    _quarantine(line_no, f"unparseable log_schema_version {version!r}")
                    continue

                if record["event"] != LogEventType.BAR.value:
                    continue  # startup / shutdown / breaker records are not gate input

                # Only completed cycles feed the gates: an error/skipped bar must not leak a stale or
                # partial posterior into the drift baseline comparison.
                if record.get("outcome") != BarOutcome.OK.value:
                    continue

                extras = record.get("extras", {})
                if not isinstance(extras, Mapping):
                    _quarantine(line_no, f"'extras' must be a mapping, got {type(extras).__name__}")
                    continue
                features = record.get("features", {})
                if not isinstance(features, Mapping):
                    _quarantine(line_no, f"'features' must be a mapping, got {type(features).__name__}")
                    continue

                # Warmup / null-decision rows: any gate input missing → drop silently.
                if record.get("label_bar_ts") is None:
                    continue
                probs = extras.get("probs")
                loglik = extras.get("loglik")
                if probs is None or loglik is None:
                    continue

                row_feature_names = tuple(features.keys())
                if feature_names is None:
                    feature_names = row_feature_names
                elif row_feature_names != feature_names:
                    _quarantine(line_no, f"feature-name set changed: expected {feature_names}, "
                                         f"got {row_feature_names} (rotate to a new file on schema change)")
                    continue

                if not isinstance(probs, list):
                    _quarantine(line_no, f"extras['probs'] must be a list, got {type(probs).__name__}")
                    continue
                if n_states is None:
                    n_states = len(probs)
                elif len(probs) != n_states:
                    _quarantine(line_no, f"posterior length changed: expected {n_states}, got {len(probs)} "
                                         f"(rotate to a new file on n_states change)")
                    continue

                try:
                    row_features = [float(features[name]) for name in feature_names]
                    row_label = _to_utc(record["label_bar_ts"])
                    row_probs = [float(p) for p in probs]
                    row_loglik = float(loglik)
                except (KeyError, TypeError, ValueError) as exc:
                    _quarantine(line_no, f"non-numeric gate values: {exc}")
                    continue
                label_ts.append(row_label)
                feature_rows.append(row_features)
                posterior_rows.append(row_probs)
                loglik_values.append(row_loglik)

        if file_quarantined:
            quarantined_total += len(file_quarantined)
            first_ln, first_reason = file_quarantined[0]
            warnings.warn(
                f"read_inference_log: quarantined {len(file_quarantined)} line(s) in {path}; "
                f"first at line {first_ln}: {first_reason}", RuntimeWarning,
            )

    if not label_ts:
        raise ValueError(
            f"read_inference_log: no fully-populated bar rows found across {len(resolved)} file(s) "
            f"— every line was empty, non-bar, a non-ok outcome, a warmup row, or quarantined "
            f"(quarantined={quarantined_total})."
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
                        loglik_baselines: LoglikDriftBaselines, *,
                        max_entropy_abs_z: float = 3.0, max_occupancy_drift_l1: float = 0.2,
                        max_flip_rate_drift_abs: float = 0.1, max_ks_statistic: float = 0.1,
                        ks_alpha: float = 0.01, max_loglik_abs_z: float = 3.0) -> MonitoringCycleReport:
    """Evaluate all three streaming gates against ``log`` and aggregate.

    Gate thresholds are forwarded to the underlying scorers; the defaults match each scorer's own
    defaults, so callers that don't tune them get identical behaviour. Tuning callers (the streaming
    monitor) pass them here rather than calling the scorers directly, so the pre-validation guards
    below (feature-name alignment, minimum window) are never bypassed.

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

    posterior_report = score_posterior_health(log.posteriors, posterior_baselines,
                                              max_entropy_abs_z=max_entropy_abs_z,
                                              max_occupancy_drift_l1=max_occupancy_drift_l1,
                                              max_flip_rate_drift_abs=max_flip_rate_drift_abs)
    feature_report = score_feature_health(log.features, feature_baselines,
                                          max_ks_statistic=max_ks_statistic, alpha=ks_alpha)
    loglik_report = score_loglik_health(log.logliks, loglik_baselines, max_abs_z=max_loglik_abs_z)

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
