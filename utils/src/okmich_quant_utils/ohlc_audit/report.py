"""Audit report: console summary, files (findings / summary / events), JSON. Blind mode per SPEC §8."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from .audit import BrokerAuditReport
from .enums import SEVERITY_RANK, Severity, Verdict


def default_out_dir(report: BrokerAuditReport) -> Path:
    """A per-run folder under the OS temp dir. Reports never go into the market-data tree (pass --out to keep one)."""
    return Path(tempfile.gettempdir()) / "raw-data-audit" / report.broker / f"{report.run_at:%Y%m%dT%H%M%SZ}"


def write(report: BrokerAuditReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"findings": out_dir / "findings.csv", "summary": out_dir / "summary.csv", "events": out_dir / "events.csv",
             "json": out_dir / "summary.json"}
    report.findings_frame().to_csv(paths["findings"], index=False)
    report.summary_frame().to_csv(paths["summary"], index=False)
    report.events.to_csv(paths["events"], index=False)
    paths["json"].write_text(json.dumps(to_json(report), indent=2, default=str), encoding="utf-8")
    return paths


def to_json(report: BrokerAuditReport) -> dict:
    s = report.summary_frame()
    return {"broker": report.broker, "bar_minutes": report.bar_minutes, "run_at": str(report.run_at),
            "blind_after_raw": str(report.blind_after_raw) if report.blind_after_raw is not None else None,
            "registry": report.registry_source,
            "verdicts": s.groupby("verdict").size().to_dict(), "symbols": s.to_dict(orient="records"),
            "broker_wide_events": len(report.events)}


def _fmt_window(r) -> str:
    if pd.isna(r["start_raw"]):
        return "(no timestamp)"                        # D1 MISSING_STAMP: rows without a stamp have no window
    return f"{pd.Timestamp(r['start_raw']):%Y-%m-%d %H:%M} -> {pd.Timestamp(r['end_raw']):%Y-%m-%d %H:%M}"


def format_console(report: BrokerAuditReport, max_rows: int = 40) -> str:
    lines = ["=" * 110, f"OHLC AUDIT  {report.broker}  {report.bar_minutes}m bars   run {report.run_at:%Y-%m-%d %H:%M} UTC   "
             f"symbols {len(report.symbols)}" + (f"   blind after raw {report.blind_after_raw:%Y-%m-%d %H:%M}" if report.blind_after_raw is not None else ""),
             f"registry: {report.registry_source or 'none given (every finding is unregistered)'}", "=" * 110]
    s = report.summary_frame()
    cols = ["symbol", "class", "bars", "coverage_pct", "critical", "major", "minor", "info", "unregistered_crit_major", "registry_windows", "verdict"]
    lines.append(s[cols].rename(columns={"unregistered_crit_major": "unreg C/M", "registry_windows": "registry", "coverage_pct": "cover%"})
                 .to_string(index=False))
    counts = s["verdict"].value_counts()
    lines.append("")
    lines.append("Verdicts: " + ", ".join(f"{v.value}={int(counts.get(v.value, 0))}" for v in Verdict))
    ev = report.events
    if len(ev):
        new = ev[~ev["registered"]]
        lines += ["", f"Broker-wide events (MAJOR/CRITICAL on >= 3 symbols): {len(ev)}, of which not fully registered: {len(new)}"]
        if len(new):
            show = new.copy()
            show["window_raw"] = show.apply(_fmt_window, axis=1)
            lines.append(show[["kind", "severity", "window_raw", "n_symbols", "symbols", "blind"]].head(max_rows).to_string(index=False))
    f = report.findings_frame()
    if len(f):
        det = f[~f["detector"].isin(["registry", "override"])]
        heavy = det[det["severity"].isin([Severity.CRITICAL.value, Severity.MAJOR.value]) & (det["registry_id"] == "")].copy()
        lines += ["", f"Unregistered CRITICAL/MAJOR findings: {len(heavy)}"]
        if len(heavy):
            heavy["rank"] = heavy["severity"].map({k.value: v for k, v in SEVERITY_RANK.items()})
            heavy = heavy.sort_values(["rank", "start_raw"], ascending=[False, True])
            heavy["window_raw"] = heavy.apply(_fmt_window, axis=1)
            lines.append(heavy[["symbol", "detector", "kind", "severity", "window_raw", "n_bars", "evidence"]].head(max_rows).to_string(index=False))
            if len(heavy) > max_rows:
                lines.append(f"... {len(heavy) - max_rows} more in findings.csv")
    lines.append("=" * 110)
    return "\n".join(lines)
