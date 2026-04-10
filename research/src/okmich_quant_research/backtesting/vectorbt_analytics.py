"""
VbtTradeAnalytics - A vectorbt trade dimensions analyzer
Derives time/context insights that vectorbt doesn't provide out of the box.

Usage:
    from okmich_quant_research.backtesting.vectorbt_analytics import VbtTradeAnalytics

    # From a vectorbt portfolio
    ta = VbtTradeAnalytics(pf.trades.records_readable)
    ta.show_dashboard()

    # Or save to HTML
    ta.show_dashboard(output_html="dashboard.html")
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Session / duration helpers
# ---------------------------------------------------------------------------

# Session hour ranges are defined in UTC.
SESSION_RANGES = {
    "Asian":          (0,  8),
    "London":         (8,  13),
    "NY–London OL":   (13, 16),
    "New York":       (16, 21),
    "Off-hours":      (21, 24),
}
SESSION_UNKNOWN = "Unknown"

DURATION_LABELS = ["Scalp (<1h)", "Intraday (1–8h)", "Swing (8h–3d)", "Position (>3d)", "Unknown"]

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SESSION_ORDER = list(SESSION_RANGES.keys()) + [SESSION_UNKNOWN]


def _get_session(hour: float) -> str:
    if pd.isna(hour):
        return SESSION_UNKNOWN
    for name, (start, end) in SESSION_RANGES.items():
        if start <= hour < end:
            return name
    return "Off-hours"


def _get_duration_bucket(minutes: float) -> str:
    if pd.isna(minutes) or minutes < 0:
        return "Unknown"
    if minutes < 60:
        return "Scalp (<1h)"
    elif minutes < 480:
        return "Intraday (1–8h)"
    elif minutes < 4320:
        return "Swing (8h–3d)"
    else:
        return "Position (>3d)"


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class VbtTradeAnalytics:
    """
    Accepts pf.trades.records_readable (a pandas DataFrame) and produces time/context dimensional analysis with a single
    Plotly dashboard.

    The dashboard has an Entry / Exit perspective toggle — letting you ask:
      - Entry view : "When I enter during X, how do I perform?"
      - Exit view  : "When I close during X, how do I perform?"

    Required columns:
        'Entry Index', 'Exit Index', 'PnL'
    Optional columns:
        'Direction' (missing/blank values are treated as "Unknown")
    """

    _BG     = "#0d1117"
    _PANEL  = "#161b22"
    _BORDER = "#30363d"
    _TEXT   = "#e6edf3"
    _SUB    = "#8b949e"
    _GREEN  = "#3fb950"
    _RED    = "#f85149"
    _BLUE   = "#58a6ff"
    _ORANGE = "#d29922"
    _PURPLE = "#bc8cff"

    SESSION_COLORS = {
        "Asian":        "#58a6ff",
        "London":       "#3fb950",
        "NY–London OL": "#d29922",
        "New York":     "#bc8cff",
        "Off-hours":    "#8b949e",
        SESSION_UNKNOWN: "#8b949e",
    }

    def __init__(self, trades_df: pd.DataFrame, source_tz: str = "UTC"):
        """
        Parameters
        ----------
        trades_df : DataFrame from pf.trades.records_readable.
        source_tz : IANA timezone of the timestamps (e.g. "Europe/Moscow", "US/Eastern").
                    Timestamps are converted to UTC internally for all time dimensions.
                    Defaults to "UTC" (no conversion needed).
        """
        self.raw = trades_df.copy()
        self._source_tz = source_tz
        self._entry_df, self._exit_df = self._prepare(trades_df)

    # ------------------------------------------------------------------
    # Data preparation — returns (entry_df, exit_df)
    # ------------------------------------------------------------------

    _REQUIRED_COLUMNS = {"Entry Index", "Exit Index", "PnL"}

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        missing = self._REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"trades_df is missing required columns: {sorted(missing)}. "
                             f"Expected columns from pf.trades.records_readable: {sorted(self._REQUIRED_COLUMNS)}")

        df["Entry Index"] = pd.to_datetime(df["Entry Index"])
        df["Exit Index"]  = pd.to_datetime(df["Exit Index"])

        df["duration_min"] = (df["Exit Index"] - df["Entry Index"]).dt.total_seconds() / 60
        df["duration_bucket"] = df["duration_min"].map(_get_duration_bucket)

        if "Direction" in df.columns:
            direction = df["Direction"].astype("string").str.strip().fillna(SESSION_UNKNOWN)
            df["direction"] = direction.mask(direction == "", SESSION_UNKNOWN)
        else:
            df["direction"] = SESSION_UNKNOWN

        entry_df = self._add_time_dims(df, "Entry Index", self._source_tz)
        exit_df  = self._add_time_dims(df, "Exit Index", self._source_tz)

        return entry_df, exit_df

    @staticmethod
    def _add_time_dims(df: pd.DataFrame, idx_col: str, source_tz: str = "UTC") -> pd.DataFrame:
        df = df.copy()
        ts = df[idx_col]
        # Keep hour/dow/month in the input data's native tz so analytics align with whatever
        # clock the live system reads (typically broker-local). Only convert to UTC for the
        # session lookup, since sessions are defined against universal market hours.
        if ts.dt.tz is None:
            ts_local = ts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        else:
            ts_local = ts
        ts_utc = ts_local.dt.tz_convert("UTC")
        df["_ts"] = ts_local
        df["hour"] = ts_local.dt.hour
        df["dow"] = ts_local.dt.day_name()
        df["month"] = ts_local.dt.month_name()
        df["month_n"] = ts_local.dt.month
        df["quarter"] = ts_local.dt.quarter.map(lambda q: f"Q{q}")
        df["session"]  = ts_utc.dt.hour.map(_get_session)
        return df

    # ------------------------------------------------------------------
    # Hour-filter impact analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _hour_filter_analysis(df: pd.DataFrame) -> dict:
        """
        Identify toxic hours and compute the return impact of filtering them.

        Returns a dict with:
            total_trades, total_pnl,
            hour_stats: [{hour, n_trades, total_pnl, mean_pnl, win_rate}, ...],
            toxic_hours: [hour, ...],        # hours with negative cumulative PnL
            profitable_hours: [hour, ...],   # top hours by cumulative PnL
            filter_scenarios: [{label, removed_hours, trades_kept, trades_removed, pnl_kept, pnl_removed, pnl_delta_pct}, ...]
        """
        total_pnl = df["PnL"].sum()
        total_trades = len(df)

        # Per-hour stats
        hour_groups = df.groupby("hour")["PnL"]
        hour_stats = []
        for hour, grp in hour_groups:
            hour_stats.append({
                "hour": int(hour),
                "n_trades": len(grp),
                "total_pnl": grp.sum(),
                "mean_pnl": grp.mean(),
                "win_rate": (grp > 0).mean() * 100,
            })
        hour_stats.sort(key=lambda x: x["hour"])

        # Identify toxic hours (negative cumulative PnL, at least 5 trades)
        toxic = [h for h in hour_stats if h["total_pnl"] < 0 and h["n_trades"] >= 5]
        toxic.sort(key=lambda x: x["total_pnl"])
        toxic_hours = [h["hour"] for h in toxic]

        # Top profitable hours
        profitable = [h for h in hour_stats if h["total_pnl"] > 0]
        profitable.sort(key=lambda x: x["total_pnl"], reverse=True)
        profitable_hours = [h["hour"] for h in profitable[:5]]

        # Build filter scenarios
        scenarios = [{"label": "Baseline (all hours)", "removed_hours": [], "trades_kept": total_trades, "trades_removed": 0, "pnl_kept": total_pnl, "pnl_removed": 0.0, "pnl_delta_pct": 0.0}]

        if toxic_hours:
            # Scenario: remove all toxic hours
            toxic_mask = df["hour"].isin(toxic_hours)
            kept = df[~toxic_mask]
            removed = df[toxic_mask]
            scenarios.append({
                "label": f"Remove toxic hours {toxic_hours}",
                "removed_hours": toxic_hours,
                "trades_kept": len(kept),
                "trades_removed": len(removed),
                "pnl_kept": kept["PnL"].sum(),
                "pnl_removed": removed["PnL"].sum(),
                "pnl_delta_pct": (kept["PnL"].sum() - total_pnl) / abs(total_pnl) * 100 if total_pnl != 0 else 0.0,
            })

            # Scenario: remove toxic + low-activity hours (< 5 trades)
            low_activity = [h["hour"] for h in hour_stats if h["n_trades"] < 5 and h["hour"] not in toxic_hours]
            if low_activity:
                extended = toxic_hours + low_activity
                ext_mask = df["hour"].isin(extended)
                kept_ext = df[~ext_mask]
                removed_ext = df[ext_mask]
                scenarios.append({
                    "label": f"Remove toxic + low-activity hours",
                    "removed_hours": sorted(extended),
                    "trades_kept": len(kept_ext),
                    "trades_removed": len(removed_ext),
                    "pnl_kept": kept_ext["PnL"].sum(),
                    "pnl_removed": removed_ext["PnL"].sum(),
                    "pnl_delta_pct": (kept_ext["PnL"].sum() - total_pnl) / abs(total_pnl) * 100 if total_pnl != 0 else 0.0,
                })

        # Per-session stats
        session_groups = df.groupby("session")["PnL"]
        session_stats = []
        for session, grp in session_groups:
            session_stats.append({
                "session": session,
                "n_trades": len(grp),
                "total_pnl": grp.sum(),
                "win_rate": (grp > 0).mean() * 100,
            })

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "hour_stats": hour_stats,
            "toxic_hours": toxic_hours,
            "profitable_hours": profitable_hours,
            "session_stats": session_stats,
            "filter_scenarios": scenarios,
        }

    def _build_filter_analysis_table(self, df: pd.DataFrame) -> go.Table:
        """Build a Plotly Table trace summarizing hour-filter impact analysis."""
        analysis = self._hour_filter_analysis(df)

        # --- Section 1: Hour breakdown ---
        hour_stats = analysis["hour_stats"]
        toxic_set = set(analysis["toxic_hours"])
        profitable_set = set(analysis["profitable_hours"])

        h_hours, h_trades, h_pnl, h_mean, h_wr, h_verdict = [], [], [], [], [], []
        for h in hour_stats:
            h_hours.append(str(h["hour"]))
            h_trades.append(str(h["n_trades"]))
            h_pnl.append(f"{h['total_pnl']:+.2f}")
            h_mean.append(f"{h['mean_pnl']:+.4f}")
            h_wr.append(f"{h['win_rate']:.0f}%")
            if h["hour"] in toxic_set:
                h_verdict.append("TOXIC")
            elif h["hour"] in profitable_set:
                h_verdict.append("STRONG")
            elif h["n_trades"] < 5:
                h_verdict.append("LOW-N")
            else:
                h_verdict.append("")

        # Color rows by verdict
        row_colors = []
        for v in h_verdict:
            if v == "TOXIC":
                row_colors.append(self._RED)
            elif v == "STRONG":
                row_colors.append(self._GREEN)
            elif v == "LOW-N":
                row_colors.append(self._ORANGE)
            else:
                row_colors.append(self._TEXT)

        return go.Table(
            header=dict(
                values=["Hour", "Trades", "Total PnL", "Mean PnL", "Win Rate", "Verdict"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[h_hours, h_trades, h_pnl, h_mean, h_wr, h_verdict],
                fill_color=self._BG,
                font=dict(color=[row_colors] * 6, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

    def _build_scenario_table(self, df: pd.DataFrame) -> go.Table:
        """Build a Plotly Table trace for filter scenarios."""
        analysis = self._hour_filter_analysis(df)
        scenarios = analysis["filter_scenarios"]

        s_label, s_hours, s_kept, s_removed, s_pnl, s_delta = [], [], [], [], [], []
        for s in scenarios:
            s_label.append(s["label"])
            s_hours.append(str(s["removed_hours"]) if s["removed_hours"] else "—")
            s_kept.append(str(s["trades_kept"]))
            s_removed.append(str(s["trades_removed"]))
            s_pnl.append(f"{s['pnl_kept']:+.2f}")
            s_delta.append(f"{s['pnl_delta_pct']:+.1f}%" if s["pnl_delta_pct"] != 0 else "—")

        return go.Table(
            header=dict(
                values=["Scenario", "Blocked Hours", "Trades Kept", "Removed", "Net PnL", "PnL Delta %"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[s_label, s_hours, s_kept, s_removed, s_pnl, s_delta],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

    @staticmethod
    def _format_recommendation(analysis: dict) -> list:
        """Format the hour-filter analysis into human-readable recommendation lines."""
        lines = []
        total_pnl = analysis["total_pnl"]
        toxic = analysis["toxic_hours"]
        profitable = analysis["profitable_hours"]
        scenarios = analysis["filter_scenarios"]

        if not toxic:
            lines.append("No consistently toxic hours detected (all hours with 5+ trades are net positive).")
            return lines

        # Toxic hours summary
        toxic_stats = [h for h in analysis["hour_stats"] if h["hour"] in toxic]
        toxic_pnl = sum(h["total_pnl"] for h in toxic_stats)
        toxic_trades = sum(h["n_trades"] for h in toxic_stats)
        lines.append(f"TOXIC HOURS: {toxic} — {toxic_trades} trades contributing {toxic_pnl:+.2f} PnL drag")

        # Best scenario
        if len(scenarios) > 1:
            best = max(scenarios[1:], key=lambda s: s["pnl_kept"])
            lines.append(f"BEST FILTER: {best['label']}")
            lines.append(f"  Removes {best['trades_removed']} trades, net PnL improves from {total_pnl:+.2f} to {best['pnl_kept']:+.2f} ({best['pnl_delta_pct']:+.1f}%)")
            lines.append(f"  Suggested blocked_hours config: {best['removed_hours']}")

        # Top profitable hours
        if profitable:
            top = analysis["hour_stats"]
            top_sorted = sorted([h for h in top if h["total_pnl"] > 0], key=lambda x: x["total_pnl"], reverse=True)[:3]
            top_str = ", ".join(f"hour {h['hour']} ({h['total_pnl']:+.2f}, {h['n_trades']}t)" for h in top_sorted)
            lines.append(f"TOP HOURS: {top_str}")

        # Session summary
        for s in analysis["session_stats"]:
            status = "+" if s["total_pnl"] > 0 else "-"
            lines.append(f"  {s['session']:15s}: {s['n_trades']:3d} trades, PnL={s['total_pnl']:+.2f}, WR={s['win_rate']:.0f}% [{status}]")

        return lines

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _pnl_by(self, df, col, sort_col=None):
        g = df.groupby(col)["PnL"].sum().reset_index()
        g.columns = [col, "PnL"]
        if sort_col:
            order = df[[col, sort_col]].drop_duplicates().sort_values(sort_col)
            g = g.set_index(col).reindex(order[col]).reset_index()
        return g

    def _pnl_heatmap(self, df):
        pivot = df.pivot_table(index="dow", columns="hour",
                               values="PnL", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])
        return pivot

    def _count_heatmap(self, df):
        pivot = df.pivot_table(index="dow", columns="hour",
                               values="PnL", aggfunc="size", fill_value=0)
        pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])
        return pivot

    def _cum_pnl_by_session(self, df):
        result = {}
        df_s = df.sort_values("_ts")
        for session, grp in df_s.groupby("session"):
            grp = grp.sort_values("_ts")
            result[session] = {
                "x": grp["_ts"].values,
                "y": grp["PnL"].cumsum().values,
            }
        return result

    @staticmethod
    def _bar_colors(values):
        return [VbtTradeAnalytics._GREEN if v >= 0 else VbtTradeAnalytics._RED for v in values]

    # ------------------------------------------------------------------
    # Build one full set of traces for a given perspective df
    # ------------------------------------------------------------------

    def _build_traces(self, df, perspective: str):
        """
        Returns a list of dicts:
            { "trace": go.BaseTraceType, "row": int, "col": int }
        All traces carry a custom meta tag so we can toggle visibility.
        """
        traces = []
        tag = perspective  # "entry" or "exit"

        def bar(x, y, name):
            colors = self._bar_colors(y)
            return go.Bar(x=list(x), y=list(y), marker_color=colors, marker_line_width=0, name=name, showlegend=False,
                          visible=(tag == "entry"), meta=tag)

        # ① Hour
        h = self._pnl_by(df, "hour").sort_values("hour")
        traces.append({"trace": bar(h["hour"], h["PnL"], "Hour PnL"), "row": 1, "col": 1})

        # ② Day of week
        d = self._pnl_by(df, "dow")
        d["dow"] = pd.Categorical(d["dow"], categories=DOW_ORDER, ordered=True)
        d = d.sort_values("dow")
        traces.append({"trace": bar(d["dow"], d["PnL"], "DoW PnL"), "row": 1, "col": 2})

        # ③ Session
        s = self._pnl_by(df, "session")
        s["session"] = pd.Categorical(s["session"], categories=SESSION_ORDER, ordered=True)
        s = s.sort_values("session")
        traces.append({"trace": bar(s["session"], s["PnL"], "Session PnL"), "row": 2, "col": 1})

        # ④ Month
        m = self._pnl_by(df, "month", sort_col="month_n")
        traces.append({"trace": bar(m["month"], m["PnL"], "Month PnL"), "row": 2, "col": 2})

        # ⑤ Quarter
        q = self._pnl_by(df, "quarter").sort_values("quarter")
        traces.append({"trace": bar(q["quarter"], q["PnL"], "Quarter PnL"), "row": 3, "col": 1})

        # ⑥ Duration bucket
        db = self._pnl_by(df, "duration_bucket")
        db["duration_bucket"] = pd.Categorical(db["duration_bucket"],
                                               categories=DURATION_LABELS, ordered=True)
        db = db.sort_values("duration_bucket").dropna(subset=["duration_bucket"])
        traces.append({"trace": bar(db["duration_bucket"], db["PnL"], "Duration PnL"), "row": 3, "col": 2})

        # ⑦ Direction
        dir_df = self._pnl_by(df, "direction")
        traces.append({"trace": bar(dir_df["direction"], dir_df["PnL"], "Dir PnL"), "row": 4, "col": 1})

        # ⑧ Count heatmap
        cnt = self._count_heatmap(df)
        traces.append({"trace": go.Heatmap(
            z=cnt.values,
            x=[str(c) for c in cnt.columns],
            y=list(cnt.index),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(x=1.02, len=0.20, y=0.27, thickness=10,
                          tickfont=dict(color=self._SUB, size=9)),
            name="Trade Count",
            visible=(tag == "entry"),
            meta=tag,
        ), "row": 4, "col": 2})

        # ⑨ PnL heatmap
        pnl_hm = self._pnl_heatmap(df)
        max_abs = max(abs(pnl_hm.values.max()), abs(pnl_hm.values.min()), 1) if pnl_hm.size > 0 else 1
        traces.append({"trace": go.Heatmap(
            z=pnl_hm.values,
            x=[str(c) for c in pnl_hm.columns],
            y=list(pnl_hm.index),
            colorscale=[[0.0, self._RED], [0.5, self._PANEL], [1.0, self._GREEN]],
            zmid=0, zmin=-max_abs, zmax=max_abs,
            showscale=True,
            colorbar=dict(x=0.46, len=0.20, y=0.06, thickness=10,
                          tickfont=dict(color=self._SUB, size=9)),
            name="PnL Heatmap",
            visible=(tag == "entry"),
            meta=tag,
        ), "row": 5, "col": 1})

        # ⑩ Cumulative PnL by session
        cum = self._cum_pnl_by_session(df)
        for session, data in cum.items():
            traces.append({"trace": go.Scatter(
                x=data["x"], y=data["y"],
                mode="lines",
                name=session if tag == "entry" else f"{session} ",   # unique legend key
                line=dict(color=self.SESSION_COLORS.get(session, self._TEXT), width=1.5),
                showlegend=(tag == "entry"),
                visible=(tag == "entry"),
                meta=tag,
            ), "row": 5, "col": 2})

        return traces

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def show_dashboard(self, output_html: str = None, height: int = 2300):
        fig = make_subplots(
            rows=7, cols=2,
            subplot_titles=[
                "① PnL by Hour of Day",        "② PnL by Day of Week",
                "③ PnL by Market Session",      "④ PnL by Month",
                "⑤ PnL by Quarter",             "⑥ PnL by Trade Duration",
                "⑦ PnL by Direction",           "⑧ Trade Count Heatmap (Hour × Day)",
                "⑨ PnL Heatmap (Hour × Day)",   "⑩ Cumulative PnL by Session",
                "⑪ Hour Breakdown & Verdict",   "⑫ Hour-Filter Impact Scenarios",
                "⑬ Hour-Filter Recommendation",
            ],
            vertical_spacing=0.035,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "xy"},      {"type": "xy"}],
                [{"type": "xy"},      {"type": "xy"}],
                [{"type": "xy"},      {"type": "xy"}],
                [{"type": "xy"},      {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "xy"}],
                [{"type": "table"},   {"type": "table"}],
                [{"type": "table", "colspan": 2}, None],
            ],
        )

        # Build both sets of traces
        entry_traces = self._build_traces(self._entry_df, "entry")
        exit_traces  = self._build_traces(self._exit_df,  "exit")

        all_traces = entry_traces + exit_traces
        for t in all_traces:
            fig.add_trace(t["trace"], row=t["row"], col=t["col"])

        # Hour-filter analysis tables (entry perspective — always visible)
        fig.add_trace(self._build_filter_analysis_table(self._entry_df), row=6, col=1)
        fig.add_trace(self._build_scenario_table(self._entry_df), row=6, col=2)

        # Recommendation summary
        analysis = self._hour_filter_analysis(self._entry_df)
        rec_lines = self._format_recommendation(analysis)
        fig.add_trace(go.Table(
            header=dict(
                values=["Hour-Filter Recommendation"],
                fill_color=self._PANEL,
                font=dict(color=self._BLUE, size=13, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="left",
            ),
            cells=dict(
                values=[rec_lines],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="left",
                height=25,
            ),
        ), row=7, col=1)

        # ------------------------------------------------------------------
        # Visibility masks for the dropdown
        # entry traces first, then exit traces — same length each side
        # ------------------------------------------------------------------
        n_entry = len(entry_traces)
        n_exit  = len(exit_traces)
        n_tables = 3  # hour breakdown, scenarios, recommendation — always visible

        entry_vis = [True]  * n_entry + [False] * n_exit + [True] * n_tables
        exit_vis  = [False] * n_entry + [True]  * n_exit + [True] * n_tables

        # ------------------------------------------------------------------
        # Dropdown button
        # ------------------------------------------------------------------
        dropdown = dict(
            buttons=[
                dict(
                    label="📥  Entry Perspective",
                    method="update",
                    args=[
                        {"visible": entry_vis},
                        {"title": {"text": (
                            "<b>Trade Analytics Dashboard</b>"
                            "  —  <span style='color:#58a6ff'>Entry Perspective</span>"
                            "  <span style='color:#8b949e; font-size:13px'>"
                            " | When I enter during X, how do I perform?</span>"
                        )}}
                    ],
                ),
                dict(
                    label="📤  Exit Perspective",
                    method="update",
                    args=[
                        {"visible": exit_vis},
                        {"title": {"text": (
                            "<b>Trade Analytics Dashboard</b>"
                            "  —  <span style='color:#3fb950'>Exit Perspective</span>"
                            "  <span style='color:#8b949e; font-size:13px'>"
                            " | When I close during X, how do I perform?</span>"
                        )}}
                    ],
                ),
            ],
            direction="down",
            showactive=True,
            bgcolor=self._PANEL,
            bordercolor=self._BORDER,
            borderwidth=1,
            font=dict(color=self._TEXT, size=12,
                      family="'Courier New', monospace"),
            x=0.0, xanchor="left",
            y=1.055, yanchor="top",
        )

        # ------------------------------------------------------------------
        # Global layout
        # ------------------------------------------------------------------
        fig.update_layout(
            height=height,
            title=dict(
                text=(
                    "<b>Trade Analytics Dashboard</b>"
                    "  —  <span style='color:#58a6ff'>Entry Perspective</span>"
                    "  <span style='color:#8b949e; font-size:13px'>"
                    " | When I enter during X, how do I perform?</span>"
                ),
                font=dict(family="'Courier New', monospace", size=18, color=self._TEXT),
                x=0.5, xanchor="center", y=0.99,
            ),
            updatemenus=[dropdown],
            paper_bgcolor=self._BG,
            plot_bgcolor=self._PANEL,
            font=dict(family="'Courier New', monospace", color=self._TEXT, size=11),
            legend=dict(
                bgcolor=self._PANEL,
                bordercolor=self._BORDER,
                borderwidth=1,
                font=dict(size=10),
                x=1.04, y=0.08,
                title=dict(text="Session", font=dict(size=10, color=self._SUB)),
            ),
            margin=dict(l=60, r=100, t=100, b=40),
        )

        axis_style = dict(
            gridcolor=self._BORDER,
            zerolinecolor=self._BORDER,
            tickfont=dict(size=9, color=self._SUB),
            showgrid=True,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        for ann in fig.layout.annotations:
            ann.font.update(size=12, color=self._SUB,
                            family="'Courier New', monospace")

        if output_html:
            fig.write_html(output_html)
            print(f"Dashboard saved → {output_html}")
        else:
            fig.show()

        return fig
