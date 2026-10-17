"""
TemporalPerformanceAnalyzer - conditional strategy performance sliced by the TIME axis.

The time/session sibling of ``RegimePerformanceAnalyzer``: both partition strategy performance into conditional cells
and report per-cell diagnostics, differing only in the conditioning axis. Here the condition is *intrinsic* to the data
— session, hour, day-of-week, month, quarter, duration and direction all fall out of the trade timestamps, so nothing
extra is supplied. (``RegimePerformanceAnalyzer`` takes an *extrinsic* regime label instead.) It derives time/context
insights vectorbt does not provide out of the box.

Dual-mode, like its sibling:
    from okmich_quant_research.backtesting.temporal_performance_analyzer import TemporalPerformanceAnalyzer

    # Backtest mode — from a vectorbt portfolio / its trades
    ta = TemporalPerformanceAnalyzer(pf.trades.records_readable)   # or .from_portfolio(pf)
    ta.show_dashboard()

    # Alpha-hunting mode — from a values DataFrame + a signal function (no backtest needed)
    ta = TemporalPerformanceAnalyzer.from_signal(df, signal_fn, close_col="close")
    ta.show_dashboard(output_html="dashboard.html")

Clock: every time dimension is read in the DATA's timezone. A tz-aware index is used as it is; naive timestamps are
localised with ``source_tz``. Hour / day-of-week / month are that zone's wall clock. Sessions are not fixed UTC hours:
each session opens at a wall-clock time on its own market's clock (Tokyo, London, New York), so the boundaries follow
each centre's daylight-saving switches. Custom sessions (``sessions=``) are read in the data's timezone unless a
``SessionOpen`` names its own.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Session / duration helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionOpen:
    """A session opens at ``hour:minute`` on the wall clock of ``tz`` and runs until the next session opens.

    ``tz=None`` means the data's own timezone. A timestamp belongs to the session whose most recent opening is the
    latest, so the sessions always partition the day with no gaps and no overlaps.
    """

    name: str
    hour: int
    minute: int = 0
    tz: str | None = None

    def __post_init__(self):
        if not (0 <= self.hour < 24 and 0 <= self.minute < 60):
            raise ValueError(f"SessionOpen {self.name!r}: hour must be in [0, 24) and minute in [0, 60)")


# FX centres on their own clocks. In northern winter (no DST anywhere) these are exactly the old UTC table:
# Asian 00-08, London 08-13, NY-London overlap 13-16, New York 16-21, Off-hours 21-24 UTC. In summer every European
# and US boundary moves one hour earlier in UTC, and in the weeks when US and UK DST disagree the overlap stretches.
DEFAULT_SESSIONS = (
    SessionOpen("Asian", 9, tz="Asia/Tokyo"),
    SessionOpen("London", 8, tz="Europe/London"),
    SessionOpen("NY–London OL", 8, tz="America/New_York"),
    SessionOpen("New York", 16, tz="Europe/London"),
    SessionOpen("Off-hours", 16, tz="America/New_York"),
)
SESSION_UNKNOWN = "Unknown"

DURATION_LABELS = ["Scalp (<1h)", "Intraday (1–8h)", "Swing (8h–3d)", "Position (>3d)", "Unknown"]

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SESSION_ORDER = [s.name for s in DEFAULT_SESSIONS] + [SESSION_UNKNOWN]

# The pre-DST table, kept only for the deprecated ``_get_session``.
_LEGACY_UTC_RANGES = {"Asian": (0, 8), "London": (8, 13), "NY–London OL": (13, 16), "New York": (16, 21),
                      "Off-hours": (21, 24)}


def assign_sessions(timestamps, sessions: Sequence[SessionOpen] = DEFAULT_SESSIONS, data_tz=None) -> np.ndarray:
    """Session name per timestamp. ``timestamps`` must be tz-aware (NaT maps to ``SESSION_UNKNOWN``).

    ``data_tz`` resolves sessions declared with ``tz=None``; it defaults to the timestamps' own timezone.
    """
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        raise ValueError("assign_sessions needs tz-aware timestamps; localise them to the data's timezone first")
    if not sessions:
        raise ValueError("sessions must contain at least one SessionOpen")
    data_tz = data_tz if data_tz is not None else ts.tz
    ts_utc = ts.tz_convert("UTC")
    n = len(ts_utc)
    elapsed = np.full((len(sessions), n), np.inf)
    for j, session in enumerate(sessions):
        tz = session.tz if session.tz is not None else data_tz
        wall = ts_utc.tz_convert(tz).tz_localize(None)
        opened = wall.normalize() + pd.Timedelta(hours=session.hour, minutes=session.minute)
        opened = opened.where(opened <= wall, opened - pd.Timedelta(days=1))
        # A custom opening inside a DST gap or fold resolves to the first valid instant; the default openings never do.
        opened_utc = opened.tz_localize(tz, ambiguous=np.ones(n, dtype=bool), nonexistent="shift_forward")
        seconds = (ts_utc - opened_utc.tz_convert("UTC")).total_seconds().to_numpy()
        seconds = np.where(seconds < 0, seconds + 86_400, seconds)
        elapsed[j] = np.where(np.isnan(seconds), np.inf, seconds)
    names = np.array([s.name for s in sessions] + [SESSION_UNKNOWN], dtype=object)
    pick = np.argmin(elapsed, axis=0)
    pick[np.isinf(elapsed.min(axis=0)) | ts_utc.isna()] = len(sessions)
    return names[pick]


def _get_session(hour: float) -> str:
    """Deprecated: maps a UTC hour to the fixed, DST-blind table. Use ``assign_sessions`` on tz-aware timestamps."""
    warnings.warn("_get_session uses fixed UTC hours and ignores daylight saving; use assign_sessions",
                  DeprecationWarning, stacklevel=2)
    if pd.isna(hour):
        return SESSION_UNKNOWN
    for name, (start, end) in _LEGACY_UTC_RANGES.items():
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

class TemporalPerformanceAnalyzer:
    """
    Accepts pf.trades.records_readable (a pandas DataFrame) and produces time/context dimensional analysis with a single
    Plotly dashboard. See ``from_signal`` / ``from_portfolio`` for the other input modes.

    The dashboard has an Entry / Exit perspective toggle — letting you ask:
      - Entry view : "When I enter during X, how do I perform?"
      - Exit view  : "When I close during X, how do I perform?"

    Required columns (vectorbt ≥ 0.25 naming):
        'Entry Timestamp', 'Exit Timestamp', 'PnL'
    Older vectorbt outputs using 'Entry Index' / 'Exit Index' are auto-renamed.
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

    def __init__(self, trades_df: pd.DataFrame, source_tz: str | None = None,
                 sessions: Sequence[SessionOpen] | None = None):
        """
        Parameters
        ----------
        trades_df : DataFrame from pf.trades.records_readable.
        source_tz : IANA timezone of NAIVE timestamps (e.g. "America/New_York"). Tz-aware timestamps already carry
                    their zone and are used as they are; passing a source_tz that disagrees with them raises. Naive
                    timestamps without a source_tz are assumed UTC, with a warning.
        sessions : session openings (``SessionOpen``); defaults to ``DEFAULT_SESSIONS`` (Tokyo / London / New York
                   on their own DST-aware clocks). A ``SessionOpen`` with ``tz=None`` is read in the data's timezone.
        """
        self.raw = trades_df.copy()
        self._source_tz = source_tz
        self._sessions = tuple(sessions) if sessions is not None else DEFAULT_SESSIONS
        if not self._sessions:
            raise ValueError("sessions must contain at least one SessionOpen")
        names = [s.name for s in self._sessions]
        if len(set(names)) != len(names) or SESSION_UNKNOWN in names:
            raise ValueError(f"session names must be unique and not {SESSION_UNKNOWN!r}: {names}")
        self._session_order = names + [SESSION_UNKNOWN]
        self.tz = None  # the data's timezone, resolved in _prepare
        self._entry_df, self._exit_df = self._prepare(trades_df)

    # ------------------------------------------------------------------
    # Alternate constructors (dual-mode ingestion)
    # ------------------------------------------------------------------

    @classmethod
    def from_portfolio(cls, portfolio, source_tz: str | None = None,
                       sessions: Sequence[SessionOpen] | None = None) -> "TemporalPerformanceAnalyzer":
        """Backtest mode: build from a vectorbt ``Portfolio`` via ``portfolio.trades.records_readable``."""
        return cls(portfolio.trades.records_readable, source_tz=source_tz, sessions=sessions)

    @classmethod
    def from_signal(cls, data: pd.DataFrame, signal_fn, source_tz: str | None = None, close_col: str = "close",
                    sessions: Sequence[SessionOpen] | None = None, **vbt_kwargs) -> "TemporalPerformanceAnalyzer":
        """Alpha-hunting mode: build trades from a values DataFrame and a signal function.

        ``signal_fn(data)`` returns a signed position series ({-1, 0, +1}); it is run through
        vectorbt (see ``signal_adapter.signal_to_portfolio``) and the resulting trades are
        analysed. Extra ``vbt_kwargs`` (``fees``, ``slippage``, ``init_cash``, ``freq`` ...)
        pass through to the portfolio build. The time condition (session/hour/...) is intrinsic
        to the trade timestamps, so no label is supplied. A tz-aware ``data.index`` carries its
        timezone through to the trades.
        """
        from .signal_adapter import signal_to_portfolio
        pf = signal_to_portfolio(data, signal_fn, close_col=close_col, **vbt_kwargs)
        return cls.from_portfolio(pf, source_tz=source_tz, sessions=sessions)

    # ------------------------------------------------------------------
    # Data preparation — returns (entry_df, exit_df)
    # ------------------------------------------------------------------

    _REQUIRED_COLUMNS = {"Entry Timestamp", "Exit Timestamp", "PnL"}
    _LEGACY_RENAME = {"Entry Index": "Entry Timestamp", "Exit Index": "Exit Timestamp"}

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        legacy = {old: new for old, new in self._LEGACY_RENAME.items() if old in df.columns and new not in df.columns}
        if legacy:
            df = df.rename(columns=legacy)

        missing = self._REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"trades_df is missing required columns: {sorted(missing)}. "
                             f"Expected columns from pf.trades.records_readable: {sorted(self._REQUIRED_COLUMNS)}")

        df["Entry Timestamp"] = self._localise(pd.to_datetime(df["Entry Timestamp"]), "Entry Timestamp")
        df["Exit Timestamp"]  = self._localise(pd.to_datetime(df["Exit Timestamp"]), "Exit Timestamp")

        df["duration_min"] = (df["Exit Timestamp"] - df["Entry Timestamp"]).dt.total_seconds() / 60
        df["duration_bucket"] = df["duration_min"].map(_get_duration_bucket)

        if "Direction" in df.columns:
            direction = df["Direction"].astype("string").str.strip().fillna(SESSION_UNKNOWN)
            df["direction"] = direction.mask(direction == "", SESSION_UNKNOWN)
        else:
            df["direction"] = SESSION_UNKNOWN

        entry_df = self._add_time_dims(df, "Entry Timestamp")
        exit_df  = self._add_time_dims(df, "Exit Timestamp")

        return entry_df, exit_df

    def _localise(self, ts: pd.Series, col: str) -> pd.Series:
        """Make timestamps tz-aware in the DATA's timezone and record that zone in ``self.tz``."""
        if ts.dt.tz is not None:
            if self._source_tz is not None:
                relabelled = ts.dt.tz_convert(self._source_tz).dt.tz_localize(None)
                if not relabelled.equals(ts.dt.tz_localize(None)):
                    raise ValueError(f"'{col}' is tz-aware ({ts.dt.tz}) but source_tz={self._source_tz!r} disagrees. "
                                     "source_tz only labels naive timestamps; convert the index instead.")
            self.tz = ts.dt.tz
            return ts
        tz = self._source_tz
        if tz is None:
            if self.tz is None and len(ts):
                warnings.warn("Trade timestamps are naive and no source_tz was given: assuming UTC. Sessions and "
                              "hour-of-day are wrong if the data is on another clock; pass source_tz or a tz-aware "
                              "index.", UserWarning, stacklevel=4)
            tz = "UTC"
        self.tz = self.tz if self.tz is not None else tz
        return ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")

    def _add_time_dims(self, df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
        df = df.copy()
        ts_local = df[idx_col]
        # hour/dow/month are the data's own wall clock, so analytics align with whatever clock the live system reads
        # (typically broker-local). Sessions open on each market's own clock (see assign_sessions).
        df["_ts"] = ts_local
        df["hour"] = ts_local.dt.hour
        df["dow"] = ts_local.dt.day_name()
        df["month"] = ts_local.dt.month_name()
        df["month_n"] = ts_local.dt.month
        df["quarter"] = ts_local.dt.quarter.map(lambda q: f"Q{q}")
        df["session"] = assign_sessions(ts_local, self._sessions, data_tz=self.tz) if len(df) else []
        return df

    # ------------------------------------------------------------------
    # Per-hour / per-session breakdown (descriptive)
    # ------------------------------------------------------------------

    @staticmethod
    def _hour_stats(df: pd.DataFrame) -> list:
        """
        Per-hour trade statistics, sorted by hour: [{hour, n_trades, total_pnl, mean_pnl, win_rate}, ...].

        Descriptive only. Blocking the hours that lost on this same sample raises its PnL by construction, so no
        blocked-hours filter is suggested; an hour gate must be chosen on one sample and judged on another.
        """
        hour_stats = []
        for hour, grp in df.groupby("hour")["PnL"]:
            hour_stats.append({
                "hour": int(hour),
                "n_trades": len(grp),
                "total_pnl": grp.sum(),
                "mean_pnl": grp.mean(),
                "win_rate": (grp > 0).mean() * 100,
            })
        hour_stats.sort(key=lambda x: x["hour"])
        return hour_stats

    @staticmethod
    def _session_stats(df: pd.DataFrame) -> list:
        """Per-session trade statistics: [{session, n_trades, total_pnl, win_rate}, ...]."""
        session_stats = []
        for session, grp in df.groupby("session")["PnL"]:
            session_stats.append({
                "session": session,
                "n_trades": len(grp),
                "total_pnl": grp.sum(),
                "win_rate": (grp > 0).mean() * 100,
            })
        return session_stats

    def _build_hour_table(self, df: pd.DataFrame) -> go.Table:
        """Build a Plotly Table trace of the per-hour breakdown."""
        hour_stats = self._hour_stats(df)
        return go.Table(
            header=dict(
                values=["Hour", "Trades", "Total PnL", "Mean PnL", "Win Rate"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[
                    [str(h["hour"]) for h in hour_stats],
                    [str(h["n_trades"]) for h in hour_stats],
                    [f"{h['total_pnl']:+.2f}" for h in hour_stats],
                    [f"{h['mean_pnl']:+.4f}" for h in hour_stats],
                    [f"{h['win_rate']:.0f}%" for h in hour_stats],
                ],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

    def _build_session_table(self, df: pd.DataFrame) -> go.Table:
        """Build a Plotly Table trace of the per-session breakdown, in session order."""
        order = {name: i for i, name in enumerate(self._session_order)}
        session_stats = sorted(self._session_stats(df), key=lambda s: order.get(s["session"], len(order)))
        return go.Table(
            header=dict(
                values=["Session", "Trades", "Total PnL", "Win Rate"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[
                    [str(s["session"]) for s in session_stats],
                    [str(s["n_trades"]) for s in session_stats],
                    [f"{s['total_pnl']:+.2f}" for s in session_stats],
                    [f"{s['win_rate']:.0f}%" for s in session_stats],
                ],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

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
        return [TemporalPerformanceAnalyzer._GREEN if v >= 0 else TemporalPerformanceAnalyzer._RED for v in values]

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
        s["session"] = pd.Categorical(s["session"], categories=self._session_order, ordered=True)
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
            colorbar=dict(x=1.02, len=0.13, y=0.41, thickness=10,  # centred on row 4 of the 6-row grid
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
            colorbar=dict(x=0.46, len=0.13, y=0.24, thickness=10,  # centred on row 5 of the 6-row grid
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

    def show_dashboard(self, output_html: str = None, height: int = 2000):
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=[
                f"① PnL by Hour of Day ({self.tz})", "② PnL by Day of Week",
                "③ PnL by Market Session",      "④ PnL by Month",
                "⑤ PnL by Quarter",             "⑥ PnL by Trade Duration",
                "⑦ PnL by Direction",           "⑧ Trade Count Heatmap (Hour × Day)",
                "⑨ PnL Heatmap (Hour × Day)",   "⑩ Cumulative PnL by Session",
                "⑪ Hour Breakdown (entry)",     "⑫ Session Breakdown (entry)",
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
            ],
        )

        # Build both sets of traces
        entry_traces = self._build_traces(self._entry_df, "entry")
        exit_traces  = self._build_traces(self._exit_df,  "exit")

        all_traces = entry_traces + exit_traces
        for t in all_traces:
            fig.add_trace(t["trace"], row=t["row"], col=t["col"])

        # Breakdown tables (entry perspective — always visible)
        fig.add_trace(self._build_hour_table(self._entry_df), row=6, col=1)
        fig.add_trace(self._build_session_table(self._entry_df), row=6, col=2)

        # ------------------------------------------------------------------
        # Visibility masks for the dropdown
        # entry traces first, then exit traces — same length each side
        # ------------------------------------------------------------------
        n_entry = len(entry_traces)
        n_exit  = len(exit_traces)
        n_tables = 2  # hour and session breakdowns — always visible

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
            print(f"Dashboard saved -> {output_html}")
        else:
            fig.show()

        return fig
