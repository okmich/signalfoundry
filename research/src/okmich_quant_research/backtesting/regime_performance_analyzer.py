from typing import Dict, List, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RegimePerformanceAnalyzer:
    """Analyse a VectorBT portfolio split by market regime labels.

    Parameters
    ----------
    portfolio:
        A ``vbt.Portfolio`` instance produced by any backtesting run.
    regime_labels:
        A ``pd.Series`` with a ``DatetimeIndex`` and integer regime IDs.
        Minor gaps are forward-filled to align with the portfolio index.
    regime_names:
        Optional mapping from integer regime ID to display name.
    annualization_factor:
        Bars per year used for Sharpe / Sortino calculation.
        Typical values: 252 (daily), 8760 (hourly), 52560 (5-min).
    """

    _BG = "#0d1117"
    _PANEL = "#161b22"
    _BORDER = "#30363d"
    _TEXT = "#e6edf3"
    _SUB = "#8b949e"
    _GREEN = "#3fb950"
    _RED = "#f85149"
    _BLUE = "#58a6ff"
    _ORANGE = "#d29922"
    _UNKNOWN_REGIME = "Unknown"

    _REGIME_COLORS = [
        "#2E86AB",
        "#F77F00",
        "#06A77D",
        "#D62828",
        "#8338EC",
        "#FB5607",
        "#3A86FF",
        "#FFBE0B",
        "#FF006E",
        "#8AC926",
    ]

    def __init__(
        self,
        portfolio: vbt.Portfolio,
        regime_labels: pd.Series,
        regime_names: Optional[Dict[int, str]] = None,
        annualization_factor: int = 252,
    ) -> None:
        self.portfolio = portfolio
        self.regime_names = regime_names or {}
        self.annualization_factor = annualization_factor

        pf_index = portfolio.wrapper.index
        if len(regime_labels.index.intersection(pf_index)) == 0:
            raise ValueError("regime_labels index has no overlap with the portfolio index.")
        aligned_labels = regime_labels.reindex(pf_index, method="ffill")
        self._labels: pd.Series = aligned_labels.where(aligned_labels.notna(), self._UNKNOWN_REGIME)
        known_regimes = sorted(
            self._labels[self._labels != self._UNKNOWN_REGIME].unique().tolist(),
            key=lambda x: str(x),
        )
        include_unknown = bool((self._labels == self._UNKNOWN_REGIME).any())
        self._regimes: List[Union[int, str]] = known_regimes + ([self._UNKNOWN_REGIME] if include_unknown else [])
        self._returns: pd.Series = self._ensure_single_series(portfolio.returns(), "portfolio.returns()")
        self._trades: pd.DataFrame = portfolio.trades.records_readable
        duration_col = self._duration_column(self._trades)
        duration_series = self._trades[duration_col].dropna() if duration_col and duration_col in self._trades.columns else pd.Series(dtype=float)
        self._duration_is_timedelta = bool(
            duration_col
            and duration_col in self._trades.columns
            and (
                pd.api.types.is_timedelta64_dtype(self._trades[duration_col])
                or (len(duration_series) > 0 and isinstance(duration_series.iloc[0], pd.Timedelta))
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _display_name(self, regime_id: Union[int, str]) -> str:
        return self.regime_names.get(regime_id, str(regime_id))

    def _regime_color(self, idx: int) -> str:
        return self._REGIME_COLORS[idx % len(self._REGIME_COLORS)]

    @staticmethod
    def _safe_float(value: Union[float, int, np.number]) -> float:
        if pd.isna(value):
            return np.nan
        return float(value)

    @staticmethod
    def _ensure_single_series(data: Union[pd.Series, pd.DataFrame], source_name: str) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0]
            raise ValueError(
                f"{source_name} returned {data.shape[1]} columns. "
                "RegimePerformanceAnalyzer supports single-column portfolios only."
            )
        raise TypeError(f"{source_name} must be a pandas Series or 1-column DataFrame.")

    @staticmethod
    def _fmt_pct(value: float, decimals: int = 2) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value * 100:.{decimals}f}%"

    @staticmethod
    def _fmt_num(value: float, decimals: int = 3) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.{decimals}f}"

    @staticmethod
    def _fmt_signed(value: float, decimals: int = 3) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:+.{decimals}f}"

    @staticmethod
    def _fmt_duration(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.2f}"

    @staticmethod
    def _entry_timestamp_column(trades: pd.DataFrame) -> str:
        for col in ("Entry Timestamp", "Entry Index"):
            if col in trades.columns:
                return col
        raise ValueError("Trades DataFrame must contain 'Entry Timestamp' or 'Entry Index'.")

    @staticmethod
    def _duration_column(trades: pd.DataFrame) -> Optional[str]:
        for col in ("Duration", "Duration Bars"):
            if col in trades.columns:
                return col
        return None

    @staticmethod
    def _compute_avg_duration(trades: pd.DataFrame, duration_col: Optional[str]) -> float:
        if not duration_col or duration_col not in trades.columns:
            return np.nan
        series = trades[duration_col].dropna()
        if len(series) == 0:
            return np.nan
        if pd.api.types.is_timedelta64_dtype(series) or isinstance(series.iloc[0], pd.Timedelta):
            td_series = pd.to_timedelta(series, errors="coerce").dropna()
            if len(td_series) == 0:
                return np.nan
            return float(td_series.dt.total_seconds().mean() / 3600.0)
        return float(series.mean())

    def _compute_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return np.nan
        return float(returns.mean() / returns.std() * np.sqrt(self.annualization_factor))

    def _compute_sortino(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return np.nan
        downside = returns[returns < 0]
        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return np.nan
        return float(returns.mean() / downside_std * np.sqrt(self.annualization_factor))

    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return np.nan
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        return float(dd.min())

    def _returns_by_regime(self) -> Dict[Union[int, str], pd.Series]:
        result: Dict[Union[int, str], pd.Series] = {}
        for rid in self._regimes:
            mask = self._labels == rid
            result[rid] = self._returns[mask]
        return result

    def _trades_by_regime(self) -> Dict[Union[int, str], pd.DataFrame]:
        """Assign each trade to the regime at its entry bar."""
        if self._trades.empty:
            return {rid: pd.DataFrame() for rid in self._regimes}

        entry_col = self._entry_timestamp_column(self._trades)
        entry_times = pd.to_datetime(self._trades[entry_col])
        entry_regimes = self._labels.reindex(entry_times, method="ffill")
        entry_regimes = entry_regimes.where(entry_regimes.notna(), self._UNKNOWN_REGIME)
        entry_regimes.index = self._trades.index

        result: Dict[Union[int, str], pd.DataFrame] = {}
        for rid in self._regimes:
            mask = entry_regimes == rid
            result[rid] = self._trades[mask]
        return result

    def _summary_for_regime(
        self, regime_name: str, return_stats: pd.DataFrame, trade_stats: pd.DataFrame, exposure: pd.DataFrame
    ) -> str:
        r = return_stats.loc[regime_name]
        t = trade_stats.loc[regime_name]
        e = exposure.loc[regime_name]
        score = 0.0
        score += self._safe_float(r["sharpe"]) if not pd.isna(r["sharpe"]) else -10.0
        score += self._safe_float(t["profit_factor"]) if not pd.isna(t["profit_factor"]) else 0.0
        score += self._safe_float(t["expectancy"]) if not pd.isna(t["expectancy"]) else -1.0
        score += self._safe_float(r["total_return"]) * 2 if not pd.isna(r["total_return"]) else -1.0
        penalty = 0.0
        if pd.notna(t["n_trades"]) and t["n_trades"] < 25:
            penalty -= 0.6
        if pd.notna(e["pct_time"]) and e["pct_time"] < 0.05:
            penalty -= 0.3
        return f"{regime_name}|{score + penalty:.6f}|{int(t['n_trades']) if pd.notna(t['n_trades']) else 0}"

    def _recommendation_lines(
        self, return_stats: pd.DataFrame, trade_stats: pd.DataFrame, exposure: pd.DataFrame
    ) -> List[str]:
        if len(return_stats) == 0:
            return ["No regimes available after alignment."]

        ranked = sorted(
            [self._summary_for_regime(name, return_stats, trade_stats, exposure) for name in return_stats.index],
            key=lambda x: float(x.split("|")[1]),
            reverse=True,
        )
        top_parts = ranked[0].split("|")
        worst_parts = ranked[-1].split("|")
        top_regime, top_score, top_n = top_parts[0], float(top_parts[1]), int(top_parts[2])
        worst_regime, worst_score, worst_n = worst_parts[0], float(worst_parts[1]), int(worst_parts[2])

        lines = [
            f"Best composite regime: {top_regime} (score={top_score:+.2f}, trades={top_n}).",
            f"Weakest composite regime: {worst_regime} (score={worst_score:+.2f}, trades={worst_n}).",
        ]

        weak_sample = trade_stats[trade_stats["n_trades"] < 25]
        if len(weak_sample) > 0:
            lines.append(
                f"Sample-size warning: {len(weak_sample)} regime(s) have <25 trades; treat edge estimates as unstable."
            )
        else:
            lines.append("Sample-size check: all regimes have >=25 trades.")

        negative_expectancy = trade_stats[trade_stats["expectancy"] < 0]
        if len(negative_expectancy) > 0:
            names = ", ".join(negative_expectancy.index.tolist())
            lines.append(f"Execution warning: negative per-trade expectancy in {names}.")
        else:
            lines.append("Execution check: no regime with negative expectancy.")

        high_dd = return_stats[return_stats["max_drawdown"] < -0.20]
        if len(high_dd) > 0:
            names = ", ".join(high_dd.index.tolist())
            lines.append(f"Risk warning: max drawdown worse than -20% in {names}.")
        else:
            lines.append("Risk check: no regime breaches -20% max drawdown.")
        return lines

    def _build_return_table(self, return_stats: pd.DataFrame) -> go.Table:
        return go.Table(
            header=dict(
                values=["Regime", "Bars", "Total Ret", "Ann Ret", "Vol", "Sharpe", "Sortino", "Max DD"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[
                    return_stats.index.tolist(),
                    [int(v) for v in return_stats["n_bars"].fillna(0)],
                    [self._fmt_pct(v) for v in return_stats["total_return"]],
                    [self._fmt_pct(v) for v in return_stats["ann_return"]],
                    [self._fmt_pct(v) for v in return_stats["volatility"]],
                    [self._fmt_num(v) for v in return_stats["sharpe"]],
                    [self._fmt_num(v) for v in return_stats["sortino"]],
                    [self._fmt_pct(v) for v in return_stats["max_drawdown"]],
                ],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

    def _build_trade_exposure_table(self, trade_stats: pd.DataFrame, exposure: pd.DataFrame) -> go.Table:
        joined = trade_stats.join(exposure, how="left", rsuffix="_exp")
        duration_header = "Avg Dur (hrs)" if self._duration_is_timedelta else "Avg Dur (bars)"
        return go.Table(
            header=dict(
                values=["Regime", "Trades", "Win Rate", "PF", "Expectancy", duration_header, "Time %", "In Pos %"],
                fill_color=self._PANEL,
                font=dict(color=self._TEXT, size=11, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
            cells=dict(
                values=[
                    joined.index.tolist(),
                    [int(v) if pd.notna(v) else 0 for v in joined["n_trades"]],
                    [self._fmt_pct(v) for v in joined["win_rate"]],
                    [self._fmt_num(v) for v in joined["profit_factor"]],
                    [self._fmt_signed(v) for v in joined["expectancy"]],
                    [self._fmt_duration(v) for v in joined["avg_duration_bars"]],
                    [self._fmt_pct(v) for v in joined["pct_time"]],
                    [self._fmt_pct(v) for v in joined["pct_time_in_position"]],
                ],
                fill_color=self._BG,
                font=dict(color=self._TEXT, size=10, family="'Courier New', monospace"),
                line_color=self._BORDER,
                align="center",
            ),
        )

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def regime_return_stats(self) -> pd.DataFrame:
        """Per-regime return statistics.

        Returns a DataFrame indexed by regime display name with columns:
        ``n_bars``, ``total_return``, ``ann_return``, ``volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``.
        """
        rows = []
        returns_by_regime = self._returns_by_regime()
        for rid in self._regimes:
            rets = returns_by_regime[rid].dropna()
            n = len(rets)
            total_ret = float((1 + rets).prod() - 1) if n > 0 else np.nan
            ann_ret = float((1 + total_ret) ** (self.annualization_factor / max(n, 1)) - 1) if n > 0 else np.nan
            vol = float(rets.std() * np.sqrt(self.annualization_factor)) if n > 1 else np.nan
            rows.append(
                {
                    "regime": self._display_name(rid),
                    "n_bars": n,
                    "total_return": total_ret,
                    "ann_return": ann_ret,
                    "volatility": vol,
                    "sharpe": self._compute_sharpe(rets),
                    "sortino": self._compute_sortino(rets),
                    "max_drawdown": self._compute_max_drawdown(rets),
                }
            )
        return pd.DataFrame(rows).set_index("regime")

    def regime_trade_stats(self) -> pd.DataFrame:
        """Per-regime trade statistics.

        Returns a DataFrame with columns:
        ``n_trades``, ``win_rate``, ``avg_win``, ``avg_loss``,
        ``profit_factor``, ``expectancy``, ``avg_duration_bars``.
        """
        rows = []
        trades_by_regime = self._trades_by_regime()
        duration_col = self._duration_column(self._trades)

        for rid in self._regimes:
            trades = trades_by_regime[rid]
            n = len(trades)
            if n == 0:
                rows.append(
                    {
                        "regime": self._display_name(rid),
                        "n_trades": 0,
                        "win_rate": np.nan,
                        "avg_win": np.nan,
                        "avg_loss": np.nan,
                        "profit_factor": np.nan,
                        "expectancy": np.nan,
                        "avg_duration_bars": np.nan,
                    }
                )
                continue

            pnl = trades["PnL"] if "PnL" in trades.columns else pd.Series(dtype=float)
            winners = pnl[pnl > 0]
            losers = pnl[pnl < 0]
            gross_profit = winners.sum()
            gross_loss = abs(losers.sum())
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan
            avg_duration = self._compute_avg_duration(trades, duration_col)

            rows.append(
                {
                    "regime": self._display_name(rid),
                    "n_trades": n,
                    "win_rate": float(len(winners) / n) if n > 0 else np.nan,
                    "avg_win": self._safe_float(winners.mean()) if len(winners) > 0 else np.nan,
                    "avg_loss": self._safe_float(losers.mean()) if len(losers) > 0 else np.nan,
                    "profit_factor": profit_factor,
                    "expectancy": self._safe_float(pnl.mean()) if n > 0 else np.nan,
                    "avg_duration_bars": avg_duration,
                }
            )
        return pd.DataFrame(rows).set_index("regime")

    def regime_exposure(self) -> pd.DataFrame:
        """Per-regime time exposure.

        Returns a DataFrame with columns:
        ``n_bars``, ``pct_time``, ``n_bars_in_position``, ``pct_time_in_position``.
        """
        total_bars = len(self._labels)
        try:
            asset_value = self.portfolio.asset_value()
        except Exception:
            in_position = pd.Series(False, index=self.portfolio.wrapper.index)
        else:
            asset_series = self._ensure_single_series(asset_value, "portfolio.asset_value()")
            in_position = asset_series != 0

        rows = []
        for rid in self._regimes:
            mask = self._labels == rid
            n_bars = int(mask.sum())
            n_in_pos = int((mask & in_position).sum()) if len(in_position) == len(mask) else 0
            rows.append(
                {
                    "regime": self._display_name(rid),
                    "n_bars": n_bars,
                    "pct_time": float(n_bars / total_bars) if total_bars > 0 else np.nan,
                    "n_bars_in_position": n_in_pos,
                    "pct_time_in_position": float(n_in_pos / n_bars) if n_bars > 0 else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("regime")

    def generate_full_report(self) -> Dict[str, pd.DataFrame]:
        """Return all three DataFrames in a single dict."""
        return {
            "return_stats": self.regime_return_stats(),
            "trade_stats": self.regime_trade_stats(),
            "exposure": self.regime_exposure(),
        }

    # ------------------------------------------------------------------
    # Plotly dashboard
    # ------------------------------------------------------------------

    def show_dashboard(self, output_html: Optional[str] = None, height: int = 2200) -> go.Figure:
        """Interactive Plotly dashboard with regime return, trade, and exposure diagnostics."""
        return_stats = self.regime_return_stats()
        trade_stats = self.regime_trade_stats()
        exposure = self.regime_exposure()

        regime_order = [self._display_name(rid) for rid in self._regimes]
        regime_colors = {self._display_name(rid): self._regime_color(i) for i, rid in enumerate(self._regimes)}

        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=[
                "1) Total Return by Regime",
                "2) Sharpe and Sortino by Regime",
                "3) Win Rate and Profit Factor by Regime",
                "4) Exposure by Regime",
                "5) Cumulative Return by Regime",
                "6) Trade PnL Distribution by Regime",
                "7) Regime Return Metrics",
                "8) Regime Trade + Exposure Metrics",
                "9) Regime Recommendation",
            ],
            vertical_spacing=0.04,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "table"}, {"type": "table"}],
                [{"type": "table", "colspan": 2}, None],
            ],
        )

        fig.add_trace(
            go.Bar(
                x=regime_order,
                y=[v * 100 if pd.notna(v) else np.nan for v in return_stats["total_return"]],
                marker_color=[regime_colors[name] for name in regime_order],
                showlegend=False,
                name="Total Return (%)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=regime_order,
                y=[v if pd.notna(v) else np.nan for v in return_stats["sharpe"]],
                marker_color=[regime_colors[name] for name in regime_order],
                opacity=0.9,
                name="Sharpe",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_order,
                y=[v if pd.notna(v) else np.nan for v in return_stats["sortino"]],
                mode="lines+markers",
                line=dict(color=self._BLUE, width=2),
                marker=dict(size=8),
                name="Sortino",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=regime_order,
                y=[v * 100 if pd.notna(v) else np.nan for v in trade_stats["win_rate"]],
                marker_color=[regime_colors[name] for name in regime_order],
                opacity=0.9,
                name="Win Rate (%)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_order,
                y=[v if pd.notna(v) else np.nan for v in trade_stats["profit_factor"]],
                mode="lines+markers",
                line=dict(color=self._ORANGE, width=2),
                marker=dict(size=8),
                name="Profit Factor",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=regime_order,
                y=[v * 100 if pd.notna(v) else np.nan for v in exposure["pct_time"]],
                marker_color=[regime_colors[name] for name in regime_order],
                opacity=0.9,
                name="Time Exposure (%)",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_order,
                y=[v * 100 if pd.notna(v) else np.nan for v in exposure["pct_time_in_position"]],
                mode="lines+markers",
                line=dict(color=self._GREEN, width=2),
                marker=dict(size=8),
                name="In-Position (%)",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        returns_by_regime = self._returns_by_regime()
        for idx, rid in enumerate(self._regimes):
            regime_name = self._display_name(rid)
            rets = returns_by_regime[rid].dropna()
            if len(rets) == 0:
                continue
            cumulative = ((1 + rets).cumprod() - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=rets.index,
                    y=cumulative.values,
                    mode="lines",
                    name=regime_name,
                    line=dict(color=self._regime_color(idx), width=2),
                    showlegend=True,
                    legendgroup=regime_name,
                ),
                row=3,
                col=1,
            )

        trades_by_regime = self._trades_by_regime()
        for idx, rid in enumerate(self._regimes):
            regime_name = self._display_name(rid)
            trades = trades_by_regime[rid]
            pnl = trades["PnL"].dropna() if "PnL" in trades.columns else pd.Series(dtype=float)
            if len(pnl) == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=pnl.values,
                    x=[regime_name] * len(pnl),
                    name=regime_name,
                    marker_color=self._regime_color(idx),
                    boxmean=True,
                    showlegend=False,
                    legendgroup=regime_name,
                ),
                row=3,
                col=2,
            )

        fig.add_trace(self._build_return_table(return_stats), row=4, col=1)
        fig.add_trace(self._build_trade_exposure_table(trade_stats, exposure), row=4, col=2)

        rec_lines = self._recommendation_lines(return_stats, trade_stats, exposure)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Regime Recommendation Summary"],
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
                    height=24,
                ),
            ),
            row=5,
            col=1,
        )

        fig.update_layout(
            height=height,
            title=dict(
                text=(
                    "<b>Regime Performance Dashboard</b>"
                    "  <span style='color:#8b949e; font-size:13px'>"
                    "| Return, risk, trade quality, and exposure diagnostics</span>"
                ),
                font=dict(family="'Courier New', monospace", size=18, color=self._TEXT),
                x=0.5,
                xanchor="center",
                y=0.99,
            ),
            barmode="group",
            paper_bgcolor=self._BG,
            plot_bgcolor=self._PANEL,
            font=dict(family="'Courier New', monospace", color=self._TEXT, size=11),
            legend=dict(
                bgcolor=self._PANEL,
                bordercolor=self._BORDER,
                borderwidth=1,
                font=dict(size=10),
                x=1.02,
                y=0.43,
                title=dict(text="Regime", font=dict(size=10, color=self._SUB)),
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
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate (%) / PF", row=2, col=1)
        fig.update_yaxes(title_text="Exposure (%)", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
        fig.update_yaxes(title_text="PnL", row=3, col=2)

        for ann in fig.layout.annotations:
            ann.font.update(size=12, color=self._SUB, family="'Courier New', monospace")

        if output_html:
            fig.write_html(output_html)
            print(f"Dashboard saved -> {output_html}")
        else:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Matplotlib plots (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _bar_chart(self, ax, labels, values, title, ylabel, pct: bool = False) -> None:
        colors = [self._regime_color(i) for i in range(len(labels))]
        ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold", color="white")
        ax.set_ylabel(ylabel, fontsize=11, color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, color="#888")
        ax.set_facecolor("#1e1e1e")
        if pct:
            ax.set_ylim(0, 100)

    def plot_regime_summary(self, figsize=(14, 10), save_path: Optional[str] = None) -> plt.Figure:
        """2x2 bar chart summary: total return, Sharpe, win rate, and trade count."""
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.patch.set_facecolor("#121212")

            ret_stats = self.regime_return_stats()
            trd_stats = self.regime_trade_stats()
            names = ret_stats.index.tolist()

            self._bar_chart(
                axes[0, 0],
                names,
                [v * 100 if not np.isnan(v) else 0 for v in ret_stats["total_return"]],
                "Total Return by Regime",
                "Return (%)",
            )
            self._bar_chart(
                axes[0, 1],
                names,
                [v if not np.isnan(v) else 0 for v in ret_stats["sharpe"]],
                "Sharpe Ratio by Regime",
                "Sharpe Ratio",
            )
            self._bar_chart(
                axes[1, 0],
                names,
                [v * 100 if not np.isnan(v) else 0 for v in trd_stats["win_rate"]],
                "Win Rate by Regime",
                "Win Rate (%)",
                pct=True,
            )
            self._bar_chart(
                axes[1, 1],
                names,
                [v for v in trd_stats["n_trades"]],
                "Trade Count by Regime",
                "# Trades",
            )

            fig.suptitle("Regime Performance Summary", fontsize=16, fontweight="bold", color="white")
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig

    def plot_regime_returns(self, figsize=(14, 6), save_path: Optional[str] = None) -> plt.Figure:
        """One subplot per regime showing cumulative return over that regime's bars."""
        n_regimes = len(self._regimes)
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, n_regimes, figsize=figsize, squeeze=False)
            fig.patch.set_facecolor("#121212")

            returns_by_regime = self._returns_by_regime()
            for idx, rid in enumerate(self._regimes):
                ax = axes[0, idx]
                rets = returns_by_regime[rid].dropna()
                if len(rets) > 0:
                    cum = (1 + rets).cumprod()
                    ax.plot(range(len(cum)), cum.values, color=self._regime_color(idx), linewidth=1.5)
                    ax.fill_between(range(len(cum)), 1, cum.values, alpha=0.2, color=self._regime_color(idx))
                ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
                ax.set_title(f"Regime: {self._display_name(rid)}", fontsize=12, fontweight="bold", color="white")
                ax.set_xlabel("Bar index", fontsize=10, color="white")
                ax.set_ylabel("Cum. Return", fontsize=10, color="white")
                ax.tick_params(colors="white")
                ax.set_facecolor("#1e1e1e")
                ax.grid(alpha=0.2, color="#888")

            fig.suptitle("Cumulative Returns by Regime", fontsize=14, fontweight="bold", color="white")
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig

    def plot_regime_trade_distribution(self, figsize=(12, 5), save_path: Optional[str] = None) -> plt.Figure:
        """Box plot of trade PnL grouped by regime."""
        trades_by_regime = self._trades_by_regime()
        pnl_col = "PnL"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#121212")
            ax.set_facecolor("#1e1e1e")

            data_groups = []
            labels = []
            colors = []
            for idx, rid in enumerate(self._regimes):
                trades = trades_by_regime[rid]
                if not trades.empty and pnl_col in trades.columns:
                    pnl = trades[pnl_col].dropna().values
                    if len(pnl) > 0:
                        data_groups.append(pnl)
                        labels.append(self._display_name(rid))
                        colors.append(self._regime_color(idx))

            if data_groups:
                boxplot = ax.boxplot(
                    data_groups,
                    tick_labels=labels,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                )
                for patch, color in zip(boxplot["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for element in ["whiskers", "caps", "fliers"]:
                    for item in boxplot[element]:
                        item.set_color("#aaa")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No trade data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                    fontsize=12,
                )

            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.set_title("Trade PnL Distribution by Regime", fontsize=14, fontweight="bold", color="white")
            ax.set_xlabel("Regime", fontsize=11, color="white")
            ax.set_ylabel("PnL", fontsize=11, color="white")
            ax.tick_params(colors="white")
            ax.grid(axis="y", alpha=0.2, color="#888")
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
