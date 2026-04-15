from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import vectorbt as vbt

import matplotlib
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

    def __init__(self, portfolio: vbt.Portfolio, regime_labels: pd.Series, regime_names: Optional[Dict[int, str]] = None,
                 annualization_factor: int = 252) -> None:
        self.portfolio = portfolio
        self.regime_names = regime_names or {}
        self.annualization_factor = annualization_factor

        # Align labels to portfolio index
        pf_index = portfolio.wrapper.index
        if len(regime_labels.index.intersection(pf_index)) == 0:
            raise ValueError("regime_labels index has no overlap with the portfolio index.")
        self._labels: pd.Series = regime_labels.reindex(pf_index, method="ffill").astype(int)
        self._regimes: List[int] = sorted(self._labels.dropna().unique().tolist())

        # Cache expensive vbt calls
        self._returns: pd.Series = portfolio.returns()
        self._trades: pd.DataFrame = portfolio.trades.records_readable

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _display_name(self, regime_id: int) -> str:
        return self.regime_names.get(regime_id, str(regime_id))

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

    def _returns_by_regime(self) -> Dict[int, pd.Series]:
        result: Dict[int, pd.Series] = {}
        for rid in self._regimes:
            mask = self._labels == rid
            result[rid] = self._returns[mask]
        return result

    def _trades_by_regime(self) -> Dict[int, pd.DataFrame]:
        """Assign each trade to the regime at its entry bar."""
        if self._trades.empty:
            return {rid: pd.DataFrame() for rid in self._regimes}

        entry_col = "Entry Timestamp"
        entry_times = self._trades[entry_col]
        # reindex labels to trade entry timestamps (forward-fill)
        entry_regimes = self._labels.reindex(entry_times, method="ffill")
        entry_regimes.index = self._trades.index  # align back to trades index

        result: Dict[int, pd.DataFrame] = {}
        for rid in self._regimes:
            mask = entry_regimes == rid
            result[rid] = self._trades[mask]
        return result

    # ------------------------------------------------------------------
    # public analysis methods
    # ------------------------------------------------------------------

    def regime_return_stats(self) -> pd.DataFrame:
        """Per-regime return statistics.

        Returns a DataFrame indexed by regime display name with columns:
        ``n_bars``, ``total_return``, ``ann_return``, ``volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``.
        """
        rows = []
        rbr = self._returns_by_regime()
        for rid in self._regimes:
            rets = rbr[rid].dropna()
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
        df = pd.DataFrame(rows).set_index("regime")
        return df

    def regime_trade_stats(self) -> pd.DataFrame:
        """Per-regime trade statistics.

        Returns a DataFrame with columns:
        ``n_trades``, ``win_rate``, ``avg_win``, ``avg_loss``,
        ``profit_factor``, ``expectancy``, ``avg_duration_bars``.
        """
        rows = []
        tbr = self._trades_by_regime()
        for rid in self._regimes:
            trades = tbr[rid]
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

            pnl_col = "PnL"
            pnl = trades[pnl_col] if pnl_col in trades.columns else pd.Series(dtype=float)
            winners = pnl[pnl > 0]
            losers = pnl[pnl < 0]
            gross_profit = winners.sum()
            gross_loss = abs(losers.sum())
            profit_factor = (
                float(gross_profit / gross_loss) if gross_loss > 0 else np.nan
            )
            win_rate = len(winners) / n if n > 0 else np.nan

            # Duration in bars
            dur_col = "Duration"
            avg_dur = np.nan
            if dur_col in trades.columns:
                try:
                    avg_dur = float(trades[dur_col].mean())
                except Exception:
                    avg_dur = np.nan

            rows.append(
                {
                    "regime": self._display_name(rid),
                    "n_trades": n,
                    "win_rate": float(win_rate),
                    "avg_win": float(winners.mean()) if len(winners) > 0 else np.nan,
                    "avg_loss": float(losers.mean()) if len(losers) > 0 else np.nan,
                    "profit_factor": profit_factor,
                    "expectancy": float(pnl.mean()) if n > 0 else np.nan,
                    "avg_duration_bars": avg_dur,
                }
            )
        df = pd.DataFrame(rows).set_index("regime")
        return df

    def regime_exposure(self) -> pd.DataFrame:
        """Per-regime time exposure.

        Returns a DataFrame with columns:
        ``n_bars``, ``pct_time``, ``n_bars_in_position``, ``pct_time_in_position``.
        """
        total_bars = len(self._labels)
        try:
            in_position = self.portfolio.asset_value() != 0
        except Exception:
            in_position = pd.Series(False, index=self.portfolio.wrapper.index)

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
        df = pd.DataFrame(rows).set_index("regime")
        return df

    def generate_full_report(self) -> Dict[str, pd.DataFrame]:
        """Return all three DataFrames in a single dict.

        Returns
        -------
        dict with keys ``"return_stats"``, ``"trade_stats"``, ``"exposure"``.
        """
        return {
            "return_stats": self.regime_return_stats(),
            "trade_stats": self.regime_trade_stats(),
            "exposure": self.regime_exposure(),
        }

    # ------------------------------------------------------------------
    # plot helpers
    # ------------------------------------------------------------------

    _REGIME_COLORS = [
        "#2E86AB", "#F77F00", "#06A77D", "#D62828", "#8338EC",
        "#FB5607", "#3A86FF", "#FFBE0B", "#FF006E", "#8AC926",
    ]

    def _regime_color(self, idx: int) -> str:
        return self._REGIME_COLORS[idx % len(self._REGIME_COLORS)]

    def _bar_chart(self, ax, labels, values, title, ylabel, pct=False):
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

    # ------------------------------------------------------------------
    # public plot methods
    # ------------------------------------------------------------------

    def plot_regime_summary(self, figsize=(14, 10), save_path: Optional[str] = None) -> plt.Figure:
        """2×2 bar chart summary: total return, Sharpe, win rate, trade count."""
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.patch.set_facecolor("#121212")

            ret_stats = self.regime_return_stats()
            trd_stats = self.regime_trade_stats()

            names = ret_stats.index.tolist()

            self._bar_chart(
                axes[0, 0], names,
                [v * 100 if not np.isnan(v) else 0 for v in ret_stats["total_return"]],
                "Total Return by Regime", "Return (%)",
            )
            self._bar_chart(
                axes[0, 1], names,
                [v if not np.isnan(v) else 0 for v in ret_stats["sharpe"]],
                "Sharpe Ratio by Regime", "Sharpe Ratio",
            )
            self._bar_chart(
                axes[1, 0], names,
                [v * 100 if not np.isnan(v) else 0 for v in trd_stats["win_rate"]],
                "Win Rate by Regime", "Win Rate (%)", pct=True,
            )
            self._bar_chart(
                axes[1, 1], names,
                [v for v in trd_stats["n_trades"]],
                "Trade Count by Regime", "# Trades",
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

            rbr = self._returns_by_regime()
            for idx, rid in enumerate(self._regimes):
                ax = axes[0, idx]
                rets = rbr[rid].dropna()
                if len(rets) > 0:
                    cum = (1 + rets).cumprod()
                    ax.plot(
                        range(len(cum)), cum.values,
                        color=self._regime_color(idx), linewidth=1.5,
                    )
                    ax.fill_between(
                        range(len(cum)), 1, cum.values,
                        alpha=0.2, color=self._regime_color(idx),
                    )
                ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
                ax.set_title(
                    f"Regime: {self._display_name(rid)}",
                    fontsize=12, fontweight="bold", color="white",
                )
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
        tbr = self._trades_by_regime()
        pnl_col = "PnL"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#121212")
            ax.set_facecolor("#1e1e1e")

            data_groups = []
            labels = []
            colors = []
            for idx, rid in enumerate(self._regimes):
                trades = tbr[rid]
                if not trades.empty and pnl_col in trades.columns:
                    pnl = trades[pnl_col].dropna().values
                    if len(pnl) > 0:
                        data_groups.append(pnl)
                        labels.append(self._display_name(rid))
                        colors.append(self._regime_color(idx))

            if data_groups:
                bp = ax.boxplot(
                    data_groups,
                    tick_labels=labels,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for element in ["whiskers", "caps", "fliers"]:
                    for item in bp[element]:
                        item.set_color("#aaa")
            else:
                ax.text(0.5, 0.5, "No trade data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=12)

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
