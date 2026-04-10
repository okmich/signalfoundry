"""
Static matplotlib visualisations for ROC analysis results.

Source / Attribution
--------------------
The visualisation concepts — ROC curve, signal-vs-returns scatter, quantile analysis, and cumulative-returns comparison
— are described in the visualisation guide distributed alongside:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

The ``ROCVisualizer`` class is an original implementation for this project, inspired by Masters' descriptions of what
he considers the most informative views for evaluating indicator predictive power.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .roc_analysis import ROCResults


class ROCVisualizer:
    """
    Static matplotlib visualisations for :class:`~roc_analysis.ROCResults`.

    All methods are static.  Import matplotlib lazily so that the module can be imported without a display (e.g. in a headless server context).
    """

    # ------------------------------------------------------------------ #
    # 1. ROC Curve                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_roc_curve(results: ROCResults, figsize: Tuple[int, int] = (14, 6), save_path: Optional[str] = None):
        """
        Plot profit factor vs threshold (Masters' primary ROC diagnostic).

        Left panel: long-PF (signal >= threshold) and short-PF (signal < threshold) as functions of threshold value,
        with vertical lines at the optimal thresholds and shading above the break-even PF=1 line.

        Right panel: number of trades triggered at each threshold.

        Parameters
        ----------
        results : ROCResults
            Output of :meth:`ROCAnalyzer.analyze`.
        figsize : tuple, default (14, 6)
        save_path : str or Path, optional
            If provided the figure is saved to this path (PNG, 150 dpi).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        tbl = results.roc_table
        n = results.n_cases

        # Left panel — profit factors
        ax1.plot(tbl["threshold"], tbl["pf_long_above"], "g-o",
                 lw=2.5, ms=6, label="Long PF  (signal ≥ threshold)", alpha=0.85)
        ax1.plot(tbl["threshold"], tbl["pf_short_below"], "r-s",
                 lw=2.5, ms=6, label="Short PF  (signal < threshold)", alpha=0.85)
        ax1.axvline(results.long_threshold, color="green", ls="--", lw=2, alpha=0.6,
                    label=f"Optimal long  {results.long_threshold:+.4f}")
        ax1.axvline(results.short_threshold, color="red", ls="--", lw=2, alpha=0.6,
                    label=f"Optimal short {results.short_threshold:+.4f}")
        ax1.axhline(1.0, color="black", ls="-", lw=1, alpha=0.3, label="Break-even")
        ax1.fill_between(tbl["threshold"], 1.0, tbl["pf_long_above"],
                         where=(tbl["pf_long_above"] > 1.0), alpha=0.15, color="green")
        ax1.fill_between(tbl["threshold"], 1.0, tbl["pf_short_below"],
                         where=(tbl["pf_short_below"] > 1.0), alpha=0.15, color="red")
        ax1.set_xlabel("Signal threshold", fontweight="bold")
        ax1.set_ylabel("Profit factor", fontweight="bold")
        ax1.set_title("ROC Curve — Profit Factor vs Threshold", fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right panel — trade count
        ax2.plot(tbl["threshold"], tbl["frac_above"] * n, "g-o",
                 lw=2.5, ms=6, label="Long trades (≥ threshold)", alpha=0.85)
        ax2.plot(tbl["threshold"], tbl["frac_below"] * n, "r-s",
                 lw=2.5, ms=6, label="Short trades (< threshold)", alpha=0.85)
        ax2.axvline(results.long_threshold, color="green", ls="--", lw=2, alpha=0.6)
        ax2.axvline(results.short_threshold, color="red", ls="--", lw=2, alpha=0.6)
        ax2.set_xlabel("Signal threshold", fontweight="bold")
        ax2.set_ylabel("Number of trades", fontweight="bold")
        ax2.set_title("Trade Count vs Threshold", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        best_pf = max(results.long_pf, results.short_pf)
        sig = "***" if results.best_pval < 0.01 else ("**" if results.best_pval < 0.05 else "")
        fig.suptitle(
            f"Best PF: {best_pf:.3f}   p={results.best_pval:.4f}{sig}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------ #
    # 2. Signal vs Returns scatter                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_signal_vs_returns(signals: np.ndarray, returns: np.ndarray, results: ROCResults, bins: int = 50,
                               figsize: Tuple[int, int] = (14, 10), save_path: Optional[str] = None):
        """
        Density hexbin of signal values vs forward returns.

        Shows the raw signal–return relationship.  Vertical dashed lines mark the optimal long and short thresholds.

        Returns
        -------
        matplotlib.figure.Figure
        """

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        ax_main = fig.add_subplot(gs[0:2, :])
        hb = ax_main.hexbin(signals, returns, gridsize=bins,
                            cmap="YlOrRd", mincnt=1, alpha=0.85)
        ax_main.axvline(results.long_threshold, color="green", ls="--", lw=2.5,
                        label=f"Long threshold: {results.long_threshold:+.4f}")
        ax_main.axvline(results.short_threshold, color="red", ls="--", lw=2.5,
                        label=f"Short threshold: {results.short_threshold:+.4f}")
        ax_main.axhline(0, color="black", ls="-", lw=1, alpha=0.4)
        ax_main.set_xlabel("Signal value", fontweight="bold")
        ax_main.set_ylabel("Forward return", fontweight="bold")
        ax_main.set_title("Signal vs Forward Returns (density hexbin)", fontweight="bold")
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)
        plt.colorbar(hb, ax=ax_main, label="Count")

        ax_sig = fig.add_subplot(gs[2, 0])
        ax_sig.hist(signals, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax_sig.axvline(results.long_threshold, color="green", ls="--", lw=2)
        ax_sig.axvline(results.short_threshold, color="red", ls="--", lw=2)
        ax_sig.set_xlabel("Signal value", fontweight="bold")
        ax_sig.set_ylabel("Frequency", fontweight="bold")
        ax_sig.set_title("Signal distribution", fontweight="bold")
        ax_sig.grid(True, alpha=0.3)

        ax_ret = fig.add_subplot(gs[2, 1])
        ax_ret.hist(returns, bins=50, color="coral", alpha=0.7, edgecolor="black")
        ax_ret.axvline(0, color="black", ls="-", lw=1.5)
        ax_ret.set_xlabel("Forward return", fontweight="bold")
        ax_ret.set_ylabel("Frequency", fontweight="bold")
        ax_ret.set_title("Returns distribution", fontweight="bold")
        ax_ret.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------ #
    # 3. Quantile analysis                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_quantile_analysis(signals: np.ndarray, returns: np.ndarray, n_quantiles: int = 10,
                               figsize: Tuple[int, int] = (14, 6), save_path: Optional[str] = None) -> Tuple:
        """
        Mean return, win rate, and Sharpe ratio by signal quantile.

        A monotonically increasing mean-return profile across quantiles is the
        clearest evidence that higher signal values predict better returns.

        Returns
        -------
        (matplotlib.figure.Figure, pd.DataFrame)
            Figure and per-quantile statistics DataFrame.
        """
        import matplotlib.pyplot as plt

        q_edges = np.percentile(signals, np.linspace(0, 100, n_quantiles + 1))
        q_labels = np.clip(np.digitize(signals, q_edges[:-1]) - 1, 0, n_quantiles - 1)

        rows = []
        for q in range(n_quantiles):
            mask = q_labels == q
            if mask.sum() == 0:
                continue
            r = returns[mask]
            rows.append({
                "quantile": q + 1,
                "mean_return": float(np.mean(r)),
                "win_rate": float((r > 0).mean()),
                "sharpe": float(np.mean(r) / (np.std(r) + 1e-10)),
                "n": int(mask.sum()),
            })

        df = pd.DataFrame(rows)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        colors_ret = ["red" if v < 0 else "green" for v in df["mean_return"]]
        ax1.bar(df["quantile"], df["mean_return"], color=colors_ret, alpha=0.75, edgecolor="black")
        ax1.axhline(0, color="black", lw=1)
        # Trend line
        z = np.polyfit(df["quantile"], df["mean_return"], 1)
        ax1.plot(df["quantile"], np.poly1d(z)(df["quantile"]), "b--", lw=2, alpha=0.6)
        ax1.set_xlabel(f"Signal quantile (1=lowest, {n_quantiles}=highest)", fontweight="bold")
        ax1.set_ylabel("Mean forward return", fontweight="bold")
        ax1.set_title("Mean Return by Signal Strength", fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(df["quantile"], df["win_rate"], color="steelblue", alpha=0.75, edgecolor="black")
        ax2.axhline(0.5, color="red", ls="--", lw=2, alpha=0.7, label="50 % (random)")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Signal quantile", fontweight="bold")
        ax2.set_ylabel("Win rate", fontweight="bold")
        ax2.set_title("Win Rate by Signal Strength", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        colors_sh = ["red" if v < 0 else "green" for v in df["sharpe"]]
        ax3.bar(df["quantile"], df["sharpe"], color=colors_sh, alpha=0.75, edgecolor="black")
        ax3.axhline(0, color="black", lw=1)
        ax3.set_xlabel("Signal quantile", fontweight="bold")
        ax3.set_ylabel("Sharpe ratio", fontweight="bold")
        ax3.set_title("Sharpe Ratio by Signal Strength", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Quantile Analysis  (n={len(signals):,}  ·  {n_quantiles} quantiles)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, df

    # ------------------------------------------------------------------ #
    # 4. Cumulative returns                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_cumulative_returns(signals: np.ndarray, returns: np.ndarray, results: ROCResults,
                                figsize: Tuple[int, int] = (14, 8), save_path: Optional[str] = None):
        """
        Equity curves for long strategy, short strategy, and buy-and-hold.

        Parameters
        ----------
        signals, returns : np.ndarray
            Aligned signal and return arrays.
        results : ROCResults
        figsize : tuple
        save_path : str or Path, optional

        Returns
        -------
        matplotlib.figure.Figure
        """

        long_mask = signals <= results.long_threshold
        short_mask = signals > results.short_threshold

        cum_long = np.exp(np.cumsum(np.where(long_mask, returns, 0))) - 1
        cum_short = np.exp(np.cumsum(np.where(short_mask, -returns, 0))) - 1
        cum_bh = np.exp(np.cumsum(returns)) - 1

        idx = np.arange(len(returns))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        ax1.plot(idx, cum_long * 100, "g-", lw=2,
                 label=f"Long  (PF={results.long_pf:.2f}, p={results.long_pval:.3f})")
        ax1.plot(idx, cum_short * 100, "r-", lw=2,
                 label=f"Short (PF={results.short_pf:.2f}, p={results.short_pval:.3f})")
        ax1.plot(idx, cum_bh * 100, color="grey", lw=2, ls="--", label="Buy & Hold")
        ax1.axhline(0, color="black", lw=1, alpha=0.3)
        ax1.set_ylabel("Cumulative return (%)", fontweight="bold")
        ax1.set_title("Cumulative Returns by Strategy", fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(idx, 0, 1, where=long_mask, color="green", alpha=0.35, label="Long")
        ax2.fill_between(idx, 0, 1, where=short_mask, color="red", alpha=0.35, label="Short")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Out", "In"])
        ax2.set_xlabel("Bar index", fontweight="bold")
        ax2.set_ylabel("Position", fontweight="bold")
        ax2.set_title("Trading Activity", fontweight="bold")
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------ #
    # 5. Full report                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_report(signals: np.ndarray, returns: np.ndarray, results: ROCResults, indicator_name: str = "Indicator",
                      output_dir: str = "output/roc_reports", show: bool = True) -> dict:
        """
        Generate a complete set of ROC visualisations and save them.

        Creates four PNG files plus a quantile CSV and a text summary::

            {indicator_name}_roc_curve.png
            {indicator_name}_scatter.png
            {indicator_name}_quantiles.png
            {indicator_name}_cumulative.png
            {indicator_name}_quantile_stats.csv
            {indicator_name}_summary.txt

        Parameters
        ----------
        signals, returns : np.ndarray
        results : ROCResults
        indicator_name : str
        output_dir : str or Path
        show : bool
            Call ``plt.show()`` after generating plots.

        Returns
        -------
        dict
            Keys: ``roc_curve``, ``scatter``, ``quantiles``, ``cumulative``,
            ``quantile_stats``.
        """

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ROC Report — {indicator_name}")
        print(f"{'='*60}")

        fig1 = ROCVisualizer.plot_roc_curve(
            results, save_path=out / f"{indicator_name}_roc_curve.png"
        )
        fig2 = ROCVisualizer.plot_signal_vs_returns(
            signals, returns, results,
            save_path=out / f"{indicator_name}_scatter.png",
        )
        fig3, df_q = ROCVisualizer.plot_quantile_analysis(
            signals, returns,
            save_path=out / f"{indicator_name}_quantiles.png",
        )
        fig4 = ROCVisualizer.plot_cumulative_returns(
            signals, returns, results,
            save_path=out / f"{indicator_name}_cumulative.png",
        )

        df_q.to_csv(out / f"{indicator_name}_quantile_stats.csv", index=False)
        (out / f"{indicator_name}_summary.txt").write_text(
            str(results) + "\n\nQuantile Analysis:\n" + df_q.to_string(index=False)
        )

        print(f"Report saved to: {out}")
        print(f"  4 PNG plots · 1 quantile CSV · 1 text summary\n")

        if show:
            plt.show()

        return {
            "roc_curve": fig1,
            "scatter": fig2,
            "quantiles": fig3,
            "cumulative": fig4,
            "quantile_stats": df_q,
        }
