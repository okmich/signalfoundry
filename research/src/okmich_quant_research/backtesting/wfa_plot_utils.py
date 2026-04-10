from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter threading issues
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class VisualizationMixin:
    """Mixin providing common visualization functionality for walk-forward optimizers.

    Classes using this mixin must have:
        - self.window_results (List): List of window results
        - self.all_predictions (pd.Series): Predictions across all windows
        - self.all_returns (pd.Series): Returns for backtesting classes (optional)
        - self.checkpoint_dir (Path): Directory for saving plots
        - self.env: EnvironmentDetector instance with is_interactive() method
        - self.log: Logger instance with info/warning methods
    """

    def _plot_save_and_show(
        self, fig, save_path: Optional[Path], plot_name: str
    ) -> None:
        """Save and optionally show a plot.

        Args:
            fig: Matplotlib figure
            save_path: Path to save plot (if None, uses checkpoint_dir)
            plot_name: Name of the plot for default save path
        """

        if save_path is None:
            save_path = self.checkpoint_dir / f"{plot_name}.png"

        if self.env.is_interactive():
            plt.show()

        try:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.log.info(f"Plot saved to: {save_path}", "💾")
        except Exception as e:
            self.log.warning(f"Could not save plot: {e}")

        plt.close()

    def _setup_subplot_style(
        self, ax, title: str, xlabel: str = "", ylabel: str = "", add_grid: bool = True
    ) -> None:
        """Apply consistent styling to subplot.

        Args:
            ax: Matplotlib axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            add_grid: Whether to add grid
        """
        ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        if add_grid:
            ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    def _plot_confusion_matrix(self, ax, y_true, y_pred) -> None:
        """Plot confusion matrix on given axis.

        Args:
            ax: Matplotlib axis
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        self._setup_subplot_style(ax, "Overall Confusion Matrix", add_grid=False)

    def _plot_metric_by_window(
        self,
        ax,
        results_df: pd.DataFrame,
        metric_col: str,
        title: str,
        ylabel: str,
        threshold: Optional[float] = None,
        color_positive: str = "#06A77D",
        color_negative: str = "#D62828",
    ) -> None:
        """Plot a metric by window with color coding.

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with window results
            metric_col: Column name for metric to plot
            title: Plot title
            ylabel: Y-axis label
            threshold: Optional threshold for color coding (values above are positive color)
            color_positive: Color for positive/good values
            color_negative: Color for negative/bad values
        """
        if threshold is not None:
            colors = [
                color_positive if x > threshold else color_negative
                for x in results_df[metric_col]
            ]
        else:
            colors = color_positive

        ax.bar(results_df["window_idx"], results_df[metric_col], color=colors)

        if threshold is not None:
            ax.axhline(
                y=threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.5
            )

        self._setup_subplot_style(ax, title, xlabel="Window", ylabel=ylabel)

    def _plot_dual_metric_by_window(
        self,
        ax,
        results_df: pd.DataFrame,
        metric1_col: str,
        metric2_col: str,
        title: str,
        ylabel: str,
        label1: str,
        label2: str,
    ) -> None:
        """Plot two metrics on the same axis by window.

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with window results
            metric1_col: Column name for first metric
            metric2_col: Column name for second metric
            title: Plot title
            ylabel: Y-axis label
            label1: Label for first metric
            label2: Label for second metric
        """
        ax.plot(
            results_df["window_idx"],
            results_df[metric1_col],
            marker="o",
            label=label1,
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            results_df["window_idx"],
            results_df[metric2_col],
            marker="s",
            label=label2,
            linewidth=2,
            color="#F77F00",
        )
        ax.legend()
        self._setup_subplot_style(ax, title, xlabel="Window", ylabel=ylabel)

    def _plot_equity_curve_on_axis(
        self, ax, returns: pd.Series, title: str = "Cumulative Equity Curve"
    ) -> None:
        """Plot equity curve on given axis.

        Args:
            ax: Matplotlib axis
            returns: Series of returns
            title: Plot title
        """
        equity = (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, linewidth=2, color="#2E86AB")
        self._setup_subplot_style(ax, title, xlabel="Date", ylabel="Cumulative Return")

    def _plot_drawdown_on_axis(
        self, ax, returns: pd.Series, title: str = "Drawdown"
    ) -> None:
        """Plot drawdown on given axis.

        Args:
            ax: Matplotlib axis
            returns: Series of returns
            title: Plot title
        """
        try:
            dd = returns.vbt.drawdown()
            ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#D62828")
            ax.plot(dd.index, dd.values, color="#D62828", linewidth=2)
            self._setup_subplot_style(ax, title, xlabel="Date", ylabel="Drawdown")
        except Exception as e:
            self.log.debug(f"Could not plot drawdown: {e}")
            ax.text(
                0.5,
                0.5,
                "Drawdown plotting unavailable",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                transform=ax.transAxes,
            )
            self._setup_subplot_style(ax, title, add_grid=False)

    def plot_results(self, figsize=(15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot performance visualization (dispatches to classification or regression).

        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        try:
            if len(self.all_predictions) == 0:
                self.log.warning("No predictions to plot")
                return

            # Determine task type from window results
            task_type = getattr(self, 'task_type', 'classification')
            if hasattr(self, 'window_results') and len(self.window_results) > 0:
                task_type = self.window_results[0].task_type

            if task_type == "classification":
                self._plot_classification_results(figsize, save_path)
            else:  # regression
                self._plot_regression_results(figsize, save_path)

        except Exception as e:
            self.log.warning(f"Could not create plot: {e}")

    def _plot_classification_results(self, figsize=(15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot classification performance visualization.

        Creates 2x2 grid with:
        - F1 Score by window
        - Accuracy by window
        - Confusion matrix
        - Precision vs Recall by window

        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])

        # F1 Score by window
        self._plot_metric_by_window(
            axes[0, 0],
            results_df,
            "f1_score",
            "F1 Score by Window",
            "F1 Score",
            threshold=0.5,
        )

        # Accuracy by window
        self._plot_metric_by_window(
            axes[0, 1],
            results_df,
            "accuracy",
            "Accuracy by Window",
            "Accuracy",
            threshold=0.5,
        )

        # Confusion Matrix
        all_preds_binary = (self.all_predictions > 0.5).astype(int)
        self._plot_confusion_matrix(
            axes[1, 0], self.all_true_labels, all_preds_binary
        )

        # Precision vs Recall by Window
        axes[1, 1].plot(
            [0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random"
        )
        axes[1, 1].set_xlim([0, len(results_df) + 1])
        axes[1, 1].set_ylim([0, 1])
        self._plot_dual_metric_by_window(
            axes[1, 1],
            results_df,
            "precision",
            "recall",
            "Precision vs Recall by Window",
            "Score",
            "Precision",
            "Recall",
        )

        plt.tight_layout()
        self._plot_save_and_show(
            fig, Path(save_path) if save_path else None, "wfa_results"
        )

    def _plot_regression_results(self, figsize=(18, 12), save_path: Optional[str] = None) -> None:
        """
        Plot regression performance visualization.

        Creates 2x2 grid with:
        - R² by window
        - RMSE by window
        - Predictions vs Actual scatter
        - Residual distribution

        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])

        # Plot 1: R² by window
        self._plot_metric_by_window(
            axes[0, 0],
            results_df,
            "r2",
            "R² Score by Window",
            "R²",
            threshold=0.5,
        )

        # Plot 2: RMSE by window
        colors = ["#06A77D" if x < results_df["rmse"].median() else "#D62828" for x in results_df["rmse"]]
        axes[0, 1].bar(results_df["window_idx"], results_df["rmse"], color=colors)
        axes[0, 1].axhline(
            y=results_df["rmse"].median(), color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Median"
        )
        self._setup_subplot_style(axes[0, 1], "RMSE by Window", xlabel="Window", ylabel="RMSE")
        axes[0, 1].legend()

        # Plot 3: Predictions vs Actual scatter
        axes[1, 0].scatter(
            self.all_true_labels, self.all_predictions, alpha=0.5, s=10, color="#2E86AB"
        )
        # Add diagonal line (perfect predictions)
        min_val = min(self.all_true_labels.min(), self.all_predictions.min())
        max_val = max(self.all_true_labels.max(), self.all_predictions.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
        self._setup_subplot_style(axes[1, 0], "Predictions vs Actual", xlabel="Actual", ylabel="Predicted")
        axes[1, 0].legend()

        # Plot 4: Residual distribution
        residuals = self.all_predictions - self.all_true_labels
        axes[1, 1].hist(residuals, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
        self._setup_subplot_style(
            axes[1, 1], "Residual Distribution", xlabel="Residual (Predicted - Actual)", ylabel="Frequency"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        self._plot_save_and_show(
            fig, Path(save_path) if save_path else None, "wfa_regression_results"
        )


class BacktestVisualizationMixin(VisualizationMixin):
    """Extended visualization mixin for backtesting optimizers.

    Adds trading-specific visualization methods on top of base VisualizationMixin.

    Classes using this mixin must have (in addition to VisualizationMixin requirements):
        - self.all_returns (pd.Series): Portfolio returns
        - self.signal_generator_fn (optional): Signal generator function
    """

    def plot_results(self, figsize=(18, 12), save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive performance visualization for backtesting.

        Creates 3x2 grid with:
        - Equity curve
        - Drawdown chart
        - Sharpe ratio by window
        - Return by window
        - Classification/Regression metric by window
        - Confusion matrix (classification) or Predictions vs Actual (regression)

        Args:
            figsize: Figure size tuple
            save_path: Path to save plot (if None, uses checkpoint_dir)
        """
        try:

            if len(self.all_returns) == 0:
                self.log.warning("No returns to plot")
                return

            # Determine task type
            task_type = getattr(self, 'task_type', 'classification')
            if hasattr(self, 'window_results') and len(self.window_results) > 0:
                task_type = self.window_results[0].task_type

            fig, axes = plt.subplots(3, 2, figsize=figsize)
            results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])

            # Row 1, Col 1: Equity Curve
            self._plot_equity_curve_on_axis(
                axes[0, 0], self.all_returns, "Equity Curve (Out-of-Sample)"
            )

            # Row 1, Col 2: Drawdown
            self._plot_drawdown_on_axis(axes[0, 1], self.all_returns)

            # Row 2, Col 1: Sharpe Ratio by Window
            self._plot_metric_by_window(
                axes[1, 0],
                results_df,
                "sharpe_ratio",
                "Sharpe Ratio by Window",
                "Sharpe Ratio",
                threshold=0,
            )

            # Row 2, Col 2: Return by Window
            self._plot_metric_by_window(
                axes[1, 1],
                results_df,
                "total_return",
                "Return by Window",
                "Return (%)",
                threshold=0,
            )

            # Row 3: Task-specific metrics
            if task_type == "classification":
                # Row 3, Col 1: F1 Score by Window
                self._plot_metric_by_window(
                    axes[2, 0],
                    results_df,
                    "f1_score",
                    "F1 Score by Window",
                    "F1 Score",
                    threshold=0.5,
                )

                # Row 3, Col 2: Confusion Matrix
                all_preds_binary = (self.all_predictions > 0.5).astype(int)
                self._plot_confusion_matrix(
                    axes[2, 1], self.all_true_labels, all_preds_binary
                )
            else:  # regression
                # Row 3, Col 1: R² by Window
                self._plot_metric_by_window(
                    axes[2, 0],
                    results_df,
                    "r2",
                    "R² Score by Window",
                    "R²",
                    threshold=0.5,
                )

                # Row 3, Col 2: Predictions vs Actual scatter
                axes[2, 1].scatter(
                    self.all_true_labels, self.all_predictions, alpha=0.5, s=10, color="#2E86AB"
                )
                min_val = min(self.all_true_labels.min(), self.all_predictions.min())
                max_val = max(self.all_true_labels.max(), self.all_predictions.max())
                axes[2, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")
                self._setup_subplot_style(axes[2, 1], "Predictions vs Actual", xlabel="Actual", ylabel="Predicted")
                axes[2, 1].legend()

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "wfa_backtest_results"
            )
        except Exception as e:
            self.log.warning(f"Could not create plot: {e}")

    def plot_equity_curve(
        self, figsize=(18, 6), save_path: Optional[str] = None
    ) -> None:
        """
        Plot equity curve separately with more detail.

        Args:
            figsize: Figure size
            save_path: Path to save plot
        """
        try:
            if len(self.all_returns) == 0:
                self.log.warning("No returns to plot")
                return

            fig, ax = plt.subplots(figsize=figsize)

            equity = (1 + self.all_returns).cumprod()
            ax.plot(
                equity.index,
                equity.values,
                linewidth=2,
                color="#2E86AB",
                label="Strategy",
            )

            # Add buy & hold benchmark if possible
            if hasattr(self, "raw_data") and hasattr(self, "close_col"):
                try:
                    price_returns = self.raw_data[self.close_col].pct_change()
                    aligned_returns = price_returns.reindex(self.all_returns.index)
                    benchmark_equity = (1 + aligned_returns).cumprod()
                    ax.plot(
                        benchmark_equity.index,
                        benchmark_equity.values,
                        linewidth=2,
                        color="#6C757D",
                        linestyle="--",
                        alpha=0.7,
                        label="Buy & Hold",
                    )
                except Exception:
                    pass

            ax.set_title("Equity Curve Comparison", fontsize=16, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#F8F9FA")

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "equity_curve"
            )
        except Exception as e:
            self.log.warning(f"Could not create equity curve plot: {e}")

    def plot_rolling_metrics(
        self, window_size: int = 3, figsize=(18, 10), save_path: Optional[str] = None
    ) -> None:
        """
        Plot rolling averages of key metrics.

        Args:
            window_size: Size of rolling window for averaging
            figsize: Figure size
            save_path: Path to save plot
        """
        try:
            if not self.window_results:
                self.log.warning("No results to plot")
                return

            results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Rolling Sharpe
            rolling_sharpe = results_df["sharpe_ratio"].rolling(window_size).mean()
            axes[0, 0].plot(
                results_df["window_idx"], rolling_sharpe, linewidth=2, color="#2E86AB"
            )
            axes[0, 0].axhline(
                y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5
            )
            self._setup_subplot_style(
                axes[0, 0],
                f"Rolling Sharpe Ratio ({window_size}-window)",
                xlabel="Window",
                ylabel="Sharpe",
            )

            # Rolling Return
            rolling_return = results_df["total_return"].rolling(window_size).mean()
            axes[0, 1].plot(
                results_df["window_idx"], rolling_return, linewidth=2, color="#06A77D"
            )
            axes[0, 1].axhline(
                y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5
            )
            self._setup_subplot_style(
                axes[0, 1],
                f"Rolling Return ({window_size}-window)",
                xlabel="Window",
                ylabel="Return (%)",
            )

            # Rolling F1
            rolling_f1 = results_df["f1_score"].rolling(window_size).mean()
            axes[1, 0].plot(
                results_df["window_idx"], rolling_f1, linewidth=2, color="#F77F00"
            )
            axes[1, 0].axhline(
                y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5
            )
            self._setup_subplot_style(
                axes[1, 0],
                f"Rolling F1 Score ({window_size}-window)",
                xlabel="Window",
                ylabel="F1 Score",
            )

            # Rolling Win Rate
            rolling_wr = results_df["win_rate"].rolling(window_size).mean()
            axes[1, 1].plot(
                results_df["window_idx"], rolling_wr, linewidth=2, color="#D62828"
            )
            self._setup_subplot_style(
                axes[1, 1],
                f"Rolling Win Rate ({window_size}-window)",
                xlabel="Window",
                ylabel="Win Rate (%)",
            )

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "rolling_metrics"
            )
        except Exception as e:
            self.log.warning(f"Could not create rolling metrics plot: {e}")


class HMMVisualizationMixin(VisualizationMixin):
    """Extended visualization mixin for HMM optimizers.

    Adds HMM-specific visualization methods on top of base VisualizationMixin.

    Classes using this mixin must have (in addition to VisualizationMixin requirements):
        - self.all_returns (pd.Series): Portfolio returns (for backtesting HMMs)
        - self.signal_generator_fn (optional): Signal generator function
    """

    def plot_results(self, figsize=(18, 12), save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive performance visualization for HMM models.

        Creates 3x2 grid with:
        - State predictions over time
        - HMM metrics by window (AIC/BIC, Entropy, Stability)
        - Trading performance by window (if applicable)
        - State duration distribution
        - Equity curve (if applicable)
        - Confusion matrix of state transitions

        Args:
            figsize: Figure size tuple
            save_path: Path to save plot (if None, uses checkpoint_dir)
        """
        try:
            if len(self.window_results) == 0:
                self.log.warning("No results to plot")
                return

            fig, axes = plt.subplots(3, 2, figsize=figsize)
            results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])

            # Row 1, Col 1: State predictions over time
            if len(self.all_predictions) > 0:
                axes[0, 0].plot(
                    self.all_predictions.index,
                    self.all_predictions.values,
                    linewidth=1,
                    alpha=0.7,
                )
                self._setup_subplot_style(
                    axes[0, 0],
                    "State Predictions Over Time",
                    xlabel="Date",
                    ylabel="State ID",
                )

            # Row 1, Col 2: AIC/BIC by window
            self._plot_dual_metric_by_window(
                axes[0, 1],
                results_df,
                "aic_score",
                "bic_score",
                "Model Selection Criteria by Window",
                "Score",
                "AIC",
                "BIC",
            )

            # Row 2, Col 1: Entropy and Stability
            ax2_1_twin = axes[1, 0].twinx()
            axes[1, 0].bar(
                results_df["window_idx"],
                results_df["avg_state_entropy"],
                alpha=0.6,
                color="#06A77D",
                label="Entropy",
            )
            ax2_1_twin.plot(
                results_df["window_idx"],
                results_df["state_stability"],
                marker="o",
                color="#D62828",
                linewidth=2,
                label="Stability",
            )
            axes[1, 0].set_title(
                "State Quality Metrics", fontsize=14, fontweight="bold"
            )
            axes[1, 0].set_xlabel("Window", fontsize=11)
            axes[1, 0].set_ylabel("Entropy", fontsize=11, color="#06A77D")
            ax2_1_twin.set_ylabel("Stability", fontsize=11, color="#D62828")
            axes[1, 0].tick_params(axis="y", labelcolor="#06A77D")
            ax2_1_twin.tick_params(axis="y", labelcolor="#D62828")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_facecolor("#F8F9FA")

            # Row 2, Col 2: Trading metrics (if available)
            if (
                hasattr(self, "signal_generator_fn")
                and self.signal_generator_fn
                and "sharpe_ratio" in results_df.columns
            ):
                self._plot_metric_by_window(
                    axes[1, 1],
                    results_df,
                    "sharpe_ratio",
                    "Sharpe Ratio by Window",
                    "Sharpe Ratio",
                    threshold=0,
                )
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No Trading Metrics\n(signal_generator_fn not provided)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_facecolor("#F8F9FA")

            # Row 3, Col 1: State duration distribution
            all_durations = []
            for wr in self.window_results:
                if not np.isnan(wr.median_state_duration):
                    all_durations.append(wr.median_state_duration)

            if len(all_durations) > 0:
                axes[2, 0].hist(
                    all_durations,
                    bins=20,
                    color="#2E86AB",
                    alpha=0.7,
                    edgecolor="black",
                )
                self._setup_subplot_style(
                    axes[2, 0],
                    "Distribution of Median State Durations",
                    xlabel="Duration (samples)",
                    ylabel="Frequency",
                )
            else:
                axes[2, 0].text(
                    0.5,
                    0.5,
                    "No Duration Data",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                    transform=axes[2, 0].transAxes,
                )
                axes[2, 0].set_facecolor("#F8F9FA")

            # Row 3, Col 2: Equity curve (if available) or state count distribution
            if (
                hasattr(self, "signal_generator_fn")
                and self.signal_generator_fn
                and len(self.all_returns) > 0
            ):
                self._plot_equity_curve_on_axis(
                    axes[2, 1], self.all_returns, "Cumulative Equity Curve"
                )
            else:
                # Show n_states distribution
                state_counts = results_df["n_states"].value_counts().sort_index()
                axes[2, 1].bar(
                    state_counts.index, state_counts.values, color="#F77F00", alpha=0.7
                )
                self._setup_subplot_style(
                    axes[2, 1],
                    "Optimal Number of States",
                    xlabel="Number of States",
                    ylabel="Frequency",
                )

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "hmm_wfa_results"
            )
        except Exception as e:
            self.log.warning(f"Could not create plot: {e}")

    def plot_state_transitions(
        self, figsize=(12, 10), save_path: Optional[str] = None
    ) -> None:
        """
        Plot state transition matrix heatmap across all windows.

        Shows how often the HMM transitions between different states,
        aggregated across all predictions.

        Args:
            figsize: Figure size
            save_path: Path to save plot
        """
        try:
            if len(self.all_predictions) == 0:
                self.log.warning("No predictions to plot")
                return

            # Calculate transition matrix
            states = self.all_predictions.values
            n_states = int(states.max() + 1)

            # Initialize transition matrix
            transition_matrix = np.zeros((n_states, n_states))

            # Count transitions
            for i in range(len(states) - 1):
                from_state = int(states[i])
                to_state = int(states[i + 1])
                transition_matrix[from_state, to_state] += 1

            # Normalize rows to get probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_probs = transition_matrix / row_sums

            # Create heatmap
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(transition_probs, cmap="Blues", aspect="auto")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(
                "Transition Probability", rotation=270, labelpad=20, fontsize=12
            )

            # Set ticks
            ax.set_xticks(np.arange(n_states))
            ax.set_yticks(np.arange(n_states))
            ax.set_xticklabels([f"State {i}" for i in range(n_states)])
            ax.set_yticklabels([f"State {i}" for i in range(n_states)])

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    text = ax.text(
                        j,
                        i,
                        f"{transition_probs[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if transition_probs[i, j] < 0.5 else "white",
                        fontsize=10,
                    )

            ax.set_title(
                "State Transition Matrix (Overall)",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xlabel("To State", fontsize=12)
            ax.set_ylabel("From State", fontsize=12)

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "state_transitions"
            )
        except Exception as e:
            self.log.warning(f"Could not create state transition plot: {e}")

    def plot_regime_performance(
        self, figsize=(16, 10), save_path: Optional[str] = None
    ) -> None:
        """
        Plot trading performance broken down by regime (state).

        Shows how the strategy performs in different market regimes/states.
        Only available if signal_generator_fn was provided.

        Args:
            figsize: Figure size
            save_path: Path to save plot
        """
        try:
            if not hasattr(self, "signal_generator_fn") or not self.signal_generator_fn:
                self.log.warning(
                    "No signal generator - regime performance not available"
                )
                return

            if len(self.all_predictions) == 0 or len(self.all_returns) == 0:
                self.log.warning("No predictions or returns to plot")
                return

            # Align predictions and returns
            aligned_df = pd.DataFrame(
                {"state": self.all_predictions, "returns": self.all_returns}
            ).dropna()

            if len(aligned_df) == 0:
                self.log.warning("No aligned data for regime performance")
                return

            states = sorted(aligned_df["state"].unique())

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Plot 1: Returns distribution by state
            for state in states:
                state_returns = aligned_df[aligned_df["state"] == state]["returns"]
                axes[0, 0].hist(
                    state_returns, bins=50, alpha=0.5, label=f"State {int(state)}"
                )

            self._setup_subplot_style(
                axes[0, 0],
                "Return Distribution by State",
                xlabel="Return",
                ylabel="Frequency",
            )
            axes[0, 0].legend()

            # Plot 2: Mean return by state
            mean_returns = []
            std_returns = []
            for state in states:
                state_returns = aligned_df[aligned_df["state"] == state]["returns"]
                mean_returns.append(state_returns.mean())
                std_returns.append(state_returns.std())

            colors = ["#06A77D" if x > 0 else "#D62828" for x in mean_returns]
            axes[0, 1].bar(
                [f"State {int(s)}" for s in states],
                mean_returns,
                yerr=std_returns,
                color=colors,
                alpha=0.7,
                capsize=5,
            )
            axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            self._setup_subplot_style(
                axes[0, 1], "Mean Return by State", xlabel="State", ylabel="Mean Return"
            )

            # Plot 3: Sharpe ratio by state
            sharpe_by_state = []
            for state in states:
                state_returns = aligned_df[aligned_df["state"] == state]["returns"]
                if len(state_returns) > 1 and state_returns.std() > 0:
                    sharpe = (state_returns.mean() / state_returns.std()) * np.sqrt(252)
                    sharpe_by_state.append(sharpe)
                else:
                    sharpe_by_state.append(np.nan)

            colors = ["#06A77D" if x > 0 else "#D62828" for x in sharpe_by_state]
            axes[1, 0].bar(
                [f"State {int(s)}" for s in states],
                sharpe_by_state,
                color=colors,
                alpha=0.7,
            )
            axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            self._setup_subplot_style(
                axes[1, 0],
                "Sharpe Ratio by State",
                xlabel="State",
                ylabel="Sharpe Ratio",
            )

            # Plot 4: Time spent in each state
            state_counts = aligned_df["state"].value_counts().sort_index()
            state_proportions = state_counts / len(aligned_df) * 100

            axes[1, 1].bar(
                [f"State {int(s)}" for s in state_proportions.index],
                state_proportions.values,
                color="#2E86AB",
                alpha=0.7,
            )
            self._setup_subplot_style(
                axes[1, 1],
                "Time Spent in Each State",
                xlabel="State",
                ylabel="Percentage (%)",
            )

            plt.tight_layout()
            self._plot_save_and_show(
                fig, Path(save_path) if save_path else None, "regime_performance"
            )
        except Exception as e:
            self.log.warning(f"Could not create regime performance plot: {e}")
