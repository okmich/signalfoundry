import itertools
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib import pyplot as plt
from tqdm import tqdm

from okmich_quant_core.signal import BaseSignal
from .vbt_export import QuantframeExportMixin


class VectorBtBacktester(QuantframeExportMixin):
    def __init__(self, signal_obj: BaseSignal, timeframe: str = "1H"):
        """
        Initialize backtester
        Args:
            signal_obj: Signal instance to use for strategy
            timeframe: Timeframe string (e.g., '1D', '1H', '15T')
        """
        self.signal_obj = signal_obj
        self.timeframe = timeframe
        # Initialize containers
        self.data = None
        self.portfolio = None
        self.results = None
        self.optimization_results = None
        self.mc_simulations = None

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        data = data.sort_index()
        data = data.dropna(subset=required_columns)
        return data

    def run_backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        fees: float = 0.0,
        slippage: float = 0.0,
        **kwargs,
    ):
        self.data = self._validate_data(data)
        entries_long, exits_long, entries_short, exits_short = self.signal_obj.generate(self.data)

        print(
            data.shape,
            entries_short.shape,
            exits_short.shape,
            entries_long.shape,
            exits_long.shape,
        )

        self.portfolio = vbt.Portfolio.from_signals(
            close=self.data["close"],
            open=self.data["open"],
            entries=entries_long,
            exits=exits_long,
            short_entries=entries_short,
            short_exits=exits_short,
            init_cash=initial_capital,
            fees=fees,
            slippage=slippage,
            freq=self.timeframe,
            **kwargs,
        )
        return self.portfolio

    def get_stats(self) -> pd.Series:
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        return self.portfolio.stats()

    def get_cummulative_returns(self) -> pd.Series:
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        return self.portfolio.cumulative_returns()

    def plot_equity_curve(self):
        """Plot equity curve"""
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        fig = self.portfolio.plot()
        fig.show()

    def plot_drawdowns(self):
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        fig = self.portfolio.drawdowns.plot()
        fig.show()

    def plot_all_charts(self) -> Dict[str, object]:
        """Generate all standard charts"""
        fig = self.portfolio.plot(
            subplots=[
                "trades",
                "drawdowns",
                "underwater",
                "trades",
                "cum_returns",
                "trade_pnl",
            ],
            subplot_settings=dict(drawdowns=dict(top_n=3)),
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            width=1440,
            template="plotly_dark",
        )
        fig.show()

    def optimize(
        self,
        signal_class,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        initial_capital: float = 10000,
        slippage: float = 0.0,
        fees: float = 0.0,
        metric: str = "total_return",
        maximize: bool = True,
        **kwargs,
    ):
        self.data = self._validate_data(data)
        keys = list(param_grid.keys())
        records, portfolios = [], []
        params_product_list = list(itertools.product(*param_grid.values()))
        for vals in tqdm(
            params_product_list, desc="Searching optimal parameters via backtesting..."
        ):
            params = dict(zip(keys, vals))
            signal_obj = signal_class(**params)
            entries_long, exits_long, entries_short, exits_short = signal_obj.generate(self.data)
            pf = vbt.Portfolio.from_signals(
                close=self.data["close"],
                open=self.data["open"],
                entries=entries_long,
                exits=exits_long,
                short_entries=entries_short,
                short_exits=exits_short,
                freq=self.timeframe,
                fees=fees,
                init_cash=initial_capital,
                slippage=slippage,
            )

            stats = pf.stats()
            row = stats.to_frame().T.assign(**params)
            row["Score"] = stats[metric]
            records.append(row)
            portfolios.append((pf, params))
        results = pd.concat(records, ignore_index=True)
        idx_best = results["Score"].idxmax() if maximize else results["Score"].idxmin()
        best_params = results.loc[idx_best, keys].to_dict()
        self.optimization_results = portfolios[idx_best][0]
        results.sort_values(by=[metric], ascending=False, inplace=True)
        return results, self.optimization_results, best_params

    def get_optimization_stats(self) -> pd.DataFrame:
        if self.optimization_results is None:
            raise ValueError("Run optimization first using optimize()")
        return self.optimization_results.stats

    def save_results(
        self,
        filepath: Union[str, Path],
        save_portfolio: bool = True,
        save_optimization: bool = True,
    ) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "metadata": {
                "timeframe": self.timeframe,
                "signal_obj_class": self.signal_obj.__class__.__name__,
                # Avoid trying to pickle the signal_obj itself if it contains unpickleable stuff
                # 'signal_params': getattr(self.signal_obj, 'params', {})
            }
        }
        if save_portfolio and self.portfolio is not None:
            stats = self.portfolio.stats()
            # Convert potentially problematic objects to serializable formats
            save_dict["portfolio_stats"] = (
                stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
            )
            # Save key attributes instead of the full object
            save_dict["portfolio_key_data"] = {
                "init_cash": getattr(self.portfolio, "init_cash", None),
                "final_value": getattr(self.portfolio, "final_value", None),
                # Add other simple attributes you need
            }
            # Try to save portfolio object, but catch joblib errors
            try:
                joblib.dump(self.portfolio, filepath.with_suffix(".portfolio.joblib"))
            except Exception as e:
                print(
                    f"Warning: Could not save portfolio object: {e}. Saved key data instead."
                )
                save_dict["portfolio_key_data"]["save_error"] = str(e)

        if save_optimization and self.optimization_results is not None:
            # Try to get stats and best_params, convert if needed
            try:
                opt_stats = self.optimization_results.stats
                save_dict["optimization_stats"] = (
                    opt_stats.to_dict()
                    if hasattr(opt_stats, "to_dict")
                    else dict(opt_stats)
                )
            except Exception as e:
                print(f"Warning: Could not get/save optimization stats: {e}")
                save_dict["optimization_stats"] = {"error": str(e)}

            try:
                save_dict["best_params"] = self.optimization_results.best_params
            except Exception as e:
                print(f"Warning: Could not get/save best_params: {e}")
                save_dict["best_params"] = {"error": str(e)}

            # Try to save optimization object, but catch joblib errors
            try:
                joblib.dump(
                    self.optimization_results,
                    filepath.with_suffix(".optimization.joblib"),
                )
            except Exception as e:
                print(
                    f"Warning: Could not save optimization object: {e}. Saved stats/best_params instead."
                )
                # Error info is already in save_dict

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(
                save_dict, f, default=str, indent=2
            )  # default=str handles non-serializable objects

    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        filepath = Path(filepath)
        with open(filepath.with_suffix(".json"), "r") as f:
            results = json.load(f)
        return results

    def get_trades(self) -> pd.DataFrame:
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        return self.portfolio.trades.records_readable

    def get_positions(self) -> pd.DataFrame:
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        return self.portfolio.positions.records_readable

    def generate_report(self) -> Dict[str, Any]:
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        stats = self.get_stats()
        trades = self.get_trades()
        report = {
            "summary_stats": stats.to_dict(),
            "performance_metrics": {
                "total_return": float(stats.get("Total Return [%]", 0)),
                "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)),
                "max_drawdown": float(stats.get("Max Drawdown [%]", 0)),
                "win_rate": float(stats.get("Win Rate [%]", 0)),
                "profit_factor": float(stats.get("Profit Factor", 0)),
                "expectancy": float(stats.get("Expectancy", 0)),
            },
            "trade_analysis": {
                "total_trades": len(trades),
                "winning_trades": (
                    len(trades[trades["PnL"] > 0]) if len(trades) > 0 else 0
                ),
                "losing_trades": (
                    len(trades[trades["PnL"] < 0]) if len(trades) > 0 else 0
                ),
                "avg_win": (
                    float(trades[trades["PnL"] > 0]["PnL"].mean())
                    if len(trades[trades["PnL"] > 0]) > 0
                    else 0
                ),
                "avg_loss": (
                    float(trades[trades["PnL"] < 0]["PnL"].mean())
                    if len(trades[trades["PnL"] < 0]) > 0
                    else 0
                ),
            },
        }
        return report

    def monte_carlo_simulations(self, n_simulations: int = 1000):
        """
        Performs Monte Carlo simulations by shuffling portfolio returns.
        Args:
            n_simulations (int): Number of simulations to run.
        """
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")

        # FIX 1: Call the returns method
        portfolio_returns = self.portfolio.returns()  # Get the actual returns Series
        if portfolio_returns is None or len(portfolio_returns) == 0:
            raise ValueError("Portfolio returns are empty, cannot run Monte Carlo.")

        # Use the returned Series
        original_returns = portfolio_returns.values  # Get numpy array
        # n_returns = len(original_returns) # Not strictly needed here
        sim_final_values = np.empty(n_simulations)

        initial_cash = (
            self.portfolio.init_cash()
            if callable(self.portfolio.init_cash)
            else self.portfolio.init_cash
        )

        for i in range(n_simulations):
            shuffled_returns = np.random.permutation(original_returns)
            shuffled_returns = np.nan_to_num(shuffled_returns, nan=0.0)  # Handle NaNs
            # Calculate final value directly from initial cash and shuffled returns
            cumulative_return = np.prod(1 + shuffled_returns) - 1
            sim_final_values[i] = initial_cash * (1 + cumulative_return)

        # FIX 3: Call the final_value method/property
        final_value = (
            self.portfolio.final_value()
            if callable(self.portfolio.final_value)
            else self.portfolio.final_value
        )
        original_cumulative_return = (final_value / initial_cash) - 1

        self.mc_simulations = {
            "simulated_final_values": sim_final_values,
            "n_simulations": n_simulations,
            "initial_cash": initial_cash,
            "original_cumulative_return": original_cumulative_return,  # Store the calculated value
        }
        return self.mc_simulations

    def get_monte_carlo_stats(self):
        """
        Calculates statistics from the Monte Carlo simulations.
        """
        if self.mc_simulations is None:
            raise ValueError(
                "Run Monte Carlo simulation first using monte_carlo_simulations()"
            )

        sim_values = self.mc_simulations["simulated_final_values"]
        initial_cash = self.mc_simulations["initial_cash"]
        original_cumulative_return = self.mc_simulations["original_cumulative_return"]
        original_final_value = initial_cash * (1 + original_cumulative_return)

        mean_final_value = np.mean(sim_values)
        median_final_value = np.median(sim_values)
        std_final_value = np.std(sim_values)

        sim_returns = (sim_values / initial_cash) - 1
        mean_return = np.mean(sim_returns)
        median_return = np.median(sim_returns)
        std_return = np.std(sim_returns)

        percentile_5 = np.percentile(sim_values, 5)
        percentile_95 = np.percentile(sim_values, 95)

        n_better = np.sum(sim_values > original_final_value)
        prob_outperformance = n_better / len(sim_values)

        losses = initial_cash - sim_values
        var_95 = np.percentile(losses, 95)

        stats = {
            "mean_final_value": mean_final_value,
            "median_final_value": median_final_value,
            "std_final_value": std_final_value,
            "mean_simulated_return": mean_return,
            "median_simulated_return": median_return,
            "std_simulated_return": std_return,
            "percentile_5_final_value": percentile_5,
            "percentile_95_final_value": percentile_95,
            "original_final_value": original_final_value,
            "original_cumulative_return": original_cumulative_return,
            "probability_of_outperformance": prob_outperformance,
            "value_at_risk_95_abs": var_95,
            "value_at_risk_95_pct": (var_95 / initial_cash) * 100,
        }
        return pd.Series(stats)

    def analyze_by_regime(
        self,
        regime_labels: pd.Series,
        regime_names: Optional[Dict[int, str]] = None,
        annualization_factor: int = 252,
    ):
        """Return a RegimePerformanceAnalyzer for the last run portfolio.

        Args:
            regime_labels: DatetimeIndex Series of integer regime IDs.
            regime_names: Optional mapping from regime ID to display name.
            annualization_factor: Bars per year (252 daily, 8760 hourly, etc.).

        Returns:
            RegimePerformanceAnalyzer instance.
        """
        if self.portfolio is None:
            raise ValueError("Run backtest first using run_backtest()")
        from .regime_performance_analyzer import RegimePerformanceAnalyzer
        return RegimePerformanceAnalyzer(
            self.portfolio, regime_labels, regime_names, annualization_factor
        )

    def plot_monte_carlo_returns_distribution(self, bins: int = 50):
        """
        Plots the distribution of final returns from Monte Carlo simulations using Matplotlib.
        """
        if self.mc_simulations is None:
            raise ValueError(
                "Run Monte Carlo simulation first using monte_carlo_simulations()"
            )

        sim_values = self.mc_simulations["simulated_final_values"]
        initial_cash = self.mc_simulations["initial_cash"]
        original_final_value = self.mc_simulations["initial_cash"] * (
            1 + self.mc_simulations["original_cumulative_return"]
        )

        sim_returns = (sim_values / initial_cash) - 1
        original_return = (original_final_value / initial_cash) - 1

        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins_plot, patches = ax.hist(
            sim_returns * 100, bins=bins, edgecolor="black", alpha=0.7
        )
        ax.set_xlabel("Simulated Final Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo Simulation: Distribution of Final Returns")
        ax.axvline(
            original_return * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Original Strategy Return ({original_return * 100:.2f}%)",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        return fig
