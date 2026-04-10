import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


class RegimeQualityEvaluator:
    """
    implementation of regime quality assessment for HMM-detected market regimes.
    Implements statistical validation, economic interpretation, and robustness checks.
    """

    def __init__(self, model_impl, random_state=42):
        self.n_regimes = model_impl.n_states
        self.random_state = random_state
        self.model_impl = model_impl
        self.regime_stats = {}
        self.transition_matrix = None
        self.regime_labels = {}

    def _calculate_regime_stats(self, returns, hidden_states):
        """Calculate statistical properties for each regime"""
        self.regime_stats = {}

        for state in range(self.n_regimes):
            state_returns = returns[hidden_states == state]

            self.regime_stats[state] = {
                "mean_return": np.mean(state_returns),
                "volatility": np.std(state_returns),
                "skewness": stats.skew(state_returns),
                "kurtosis": stats.kurtosis(state_returns),
                "count": len(state_returns),
                "duration": self._calculate_regime_duration(hidden_states, state),
            }

    def _calculate_regime_duration(self, hidden_states, state):
        """Calculate average duration of a regime"""
        state_changes = np.where(np.diff(hidden_states) != 0)[0]
        durations = np.diff(state_changes, prepend=-1)
        return np.mean(durations[hidden_states[state_changes] == state])

    def _calculate_transition_matrix(self, hidden_states):
        """Calculate regime transition probability matrix"""
        n_states = self.n_regimes
        transition_matrix = np.zeros((n_states, n_states))

        for i in range(len(hidden_states) - 1):
            current_state = hidden_states[i]
            next_state = hidden_states[i + 1]
            transition_matrix[current_state, next_state] += 1

        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]

        return transition_matrix

    def _label_regimes(self):
        """Automatically label regimes based on statistical properties"""
        # Sort regimes by mean return
        sorted_states = sorted(
            self.regime_stats.keys(),
            key=lambda x: self.regime_stats[x]["mean_return"],
            reverse=True,
        )

        self.regime_labels[sorted_states[0]] = "Bull"
        self.regime_labels[sorted_states[-1]] = "Bear"
        if self.n_regimes == 3:
            self.regime_labels[sorted_states[1]] = "Sideways"

        else:
            for state in sorted_states[1:-1]:
                self.regime_labels[state] = f"Neutral_{state}"

    def validate_regimes(self, returns, hidden_states):
        """
        Comprehensive regime quality assessment
        Returns dictionary with all validation metrics
        """
        validation_results = {}

        # 1. Statistical Validation
        validation_results["statistical"] = self._statistical_validation(
            returns, hidden_states
        )
        # 2. Economic Interpretation
        validation_results["economic"] = self._economic_validation(
            returns, hidden_states
        )

        return validation_results

    def _statistical_validation(self, returns, hidden_states):
        """Statistical validation of regimes"""
        results = {}
        # Regime persistence
        results["regime_durations"] = {
            self.regime_labels[state]: stats["duration"]
            for state, stats in self.regime_stats.items()
        }

        # Transition probabilities
        labeled_tm = pd.DataFrame(
            self.transition_matrix,
            index=[self.regime_labels[i] for i in range(self.n_regimes)],
            columns=[self.regime_labels[i] for i in range(self.n_regimes)],
        )
        results["transition_matrix"] = labeled_tm
        results["aic"], results["bic"] = self.model_impl.score()
        results["regime_separation"] = self._anova_test(returns, hidden_states)

        return results

    def _economic_validation(self, returns, hidden_states):
        """Economic interpretation of regimes"""
        results = {}

        # Factor performance differentiation
        results["factor_performance"] = {
            self.regime_labels[state]: stats["mean_return"]
            for state, stats in self.regime_stats.items()
        }

        # Risk-adjusted returns
        results["sharpe_ratios"] = {}
        for state, stats in self.regime_stats.items():
            if stats["volatility"] > 0:
                results["sharpe_ratios"][self.regime_labels[state]] = (
                    stats["mean_return"] / stats["volatility"]
                )
            else:
                results["sharpe_ratios"][self.regime_labels[state]] = np.nan

        # Drawdown analysis
        results["max_drawdowns"] = self._calculate_drawdowns(returns, hidden_states)

        return results

    def _anova_test(self, returns, hidden_states):
        """ANOVA test for regime separation"""
        regime_returns = [returns[hidden_states == i] for i in range(self.n_regimes)]
        f_stat, p_value = stats.f_oneway(*regime_returns)
        return {"f_statistic": f_stat, "p_value": p_value}

    def _calculate_drawdowns(self, returns, hidden_states):
        """Calculate maximum drawdown for each regime"""
        drawdowns = {}
        for state in range(self.n_regimes):
            state_returns = returns[hidden_states == state]
            cum_returns = np.cumprod(1 + state_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            drawdowns[self.regime_labels[state]] = np.min(drawdown)
        return drawdowns

    def plot_regimes(self, returns, hidden_states, title="Market Regimes"):
        """Plot returns with regime overlays"""
        plt.figure(figsize=(15, 8))

        # Plot returns
        plt.plot(returns.index, returns, color="gray", alpha=0.5, label="Returns")

        # Plot regime backgrounds
        unique_states = np.unique(hidden_states)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))

        for i, state in enumerate(unique_states):
            mask = hidden_states == state
            plt.fill_between(
                returns.index,
                np.min(returns),
                np.max(returns),
                where=mask,
                color=colors[i],
                alpha=0.3,
                label=self.regime_labels[state],
            )

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_transition_matrix(self):
        """Plot transition probability matrix"""
        plt.figure(figsize=(8, 6))
        labeled_tm = pd.DataFrame(
            self.transition_matrix,
            index=[self.regime_labels[i] for i in range(self.n_regimes)],
            columns=[self.regime_labels[i] for i in range(self.n_regimes)],
        )
        sns.heatmap(labeled_tm, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Regime Transition Probabilities")
        plt.show()

    def plot_regime_distributions(self, returns, hidden_states):
        """Plot return distributions by regime"""
        plt.figure(figsize=(12, 6))

        for state in range(self.n_regimes):
            state_returns = returns[hidden_states == state]
            sns.kdeplot(state_returns, label=self.regime_labels[state], shade=True)

        plt.title("Return Distributions by Regime")
        plt.xlabel("Returns")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regime_statistics(self, returns, hidden_states):
        """
        Plot regime distributions with statistical properties
        Shows KDE plots with key statistics for each regime
        """
        # Create figure with subplots
        fig, axes = plt.subplots(self.n_regimes, 1, figsize=(12, 4 * self.n_regimes))

        if self.n_regimes == 1:
            axes = [axes]  # Make it iterable

        for i, ax in enumerate(axes):
            regime_label = self.regime_labels[i]
            state_returns = returns[hidden_states == i]

            # Plot KDE
            sns.kdeplot(state_returns, ax=ax, shade=True, color=f"C{i}")

            # Plot mean line
            mean_return = self.regime_stats[i]["mean_return"]
            ax.axvline(
                mean_return,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_return:.4f}",
            )

            # Add text box with statistics
            stats_text = (
                f"Mean: {mean_return:.4f}\n"
                f"Volatility: {self.regime_stats[i]['volatility']:.4f}\n"
                f"Skewness: {self.regime_stats[i]['skewness']:.2f}\n"
                f"Kurtosis: {self.regime_stats[i]['kurtosis']:.2f}\n"
                f"Duration: {self.regime_stats[i]['duration']:.1f} days\n"
                f"Observations: {self.regime_stats[i]['count']}"
            )

            # Position text box
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Set titles and trend
            ax.set_title(
                f"{regime_label} Regime Distribution", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Returns")
            ax.set_ylabel("Density")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_report(self, returns, hidden_states):
        """Generate comprehensive regime quality report"""
        validation_results = self.validate_regimes(returns, hidden_states)

        print("=" * 80)
        print("REGIME QUALITY ASSESSMENT REPORT")
        print("=" * 80)

        print("\n1. STATISTICAL VALIDATION")
        print("-" * 40)
        print("Regime Durations:")
        for regime, duration in validation_results["statistical"][
            "regime_durations"
        ].items():
            print(f"  {regime}: {duration:.1f} days")

        print("\nTransition Matrix:")
        print(validation_results["statistical"]["transition_matrix"])

        print(f"\nModel Fit Metrics:")
        print(f"  AIC: {validation_results['statistical']['aic']:.2f}")
        print(f"  BIC: {validation_results['statistical']['bic']:.2f}")

        print(f"\nRegime Separation (ANOVA):")
        print(
            f"  F-statistic: {validation_results['statistical']['regime_separation']['f_statistic']:.2f}"
        )
        print(
            f"  p-value: {validation_results['statistical']['regime_separation']['p_value']:.4f}"
        )

        print("\n2. ECONOMIC INTERPRETATION")
        print("-" * 40)
        print("Factor Performance (Mean Returns):")
        for regime, perf in validation_results["economic"][
            "factor_performance"
        ].items():
            print(f"  {regime}: {perf:.4f}")

        print("\nRisk-Adjusted Returns (Sharpe Ratios):")
        for regime, sharpe in validation_results["economic"]["sharpe_ratios"].items():
            print(f"  {regime}: {sharpe:.2f}")

        print("\nMaximum Drawdowns:")
        for regime, dd in validation_results["economic"]["max_drawdowns"].items():
            print(f"  {regime}: {dd:.2%}")

        print("\n3. ROBUSTNESS CHECKS")
        print("-" * 40)
        print("Regime Sensitivity (AIC/BIC):")
        for n in range(2, 6):
            aic = validation_results["robustness"]["regime_sensitivity"]["aic"][n]
            bic = validation_results["robustness"]["regime_sensitivity"]["bic"][n]
            print(f"  {n} regimes - AIC: {aic:.2f}, BIC: {bic:.2f}")

        print("\nModel Specification Comparison (Log-Likelihood):")
        for spec, ll in validation_results["robustness"][
            "model_specifications"
        ].items():
            print(f"  {spec}: {ll:.2f}")

        print("\nSub-period Stability:")
        for i, stats in enumerate(
            validation_results["robustness"]["subperiod_stability"]
        ):
            print(f"\n  Period {i + 1}:")
            for state in range(self.n_regimes):
                regime = self.regime_labels[state]
                mean_ret = stats[state]["mean_return"]
                vol = stats[state]["volatility"]
                print(f"    {regime}: Mean={mean_ret:.4f}, Vol={vol:.4f}")

        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)

        return validation_results
