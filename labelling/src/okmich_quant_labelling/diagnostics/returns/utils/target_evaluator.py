from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression


class RegressionTargetEvaluator:
    """
    Comprehensive regression target analyzer for parameter optimization.

    Evaluates target quality across multiple dimensions:
    1. Distribution characteristics (mean, std, skewness, kurtosis)
    2. Stationarity (ADF, KPSS tests)
    3. Autocorrelation (ACF, signal persistence)
    4. Signal quality (SNR, information ratio)
    5. Predictability (correlation with forward returns, mutual information)
    6. Segment analysis (stability, consistency)
    7. Economic metrics (Sharpe-like ratios, coverage)

    Primary use cases:
    - Parameter optimization: Compare different omega/minamp/Tinactive values
    - Target type selection: Compare MOMENTUM vs SLOPE vs CUMULATIVE_RETURN
    - Labeler comparison: AutoLabel vs Amplitude-based
    - Quality assessment: Is this target suitable for modeling?
    """

    def __init__(self, df: pd.DataFrame, target_col: str, price_col: str = "close", return_col: Optional[str] = None):
        """
        Initialize evaluator.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with targets and prices
        target_col : str
            Column name for regression targets
        price_col : str, default="close"
            Column name for prices
        return_col : str, optional
            Column name for returns. If None, computed from price_col.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.price_col = price_col

        # Compute returns if not provided
        if return_col is None:
            self.df['_returns'] = self.df[price_col].pct_change()
            self.return_col = '_returns'
        else:
            self.return_col = return_col

        self.targets = self.df[target_col].dropna()

    def evaluate(self, forward_periods: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Run comprehensive evaluation.

        Parameters
        ----------
        forward_periods : list of int
            Forward return periods to test predictability

        Returns
        -------
        dict
            Nested dictionary with all evaluation metrics
        """
        results = {
            'basic_stats': self._basic_statistics(),
            'distribution': self._distribution_analysis(),
            'stationarity': self._stationarity_tests(),
            'autocorrelation': self._autocorrelation_analysis(),
            'signal_quality': self._signal_quality_metrics(),
            'predictability': self._predictability_analysis(forward_periods),
            'segment_analysis': self._segment_analysis(),
            'economic_metrics': self._economic_metrics(),
            'composite_score': None,  # Computed last
        }

        # Compute composite score
        results['composite_score'] = self._compute_composite_score(results)

        return results

    def _basic_statistics(self) -> Dict:
        """Basic statistical summary of targets."""
        targets_clean = self.targets[~np.isnan(self.targets)]

        if len(targets_clean) == 0:
            return {'error': 'No valid targets'}

        return {
            'count': int(len(targets_clean)),
            'n_nonzero': int(np.sum(targets_clean != 0)),
            'pct_nonzero': float(np.sum(targets_clean != 0) / len(targets_clean) * 100),
            'mean': float(np.mean(targets_clean)),
            'median': float(np.median(targets_clean)),
            'std': float(np.std(targets_clean)),
            'min': float(np.min(targets_clean)),
            'max': float(np.max(targets_clean)),
            'range': float(np.max(targets_clean) - np.min(targets_clean)),
            'q05': float(np.percentile(targets_clean, 5)),
            'q25': float(np.percentile(targets_clean, 25)),
            'q75': float(np.percentile(targets_clean, 75)),
            'q95': float(np.percentile(targets_clean, 95)),
            'iqr': float(np.percentile(targets_clean, 75) - np.percentile(targets_clean, 25)),
        }

    def _distribution_analysis(self) -> Dict:
        """Analyze target value distribution."""
        targets_clean = self.targets[~np.isnan(self.targets)]

        if len(targets_clean) < 3:
            return {'error': 'Insufficient data'}

        # Moments
        skewness = float(stats.skew(targets_clean))
        kurtosis = float(stats.kurtosis(targets_clean))

        # Normality tests
        _, shapiro_p = stats.shapiro(targets_clean[:min(5000, len(targets_clean))])  # Limit to 5000 for speed
        _, jarque_bera_p = stats.jarque_bera(targets_clean)

        # Outlier detection
        q1 = np.percentile(targets_clean, 25)
        q3 = np.percentile(targets_clean, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        n_outliers = np.sum((targets_clean < lower_fence) | (targets_clean > upper_fence))
        outlier_pct = n_outliers / len(targets_clean) * 100

        # Tail analysis
        n_positive = np.sum(targets_clean > 0)
        n_negative = np.sum(targets_clean < 0)
        n_zero = np.sum(targets_clean == 0)
        sign_balance = min(n_positive, n_negative) / max(n_positive, n_negative) if max(n_positive, n_negative) > 0 else 0

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': (shapiro_p > 0.05 and jarque_bera_p > 0.05),
            'shapiro_p': float(shapiro_p),
            'jarque_bera_p': float(jarque_bera_p),
            'n_outliers': int(n_outliers),
            'outlier_pct': float(outlier_pct),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'n_zero': int(n_zero),
            'sign_balance': float(sign_balance),  # 1.0 = perfect balance, 0.0 = completely imbalanced
        }

    def _stationarity_tests(self) -> Dict:
        """Test for stationarity (critical for modeling)."""
        targets_clean = self.targets[~np.isnan(self.targets)]

        if len(targets_clean) < 20:
            return {'error': 'Insufficient data for stationarity tests'}

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # Augmented Dickey-Fuller test (null: non-stationary)
            adf_result = adfuller(targets_clean, autolag='AIC')
            adf_statistic = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            is_stationary_adf = adf_pvalue < 0.05

            # KPSS test (null: stationary)
            # Suppress InterpolationWarning - it's informational, not actionable
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='The test statistic is outside')
                kpss_result = kpss(targets_clean, regression='c', nlags='auto')
            kpss_statistic = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
            is_stationary_kpss = kpss_pvalue > 0.05

            # Both tests agree
            is_stationary = is_stationary_adf and is_stationary_kpss

            return {
                'is_stationary': is_stationary,
                'adf_statistic': adf_statistic,
                'adf_pvalue': adf_pvalue,
                'is_stationary_adf': is_stationary_adf,
                'kpss_statistic': kpss_statistic,
                'kpss_pvalue': kpss_pvalue,
                'is_stationary_kpss': is_stationary_kpss,
            }
        except ImportError:
            return {'error': 'statsmodels not available for stationarity tests'}
        except Exception as e:
            return {'error': f'Stationarity test failed: {str(e)}'}

    def _autocorrelation_analysis(self) -> Dict:
        """Analyze temporal dependencies in targets."""
        targets_clean = self.targets[~np.isnan(self.targets)]

        if len(targets_clean) < 20:
            return {'error': 'Insufficient data for autocorrelation analysis'}

        # Lag-1 autocorrelation
        acf_lag1 = float(pd.Series(targets_clean).autocorr(lag=1))

        # Multiple lags
        acf_values = {}
        for lag in [1, 2, 3, 5, 10]:
            if len(targets_clean) > lag:
                acf_values[f'acf_lag{lag}'] = float(pd.Series(targets_clean).autocorr(lag=lag))

        # Ljung-Box test for serial correlation (null: no autocorrelation)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(targets_clean, lags=[10], return_df=False)
            lb_statistic = float(lb_result[0][0])
            lb_pvalue = float(lb_result[1][0])
            has_autocorrelation = lb_pvalue < 0.05
        except:
            lb_statistic = np.nan
            lb_pvalue = np.nan
            has_autocorrelation = None

        return {
            'acf_lag1': acf_lag1,
            **acf_values,
            'ljung_box_statistic': lb_statistic,
            'ljung_box_pvalue': lb_pvalue,
            'has_significant_autocorr': has_autocorrelation,
        }

    def _signal_quality_metrics(self) -> Dict:
        """Evaluate signal quality and informativeness."""
        targets_clean = self.targets[~np.isnan(self.targets)]

        if len(targets_clean) < 10:
            return {'error': 'Insufficient data'}

        # Signal-to-noise ratio
        mean_abs = np.mean(np.abs(targets_clean))
        std = np.std(targets_clean)
        snr = mean_abs / std if std > 0 else 0

        # Information ratio (mean / std)
        information_ratio = np.mean(targets_clean) / std if std > 0 else 0

        # Target persistence (what % of time does sign stay the same?)
        target_signs = np.sign(targets_clean)
        sign_changes = np.sum(np.diff(target_signs) != 0)
        persistence = 1 - (sign_changes / (len(target_signs) - 1))

        # Effective sample size (accounting for autocorrelation)
        acf_lag1 = pd.Series(targets_clean).autocorr(lag=1)
        if not np.isnan(acf_lag1) and acf_lag1 < 1:
            effective_n = len(targets_clean) * (1 - acf_lag1) / (1 + acf_lag1)
        else:
            effective_n = len(targets_clean)

        # Dynamic range utilization
        theoretical_range = np.abs(np.percentile(targets_clean, 99) - np.percentile(targets_clean, 1))
        actual_range = np.max(targets_clean) - np.min(targets_clean)
        range_utilization = actual_range / theoretical_range if theoretical_range > 0 else 0

        return {
            'snr': float(snr),
            'information_ratio': float(information_ratio),
            'target_persistence': float(persistence),
            'effective_sample_size': float(effective_n),
            'range_utilization': float(range_utilization),
        }

    def _predictability_analysis(self, forward_periods: List[int]) -> Dict:
        """Analyze how well targets predict future returns."""
        results = {}

        for period in forward_periods:
            # Compute forward returns
            fwd_returns = self.df[self.price_col].pct_change(period).shift(-period)

            # Align with targets
            mask = ~(self.df[self.target_col].isna() | fwd_returns.isna())
            if mask.sum() < 10:
                results[f'period_{period}'] = {'error': 'Insufficient data'}
                continue

            targets = self.df.loc[mask, self.target_col].values
            returns = fwd_returns[mask].values

            # Correlation
            pearson_corr = float(np.corrcoef(targets, returns)[0, 1])
            spearman_corr, _ = spearmanr(targets, returns)
            spearman_corr = float(spearman_corr)

            # Directional accuracy
            target_signs = np.sign(targets)
            return_signs = np.sign(returns)
            directional_acc = float(np.mean(target_signs == return_signs))

            # Mutual information
            try:
                mi = float(mutual_info_regression(targets.reshape(-1, 1), returns, random_state=42)[0])
            except:
                mi = np.nan

            # R² (if we predict returns using targets)
            r2 = float(r2_score(returns, targets))

            # MSE, MAE
            mse = float(mean_squared_error(returns, targets))
            mae = float(mean_absolute_error(returns, targets))

            # IC (Information Coefficient) - Spearman correlation
            ic = spearman_corr

            results[f'period_{period}'] = {
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'directional_accuracy': directional_acc,
                'mutual_information': mi,
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'ic': ic,
            }

        # Average across periods
        avg_metrics = {}
        for metric in ['pearson_corr', 'spearman_corr', 'directional_accuracy', 'ic']:
            values = [results[f'period_{p}'].get(metric, np.nan) for p in forward_periods
                      if 'error' not in results[f'period_{p}']]
            avg_metrics[f'avg_{metric}'] = float(np.nanmean(values)) if values else np.nan

        results['average'] = avg_metrics

        return results

    def _segment_analysis(self) -> Dict:
        """Analyze target behavior within trend segments."""
        targets = self.df[self.target_col]

        # Identify segments (non-zero regions)
        nonzero_mask = targets != 0
        segment_starts = np.where(nonzero_mask & ~nonzero_mask.shift(1, fill_value=False))[0]
        segment_ends = np.where(nonzero_mask & ~nonzero_mask.shift(-1, fill_value=False))[0]

        if len(segment_starts) == 0 or len(segment_ends) == 0:
            return {'error': 'No segments found'}

        # Ensure equal lengths
        min_len = min(len(segment_starts), len(segment_ends))
        segment_starts = segment_starts[:min_len]
        segment_ends = segment_ends[:min_len]

        # Segment statistics
        segment_lengths = segment_ends - segment_starts + 1
        segment_stds = []
        segment_ranges = []

        for start, end in zip(segment_starts, segment_ends):
            seg_targets = targets.iloc[start:end + 1]
            segment_stds.append(seg_targets.std())
            segment_ranges.append(seg_targets.max() - seg_targets.min())

        return {
            'n_segments': int(len(segment_starts)),
            'avg_segment_length': float(np.mean(segment_lengths)),
            'median_segment_length': float(np.median(segment_lengths)),
            'min_segment_length': int(np.min(segment_lengths)),
            'max_segment_length': int(np.max(segment_lengths)),
            'avg_segment_std': float(np.mean(segment_stds)),
            'avg_segment_range': float(np.mean(segment_ranges)),
            'segment_coverage': float(np.sum(segment_lengths) / len(targets) * 100),  # % of time in trends
        }

    def _economic_metrics(self) -> Dict:
        """Compute economic/trading-focused metrics."""
        targets = self.df[self.target_col]
        returns = self.df[self.return_col]

        # Align
        mask = ~(targets.isna() | returns.isna())
        if mask.sum() < 10:
            return {'error': 'Insufficient data'}

        targets_clean = targets[mask]
        returns_clean = returns[mask]

        # Target-weighted returns (as if we traded proportional to target)
        weighted_returns = targets_clean * returns_clean

        # Sharpe-like ratio for targets
        mean_weighted_ret = weighted_returns.mean()
        std_weighted_ret = weighted_returns.std()
        target_sharpe = mean_weighted_ret / std_weighted_ret * np.sqrt(252) if std_weighted_ret > 0 else 0

        # Hit rate (target sign = return sign)
        hit_rate = float(np.mean(np.sign(targets_clean) == np.sign(returns_clean)))

        # Payoff ratio (avg win / avg loss when following targets)
        wins = weighted_returns[weighted_returns > 0]
        losses = weighted_returns[weighted_returns < 0]
        payoff_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.nan

        # Target turnover (how often does target change significantly?)
        target_changes = np.abs(targets_clean.diff())
        avg_turnover = target_changes.mean()

        return {
            'target_sharpe': float(target_sharpe),
            'hit_rate': hit_rate,
            'payoff_ratio': float(payoff_ratio),
            'avg_target_turnover': float(avg_turnover),
            'mean_weighted_return': float(mean_weighted_ret),
            'std_weighted_return': float(std_weighted_ret),
        }

    def _compute_composite_score(self, results: Dict) -> Dict:
        """
        Compute composite quality score (0-100 scale).

        Weights different aspects based on importance for modeling.
        """
        scores = {}
        weights = {}

        # 1. Predictability (40% weight) - most important
        pred = results.get('predictability', {}).get('average', {})
        pred_score = 0
        if 'avg_ic' in pred and not np.isnan(pred['avg_ic']):
            # IC: good if |IC| > 0.05, excellent if |IC| > 0.10
            pred_score += min(abs(pred['avg_ic']) / 0.10, 1.0) * 40
        if 'avg_directional_accuracy' in pred and not np.isnan(pred['avg_directional_accuracy']):
            # Directional accuracy: good if > 0.55, excellent if > 0.60
            pred_score += max((pred['avg_directional_accuracy'] - 0.50) / 0.10, 0) * 20
        scores['predictability'] = min(pred_score, 40)
        weights['predictability'] = 40

        # 2. Signal quality (25% weight)
        sig_qual = results.get('signal_quality', {})
        sig_score = 0
        if 'snr' in sig_qual and not np.isnan(sig_qual['snr']):
            # SNR: good if > 0.5, excellent if > 1.0
            sig_score += min(sig_qual['snr'] / 1.0, 1.0) * 15
        if 'information_ratio' in sig_qual and not np.isnan(sig_qual['information_ratio']):
            # IR: good if |IR| > 0.1, excellent if |IR| > 0.5
            sig_score += min(abs(sig_qual['information_ratio']) / 0.5, 1.0) * 10
        scores['signal_quality'] = min(sig_score, 25)
        weights['signal_quality'] = 25

        # 3. Stationarity (15% weight) - important for modeling
        stationarity = results.get('stationarity', {})
        if stationarity.get('is_stationary', False):
            scores['stationarity'] = 15
        else:
            scores['stationarity'] = 0
        weights['stationarity'] = 15

        # 4. Distribution (10% weight)
        dist = results.get('distribution', {})
        dist_score = 10  # Start with full points
        # Penalize extreme skewness (>|2|)
        if 'skewness' in dist and abs(dist['skewness']) > 2:
            dist_score -= 3
        # Penalize too many outliers (>10%)
        if 'outlier_pct' in dist and dist['outlier_pct'] > 10:
            dist_score -= 3
        # Penalize very imbalanced signs
        if 'sign_balance' in dist and dist['sign_balance'] < 0.5:
            dist_score -= 2
        scores['distribution'] = max(dist_score, 0)
        weights['distribution'] = 10

        # 5. Economic metrics (10% weight)
        econ = results.get('economic_metrics', {})
        econ_score = 0
        if 'target_sharpe' in econ and not np.isnan(econ['target_sharpe']):
            # Sharpe: good if > 0.5, excellent if > 1.0
            econ_score += min(abs(econ['target_sharpe']) / 1.0, 1.0) * 5
        if 'hit_rate' in econ and not np.isnan(econ['hit_rate']):
            # Hit rate: good if > 0.55, excellent if > 0.60
            econ_score += max((econ['hit_rate'] - 0.50) / 0.10, 0) * 5
        scores['economic'] = min(econ_score, 10)
        weights['economic'] = 10

        # Total composite score
        total_score = sum(scores.values())

        return {
            'total_score': float(total_score),
            'scores_by_category': scores,
            'weights': weights,
            'interpretation': self._interpret_score(total_score),
        }

    @staticmethod
    def _interpret_score(score: float) -> str:
        """Interpret composite score."""
        if score >= 80:
            return "Excellent - High-quality target suitable for modeling"
        elif score >= 65:
            return "Good - Suitable for modeling with reasonable predictive power"
        elif score >= 50:
            return "Fair - Moderate quality, consider parameter tuning"
        elif score >= 35:
            return "Poor - Low quality, significant parameter optimization needed"
        else:
            return "Very Poor - Target not suitable for modeling"


def compare_regression_targets(df: pd.DataFrame, target_configs: List[Tuple[str, Dict]],
                               forward_periods: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Compare multiple regression target configurations side by side.

    Primary use case: Parameter optimization - compare different omega/minamp/Tinactive values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with prices and all target columns
    target_configs : list of tuple
        List of (target_col_name, config_dict) tuples
        config_dict should contain: {'labeler': 'AutoLabel', 'omega': 0.02, 'target_type': 'MOMENTUM', ...}
    forward_periods : list of int
        Forward return periods for predictability testing

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics side by side

    Examples
    --------
    >>> configs = [
    ...     ('target_omega_01', {'labeler': 'AutoLabel', 'omega': 0.01, 'target_type': 'MOMENTUM'}),
    ...     ('target_omega_02', {'labeler': 'AutoLabel', 'omega': 0.02, 'target_type': 'MOMENTUM'}),
    ...     ('target_omega_03', {'labeler': 'AutoLabel', 'omega': 0.03, 'target_type': 'MOMENTUM'}),
    ... ]
    >>> comparison_df = compare_regression_targets(df, configs)
    >>> # Sort by composite score to find best parameters
    >>> comparison_df.sort_values('composite_score', ascending=False)
    """
    results_list = []

    for target_col, config in target_configs:
        if target_col not in df.columns:
            print(f"Warning: {target_col} not found in DataFrame, skipping")
            continue

        evaluator = RegressionTargetEvaluator(df, target_col)
        results = evaluator.evaluate(forward_periods=forward_periods)

        # Flatten results for DataFrame
        row = {'target_name': target_col}
        row.update(config)

        # Add key metrics
        row['composite_score'] = results['composite_score']['total_score']
        row['interpretation'] = results['composite_score']['interpretation']

        # Basic stats
        row.update({f'basic_{k}': v for k, v in results['basic_stats'].items() if k != 'error'})

        # Distribution
        dist = results['distribution']
        row['skewness'] = dist.get('skewness', np.nan)
        row['kurtosis'] = dist.get('kurtosis', np.nan)
        row['sign_balance'] = dist.get('sign_balance', np.nan)

        # Stationarity
        row['is_stationary'] = results['stationarity'].get('is_stationary', False)

        # Signal quality
        sig_qual = results['signal_quality']
        row['snr'] = sig_qual.get('snr', np.nan)
        row['information_ratio'] = sig_qual.get('information_ratio', np.nan)
        row['persistence'] = sig_qual.get('target_persistence', np.nan)

        # Predictability (average)
        pred_avg = results['predictability'].get('average', {})
        row['avg_ic'] = pred_avg.get('avg_ic', np.nan)
        row['avg_dir_acc'] = pred_avg.get('avg_directional_accuracy', np.nan)
        row['avg_corr'] = pred_avg.get('avg_pearson_corr', np.nan)

        # Economic
        econ = results['economic_metrics']
        row['target_sharpe'] = econ.get('target_sharpe', np.nan)
        row['hit_rate'] = econ.get('hit_rate', np.nan)

        # Segments
        seg = results['segment_analysis']
        row['n_segments'] = seg.get('n_segments', np.nan)
        row['segment_coverage'] = seg.get('segment_coverage', np.nan)

        results_list.append(row)

    comparison_df = pd.DataFrame(results_list)

    # Sort by composite score (best first)
    if 'composite_score' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('composite_score', ascending=False).reset_index(drop=True)

    return comparison_df


def optimize_target_parameters(df: pd.DataFrame, labeler_class, param_grid: Dict[str, List], target_type,
                               price_col: str = "close", top_n: int = 5) -> pd.DataFrame:
    """
    Grid search for optimal labeler parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    labeler_class : class
        AutoLabelRegression or AmplitudeBasedRegressionLabeler
    param_grid : dict
        Parameter grid, e.g., {'omega': [0.01, 0.02, 0.03]} or
        {'minamp': [100, 200, 300], 'Tinactive': [10, 20]}
    target_type : RegressionTargetType
        Target type to use
    price_col : str
        Price column name
    top_n : int
        Return top N parameter combinations

    Returns
    -------
    pd.DataFrame
        Top N parameter combinations ranked by composite score

    Examples
    --------
    >>> from okmich_quant_labelling.diagnostics.returns import AutoLabelRegression, RegressionTargetType
    >>> param_grid = {'omega': [0.01, 0.015, 0.02, 0.025, 0.03]}
    >>> top_params = optimize_target_parameters(
    ...     df, AutoLabelRegression, param_grid,
    ...     RegressionTargetType.MOMENTUM, top_n=3
    ... )
    >>> print(top_params[['omega', 'composite_score', 'interpretation']])
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    target_configs = []

    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))

        # Create labeler and generate targets
        if 'omega' in params:
            # AutoLabelRegression
            labeler = labeler_class(
                omega=params['omega'],
                target_type=target_type,
                normalize=False,
            )
            if price_col in df.columns:
                series = df[price_col]
            else:
                series = df.iloc[:, 0]  # Use first column if price_col not found
            targets = labeler.label(series)
            target_col_name = f"target_{i}"
            df[target_col_name] = targets
            config = {'labeler': 'AutoLabel', **params, 'target_type': target_type.value}

        else:
            # AmplitudeBasedRegressionLabeler
            labeler = labeler_class(
                minamp=params['minamp'],
                Tinactive=params['Tinactive'],
                target_type=target_type,
                normalize=False,
            )
            targets = labeler.label(df, price_col=price_col)
            target_col_name = f"target_{i}"
            df[target_col_name] = targets
            config = {'labeler': 'Amplitude', **params, 'target_type': target_type.value}

        target_configs.append((target_col_name, config))

    # Compare all configurations
    comparison_df = compare_regression_targets(df, target_configs)

    # Return top N
    return comparison_df.head(top_n)
