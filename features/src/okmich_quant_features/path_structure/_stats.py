import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
from numba import jit, prange


# ==================== Numba-optimized helper functions ====================
@jit(nopython=True, cache=True)
def _runs_test_numba(x):
    """Numba-optimized runs test computation for a single window."""
    # Remove NaNs
    mask = ~np.isnan(x)
    x_clean = x[mask]

    if len(x_clean) < 2:
        return np.nan

    # Compute signs
    signs = np.sign(x_clean)

    # Remove zeros
    signs_nonzero = signs[signs != 0]
    n = len(signs_nonzero)

    if n < 2:
        return np.nan

    n_pos = np.sum(signs_nonzero > 0)
    n_neg = np.sum(signs_nonzero < 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Count runs
    runs = 1
    for i in range(1, n):
        if signs_nonzero[i] != signs_nonzero[i - 1]:
            runs += 1

    # Expected runs and variance
    expected_runs = 1.0 + (2.0 * n_pos * n_neg) / (n_pos + n_neg)
    var_runs = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_pos - n_neg)) / (
        ((n_pos + n_neg) ** 2) * (n_pos + n_neg - 1.0)
    )

    if var_runs <= 0:
        return np.nan

    z_stat = (runs - expected_runs) / np.sqrt(var_runs)
    return z_stat


@jit(nopython=True, cache=True)
def _shannon_entropy_numba(x, bins):
    """Numba-optimized Shannon entropy computation for a single window."""
    # Remove NaNs
    mask = ~np.isnan(x)
    x_clean = x[mask]

    if len(x_clean) < bins:
        return np.nan

    # Compute histogram manually (np.histogram not supported in nopython mode)
    x_min = np.min(x_clean)
    x_max = np.max(x_clean)

    if x_min == x_max:
        return np.nan

    # Create bins
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    hist = np.zeros(bins)

    # Count values in each bin
    for val in x_clean:
        # Find which bin this value belongs to
        bin_idx = int((val - x_min) / (x_max - x_min) * bins)
        if bin_idx >= bins:
            bin_idx = bins - 1
        hist[bin_idx] += 1

    # Filter out zero counts
    hist_nonzero = hist[hist > 0]

    if len(hist_nonzero) == 0:
        return np.nan

    # Compute probabilities
    probs = hist_nonzero / np.sum(hist_nonzero)

    # Compute Shannon entropy
    entropy = 0.0
    for p in probs:
        entropy -= p * np.log2(p)

    # Normalize by max entropy
    max_entropy = np.log2(len(probs))
    if max_entropy == 0:
        return np.nan

    return entropy / max_entropy


@jit(nopython=True, cache=True, parallel=True)
def _rolling_runs_test_numba(data, window):
    """Numba-optimized rolling runs test using parallel processing."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in prange(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        result[i] = _runs_test_numba(window_data)

    return result


@jit(nopython=True, cache=True, parallel=True)
def _rolling_shannon_entropy_numba(data, window, bins):
    """Numba-optimized rolling Shannon entropy using parallel processing."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in prange(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        result[i] = _shannon_entropy_numba(window_data, bins)

    return result


@jit(nopython=True, cache=True)
def _kendall_tau_numba(x, y):
    """
    Numba-optimized Kendall's tau correlation coefficient.

    This is a simplified version that computes the tau-b statistic.
    """
    n = len(x)
    if n < 2:
        return np.nan

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            x_diff = x[j] - x[i]
            y_diff = y[j] - y[i]

            if x_diff * y_diff > 0:
                concordant += 1
            elif x_diff * y_diff < 0:
                discordant += 1

    total_pairs = n * (n - 1) / 2.0

    if total_pairs == 0:
        return np.nan

    tau = (concordant - discordant) / total_pairs
    return tau


@jit(nopython=True, cache=True, parallel=True)
def _rolling_kendall_tau_numba(data, window):
    """Numba-optimized rolling Kendall's tau using parallel processing."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in prange(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        time_indices = np.arange(window, dtype=np.float64)
        result[i] = _kendall_tau_numba(window_data, time_indices)

    return result


# ==================== Numba-optimized Ljung-Box ====================

@jit(nopython=True, cache=True)
def _ljung_box_stat_single(x, lags):
    """Compute Ljung-Box Q statistic for a single clean (no-NaN) array."""
    n = len(x)
    if n <= lags:
        return np.nan

    mean_x = 0.0
    for i in range(n):
        mean_x += x[i]
    mean_x /= n

    # gamma_0 = sum of squared deviations (denominator for autocorrelation)
    gamma_0 = 0.0
    for i in range(n):
        gamma_0 += (x[i] - mean_x) ** 2

    if gamma_0 < 1e-30:
        return np.nan

    q = 0.0
    for k in range(1, lags + 1):
        gamma_k = 0.0
        for i in range(k, n):
            gamma_k += (x[i] - mean_x) * (x[i - k] - mean_x)
        rho_k = gamma_k / gamma_0
        q += (rho_k * rho_k) / (n - k)

    q *= n * (n + 2)
    return q


@jit(nopython=True, cache=True, parallel=True)
def _rolling_ljung_box_numba(data, window, lags):
    """Rolling Ljung-Box Q statistic using Numba parallel processing."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in prange(window - 1, n):
        window_data = data[i - window + 1 : i + 1]

        # Count non-NaN values
        clean_count = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                clean_count += 1

        if clean_count <= lags:
            continue

        # Extract clean data if needed
        if clean_count < len(window_data):
            x_clean = np.empty(clean_count)
            idx = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    x_clean[idx] = window_data[j]
                    idx += 1
        else:
            x_clean = window_data

        result[i] = _ljung_box_stat_single(x_clean, lags)

    return result


# ==================== Numba-optimized BDS (m=2) ====================

@jit(nopython=True, cache=True)
def _bds_single_numba(x, distance):
    """
    Compute BDS test statistic and p-value for embedding dimension m=2.

    Matches the statsmodels implementation exactly:
    - epsilon = distance * std(x, ddof=1)
    - Correlation sums via upper-triangle means of indicator matrices
    - K statistic via row-sum formula
    - Variance formula from Broock et al. (1996)
    - C_1 for the BDS effect is truncated per Kanzler (1999) footnote 10
    """
    n = len(x)
    if n < 10:
        return np.nan, np.nan

    # Compute epsilon = distance * std(x, ddof=1)
    mean_x = 0.0
    for i in range(n):
        mean_x += x[i]
    mean_x /= n

    var_x = 0.0
    for i in range(n):
        var_x += (x[i] - mean_x) ** 2
    var_x /= (n - 1)  # ddof=1

    if var_x < 1e-30:
        return np.nan, np.nan

    epsilon = distance * math.sqrt(var_x)

    # Compute row sums of full indicator matrix (including diagonal, which is
    # always 1 since |x[i]-x[i]| = 0 < epsilon).  Also count the upper-triangle
    # indicator hits for C_1.
    row_sums = np.ones(n)  # diagonal entries (all 1)
    upper_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[i] - x[j]) < epsilon:
                row_sums[i] += 1.0
                row_sums[j] += 1.0
                upper_count += 1

    n_upper_full = n * (n - 1) // 2
    if n_upper_full == 0:
        return np.nan, np.nan
    c1_full = upper_count / float(n_upper_full)

    # total_sum of the full indicator matrix = n (diagonal) + 2*upper_count
    total_sum = float(n) + 2.0 * upper_count

    # K statistic  (statsmodels _var formula)
    sum_row_sq = 0.0
    for i in range(n):
        sum_row_sq += row_sums[i] * row_sums[i]
    k = (sum_row_sq - 3.0 * total_sum + 2.0 * n) / (n * (n - 1.0) * (n - 2.0))

    # C_2: upper-triangle mean of joint indicator matrix
    # joint[i,j] = I(|x[i]-x[j]|<eps) * I(|x[i+1]-x[j+1]|<eps),  0<=i,j<=n-2
    n_embed = n - 1
    n_joint_upper = n_embed * (n_embed - 1) // 2
    if n_joint_upper == 0:
        return np.nan, np.nan

    joint_upper_count = 0
    for i in range(n_embed):
        for j in range(i + 1, n_embed):
            if abs(x[i] - x[j]) < epsilon and abs(x[i + 1] - x[j + 1]) < epsilon:
                joint_upper_count += 1
    c2 = joint_upper_count / float(n_joint_upper)

    # Variance for m=2 using full-sample C_1 and K
    # var = 4*(K^2 + 2*K*C1^2 + C1^4 - 4*K*C1^2)  =  4*(K - C1^2)^2
    var = 4.0 * (k ** 2 + 2.0 * k * c1_full ** 2 + c1_full ** 4 - 4.0 * k * c1_full ** 2)
    if var <= 0:
        return np.nan, np.nan
    sd = math.sqrt(var)

    # C_1 truncated (indices 1..n-1, per Kanzler footnote 10)
    trunc_upper_count = 0
    for i in range(1, n):
        for j in range(i + 1, n):
            if abs(x[i] - x[j]) < epsilon:
                trunc_upper_count += 1
    n_trunc_upper = (n - 1) * (n - 2) // 2
    if n_trunc_upper == 0:
        return np.nan, np.nan
    c1_trunc = trunc_upper_count / float(n_trunc_upper)

    # BDS statistic: sqrt(nobs) * (C_2 - C_1_trunc^2) / sd
    nobs = n - 1  # nobs_full - ninitial, ninitial = m-1 = 1
    effect = c2 - c1_trunc ** 2
    bds_stat = math.sqrt(float(nobs)) * effect / sd

    # p-value (two-tailed normal): 2*norm.sf(|z|) = erfc(|z|/sqrt(2))
    pvalue = math.erfc(abs(bds_stat) / math.sqrt(2.0))

    return bds_stat, pvalue


@jit(nopython=True, cache=True, parallel=True)
def _rolling_bds_numba(data, window, distance):
    """Rolling BDS test (m=2) using Numba parallel processing."""
    n = len(data)
    stat_result = np.empty(n)
    pval_result = np.empty(n)
    stat_result[:] = np.nan
    pval_result[:] = np.nan

    for i in prange(window - 1, n):
        window_data = data[i - window + 1 : i + 1]

        # Count non-NaN values
        clean_count = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                clean_count += 1

        if clean_count < 10:
            continue

        if clean_count < len(window_data):
            x_clean = np.empty(clean_count)
            idx = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    x_clean[idx] = window_data[j]
                    idx += 1
        else:
            x_clean = window_data

        s, p = _bds_single_numba(x_clean, distance)
        stat_result[i] = s
        pval_result[i] = p

    return stat_result, pval_result


# ==================== Public API functions ====================


def kendall_tau(prices: pd.Series, window: int = 50, use_numba: bool = True):
    """
    Computes Kendall's Tau correlation coefficient over a rolling window.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    window : int, default=50
        Rolling window size.
    use_numba : bool, default=True
        Whether to use numba-optimized implementation (faster but less accurate).
        Set to False to use scipy's kendalltau (slower but more accurate).

    Returns
    -------
    tau_series : pd.Series
        Kendall's Tau values for each window.
    """
    if use_numba:
        # Fast numba implementation; ffill only (causal) — bfill removed to prevent look-ahead.
        values = prices.ffill().values
        tau_values = _rolling_kendall_tau_numba(values, window)
        return pd.Series(tau_values, index=prices.index)
    else:
        # Original scipy implementation (more accurate)
        tau_values = prices.rolling(window).apply(
            lambda x: kendalltau(x, np.arange(len(x)))[0], raw=False
        )
        return tau_values


def runs_test(returns: pd.Series, window: int = None):
    """
    Runs test to check randomness of signs in returns.

    Optimized with numba JIT compilation and parallel processing.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int, optional
        Rolling window size. If None, computes on entire series (single value).

    Returns
    -------
    z_stat : float or pd.Series
        Standardized test statistic. Single value if window=None, Series if window is specified.
    """
    # Global version (original behavior)
    if window is None:
        return _runs_test_numba(returns.dropna().values)

    # Rolling version with numba optimization
    values = returns.values
    result = _rolling_runs_test_numba(values, window)
    return pd.Series(result, index=returns.index)


def shannon_entropy(returns: pd.Series, bins: int = 10, window: int = None):
    """
    Computes Shannon entropy of return distribution.

    Optimized with numba JIT compilation and parallel processing.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    bins : int, default=10
        Number of bins for discretization.
    window : int, optional
        Rolling window size. If None, computes on entire series (single value).

    Returns
    -------
    entropy : float or pd.Series
        Normalized entropy (0 = predictable, 1 = random).
        Single value if window=None, Series if window is specified.
    """
    # Global version (original behavior)
    if window is None:
        return _shannon_entropy_numba(returns.dropna().values, bins)

    # Rolling version with numba optimization
    values = returns.values
    result = _rolling_shannon_entropy_numba(values, window, bins)
    return pd.Series(result, index=returns.index)


def ljung_box_test(
    returns: pd.Series, lags: int = 10, window: int = None, n_jobs: int = 1
):
    """
    Ljung–Box test for autocorrelation.

    Supports parallel processing for rolling window computation.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    lags : int, default=10
        Number of lags to test.
    window : int, optional
        Rolling window size. If None, computes on entire series (single value).
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all CPU cores.

    Returns
    -------
    lb_stat : float or pd.Series
        Test statistic. Single value if window=None, Series if window is specified.
    p_value : float or pd.Series
        P-value (low p means significant autocorrelation).
        Single value if window=None, Series if window is specified.
    """

    def _ljung_box_single(x):
        """Helper function to compute Ljung-Box test on a single window."""
        x_clean = x[~np.isnan(x)]
        if len(x_clean) <= lags:
            return np.nan, np.nan

        try:
            lb_result = acorr_ljungbox(x_clean, lags=[lags], return_df=True)
            return lb_result["lb_stat"].iloc[0], lb_result["lb_pvalue"].iloc[0]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            warnings.warn(f"ljung_box_test failed on window: {e}", stacklevel=4)
            return np.nan, np.nan

    # Global version (original behavior)
    if window is None:
        return _ljung_box_single(returns.dropna().values)

    # Rolling version with Numba-accelerated computation
    stat_values = _rolling_ljung_box_numba(returns.values, window, lags)
    pval_values = np.full(len(returns), np.nan)
    valid_mask = ~np.isnan(stat_values)
    pval_values[valid_mask] = chi2.sf(stat_values[valid_mask], df=lags)

    return pd.Series(stat_values, index=returns.index), pd.Series(
        pval_values, index=returns.index
    )


def bds_test(returns: pd.Series, max_dim: int = 2, window: int = None, n_jobs: int = 1):
    """
    BDS test for nonlinear dependence.

    Supports parallel processing for rolling window computation.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    max_dim : int, default=2
        Embedding dimension.
    window : int, optional
        Rolling window size. If None, computes on entire series (single value).
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all CPU cores.

    Returns
    -------
    bds_stat : float or pd.Series
        Test statistic. Single value if window=None, Series if window is specified.
    p_value : float or pd.Series
        P-value (low p suggests nonlinear dependence).
        Single value if window=None, Series if window is specified.
    """

    def _bds_test_single(x):
        """Helper function to compute BDS test on a single window."""
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 10:  # BDS requires reasonable sample size
            return np.nan, np.nan

        try:
            stat, pvalue = bds(x_clean, max_dim=max_dim)
            return float(np.asarray(stat).flat[0]), float(np.asarray(pvalue).flat[0])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            warnings.warn(f"bds_test failed on window: {e}", stacklevel=4)
            return np.nan, np.nan

    # Global version (original behavior)
    if window is None:
        return _bds_test_single(returns.dropna().values)

    # Rolling version with Numba-accelerated computation (m=2 fast path)
    if max_dim == 2:
        stat_values, pval_values = _rolling_bds_numba(returns.values, window, 1.5)
        return pd.Series(stat_values, index=returns.index), pd.Series(
            pval_values, index=returns.index
        )

    # Fallback for max_dim > 2: use statsmodels per-window
    n = len(returns)
    indices = range(window - 1, n)
    results = []
    for i in indices:
        window_data = returns.iloc[i - window + 1 : i + 1].values
        results.append(_bds_test_single(window_data))

    stat_values = [np.nan] * (window - 1) + [r[0] for r in results]
    pval_values = [np.nan] * (window - 1) + [r[1] for r in results]

    return pd.Series(stat_values, index=returns.index), pd.Series(
        pval_values, index=returns.index
    )
