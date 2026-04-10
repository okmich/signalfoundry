"""
CausalRegimeLabeler — fully causal market regime labeller.

Describes the current market state using only data available up to and including each bar.
Produces training targets for the Regime Classification Pipeline (Strategy C).

leaks_future = False — guaranteed, not claimed.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from okmich_quant_features.path_structure._auto_corr import auto_corr
from okmich_quant_features.volatility import atr
from okmich_quant_features.microstructure.liquidity import corwin_schultz_spread, amihud_illiquidity
from okmich_quant_features.volume._volume import tick_volume_how_zscore

from .threshold_optimizer import MarketPropertyType

# Yardsticks supported by this labeller (DIRECTIONLESS_MOMENTUM excluded)
_SUPPORTED_YARDSTICKS = {
    MarketPropertyType.DIRECTION,
    MarketPropertyType.MOMENTUM,
    MarketPropertyType.VOLATILITY,
    MarketPropertyType.PATH_STRUCTURE,
    MarketPropertyType.LIQUIDITY,
}


def _rolling_slope_r2(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling linear regression slope and R² over a fixed window.

    Parameters
    ----------
    values : np.ndarray
        1-D array of log-prices.
    window : int
        Number of bars in each regression window.

    Returns
    -------
    slopes : np.ndarray
        Regression slope per bar. NaN for bars < window - 1.
    r_squared : np.ndarray
        R² per bar. NaN for bars < window - 1.
    """
    n = len(values)
    slopes = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)
    x = np.arange(window, dtype=float)

    for i in range(window - 1, n):
        y = values[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope, _, r_value, _, _ = stats.linregress(x, y)
        slopes[i] = slope
        r_squared[i] = r_value ** 2

    return slopes, r_squared


def _apply_hysteresis(metric_pct: pd.Series, upper_pct: float, lower_pct: float, min_persistence: int,
                      r_squared: Optional[pd.Series] = None, min_r_squared: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """
    Apply adaptive thresholding + hysteresis state machine.

    Parameters
    ----------
    metric_pct : pd.Series
        Rolling percentile rank of the primary metric [0, 1].
    upper_pct : float
        Percentile above which raw candidate = +1.
    lower_pct : float
        Percentile below which raw candidate = -1.
    min_persistence : int
        Consecutive bars required before state change is committed.
    r_squared : pd.Series, optional
        R² values for trend quality filter (DIRECTION yardstick only).
    min_r_squared : float
        R² below this forces raw candidate to 0 regardless of slope.

    Returns
    -------
    labels : pd.Series
        Integer labels {-1, 0, 1} aligned to metric_pct.index. NaN where
        metric_pct is NaN.
    diagnostics : pd.DataFrame
        Per-bar trace: metric_percentile, raw_candidate, candidate_state,
        consecutive_count, label.
    """
    n = len(metric_pct)
    labels = np.full(n, np.nan)
    raw_candidates = np.full(n, np.nan)
    candidate_states = np.full(n, np.nan)
    consecutive_counts = np.zeros(n, dtype=int)

    current_state = 0
    candidate_state = 0
    consecutive_count = 0

    for i in range(n):
        pct = metric_pct.iloc[i]
        if np.isnan(pct):
            continue

        # Determine raw candidate from percentile thresholds
        if pct > upper_pct:
            raw_candidate = 1
        elif pct < lower_pct:
            raw_candidate = -1
        else:
            raw_candidate = 0

        # Trend quality filter (DIRECTION yardstick)
        if r_squared is not None:
            r2 = r_squared.iloc[i]
            if not np.isnan(r2) and r2 < min_r_squared:
                raw_candidate = 0

        raw_candidates[i] = raw_candidate

        # Hysteresis state machine
        if raw_candidate == candidate_state:
            consecutive_count += 1
            if consecutive_count >= min_persistence:
                current_state = candidate_state
        else:
            candidate_state = raw_candidate
            consecutive_count = 1

        candidate_states[i] = candidate_state
        consecutive_counts[i] = consecutive_count
        labels[i] = current_state

    idx = metric_pct.index
    diagnostics = pd.DataFrame(
        {
            "metric_percentile": metric_pct.values,
            "raw_candidate": raw_candidates,
            "candidate_state": candidate_states,
            "consecutive_count": consecutive_counts,
            "label": labels,
        },
        index=idx,
    )
    return pd.Series(labels, index=idx, name="regime_label"), diagnostics


class CausalRegimeLabeler:
    """
    Fully causal regime labeller across four market yardsticks.
    Describes the current market state (directional trend, momentum, volatility, or path structure) using only
    data available up to and including each bar.

    Design principles:
    - Causality: every computation uses only data[0..i] — no look-ahead.
    - Metrics before labels: continuous rolling metric → adaptive threshold → label.
    - Adaptive thresholds: percentile-based over rolling lookback window.
    - Hysteresis: min_persistence consecutive bars required before state change.
    - One yardstick per instance.

    Parameters
    ----------
    yardstick : MarketPropertyType
        One of: MarketPropertyType.DIRECTION, MOMENTUM, VOLATILITY, PATH_STRUCTURE.
    metric_window : int, default=20
        Bars for primary metric computation (slope window, return window, etc.).
    lookback_window : int, default=100
        Bars for rolling percentile rank (adaptive threshold base).
    upper_pct : float, default=0.70
        Percentile above which raw candidate = +1 (positive state).
    lower_pct : float, default=0.30
        Percentile below which raw candidate = -1 (negative state).
    min_persistence : int, default=3
        Consecutive bars a candidate must hold before becoming current state.
    min_r_squared : float, default=0.30
        For 'direction' only: R² below this forces neutral (noisy trend filter).
    use_atr : bool, default=True
        For 'volatility' only: use ATR when OHLC columns are present.
        Falls back to rolling std of log returns if OHLC not available.
    volume_col : str, default='tick_volume'
        For 'liquidity' only: column name for tick/trade volume.

    Attributes
    ----------
    leaks_future : bool
        Always False.

    Notes
    -----
    **Regime-change detector, not a regime-state classifier.**
    This labeller fires when the metric is extreme relative to recent history (rolling lookback window).
    Once the new regime is fully established in the lookback window, labels normalize toward neutral — the new state has
    become the reference baseline. This is correct and intentional: the labeller describes transitions, not sustained states.
    A supervised classifier trained on these labels learns to generalise beyond transition moments, which is
    what makes it useful for live inference.

    Label conventions:
      direction:     +1=uptrend,  0=neutral,    -1=downtrend
      momentum:      +1=positive, 0=neutral,    -1=negative
      volatility:    +1=high vol, 0=medium vol, -1=low vol  (magnitude, not direction)
      path_structure:+1=trending, 0=transitional,-1=choppy
      liquidity:     +1=liquid,   0=normal,     -1=thin
    """

    leaks_future: bool = False

    def __init__(self, yardstick: MarketPropertyType, metric_window: int = 20,
                 lookback_window: int = 100, upper_pct: float = 0.70, lower_pct: float = 0.30,
                 min_persistence: int = 3, min_r_squared: float = 0.30, use_atr: bool = True,
                 volume_col: str = "tick_volume"):
        if not isinstance(yardstick, MarketPropertyType):
            raise TypeError(
                f"yardstick must be a MarketPropertyType enum, got {type(yardstick).__name__}"
            )

        self.yardstick = yardstick
        if self.yardstick not in _SUPPORTED_YARDSTICKS:
            raise ValueError(
                f"Unsupported yardstick '{yardstick}'. "
                f"Choose from: {[y.value for y in _SUPPORTED_YARDSTICKS]}"
            )

        if metric_window < 2:
            raise ValueError(f"metric_window must be >= 2, got {metric_window}")
        if lookback_window < metric_window:
            raise ValueError(
                f"lookback_window ({lookback_window}) must be >= metric_window ({metric_window})"
            )
        if not 0.0 < lower_pct < upper_pct < 1.0:
            raise ValueError(
                f"Must satisfy 0 < lower_pct < upper_pct < 1, "
                f"got lower_pct={lower_pct}, upper_pct={upper_pct}"
            )
        if min_persistence < 1:
            raise ValueError(f"min_persistence must be >= 1, got {min_persistence}")

        self.metric_window = metric_window
        self.lookback_window = lookback_window
        self.upper_pct = upper_pct
        self.lower_pct = lower_pct
        self.min_persistence = min_persistence
        self.min_r_squared = min_r_squared
        self.use_atr = use_atr
        self.volume_col = volume_col

    @property
    def warmup_bars(self) -> int:
        """
        Minimum bars before the first valid label is produced.

        Use this as embargo_bars in walk-forward loops to avoid label leakage at fold boundaries.
        """
        return max(self.metric_window, self.lookback_window) + self.min_persistence

    def _compute_metric(self, df: pd.DataFrame, price_col: str) -> tuple[pd.Series, Optional[pd.Series]]:
        """
        Compute primary (and optional secondary) rolling metric.

        Returns
        -------
        primary : pd.Series
            Primary metric aligned to df.index.
        secondary : pd.Series or None
            Secondary metric (R² for direction). None for other yardsticks.
        """
        close = df[price_col]

        if self.yardstick == MarketPropertyType.DIRECTION:
            log_close = np.log(close.values.astype(float))
            slopes, r2s = _rolling_slope_r2(log_close, self.metric_window)
            return (
                pd.Series(slopes, index=df.index, name="slope"),
                pd.Series(r2s, index=df.index, name="r_squared"),
            )

        if self.yardstick == MarketPropertyType.MOMENTUM:
            log_close = np.log(close)
            momentum = log_close - log_close.shift(self.metric_window)
            momentum.name = "momentum"
            # Secondary: slope acceleration (2nd derivative — rate of change of momentum)
            acceleration = momentum.diff()
            acceleration.name = "slope_acceleration"
            return momentum, acceleration

        if self.yardstick == MarketPropertyType.VOLATILITY:
            ohlc_available = all(
                c in df.columns for c in ("high", "low", "close")
            )
            if self.use_atr and ohlc_available:
                vol, _ = atr(df["high"].values, df["low"].values, close, period=self.metric_window)
                vol = pd.Series(vol, index=df.index, name="atr")
            else:
                log_rets = np.log(close / close.shift(1))
                vol = log_rets.rolling(self.metric_window, min_periods=self.metric_window).std()
                vol.name = "rolling_std"
            return vol, None

        if self.yardstick == MarketPropertyType.PATH_STRUCTURE:
            # Choppiness Index (CI):
            #   CI → 100 : maximum choppiness (sideways/oscillating)
            #   CI → ~38 : strong trend (directional)
            # Formula: 100 * log10(sum(TR, N) / (highest_high(N) - lowest_low(N))) / log10(N)
            # We negate CI before returning so that percentile rank maps correctly:
            #   high rank of -CI → low CI → trending → label +1
            #   low  rank of -CI → high CI → choppy  → label -1
            ohlc_available = all(c in df.columns for c in ("high", "low"))
            if ohlc_available:
                hi = df["high"].values.astype(float)
                lo = df["low"].values.astype(float)
                cl = close.values.astype(float)
                prev_cl = np.empty_like(cl)
                prev_cl[0] = np.nan
                prev_cl[1:] = cl[:-1]
                tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
                tr_series = pd.Series(tr, index=df.index)
                high_series = df["high"]
                low_series = df["low"]
            else:
                tr_series = close.diff().abs()
                high_series = close
                low_series = close

            tr_sum = tr_series.rolling(self.metric_window).sum()
            hl_range = high_series.rolling(self.metric_window).max() - low_series.rolling(self.metric_window).min()
            raw_ci = 100.0 * np.log10(
                tr_sum / hl_range.where(hl_range > 0)
            ) / np.log10(self.metric_window)
            ci = pd.Series(raw_ci, index=df.index, name="choppiness_index")

            # Secondary: lag-1 return autocorrelation (+1 = momentum, -1 = mean-reversion)
            log_rets = np.log(close / close.shift(1))
            ac = auto_corr(log_rets, window=self.metric_window, lag=1)
            ac.name = "return_autocorr"

            return -ci, ac

        if self.yardstick == MarketPropertyType.LIQUIDITY:
            # Composite of three complementary liquidity measures:
            #   1. Corwin-Schultz estimated spread (from H/L — wider = less liquid)
            #   2. Amihud illiquidity (|return|/(volume×close) — higher = less liquid)
            #   3. Tick volume hour-of-week z-score (higher = more liquid than typical)
            #
            # Each component is smoothed, sign-corrected (higher = more liquid),
            # then averaged into a single continuous composite. The outer rolling
            # percentile rank (shared with all other yardsticks) converts the
            # composite into the [0,1] space that the hysteresis state machine expects.
            if self.volume_col not in df.columns:
                raise KeyError(
                    f"Volume column '{self.volume_col}' not found for LIQUIDITY yardstick. "
                    f"Available: {df.columns.tolist()}"
                )
            volume = df[self.volume_col].astype(float)
            components = []

            # 1. Corwin-Schultz spread — negate so higher = more liquid
            ohlc = all(c in df.columns for c in ("high", "low"))
            if ohlc:
                cs = corwin_schultz_spread(df["high"], df["low"], close, window=2)
                cs_smooth = -cs.rolling(self.metric_window, min_periods=self.metric_window).mean()
                components.append(cs_smooth)

            # 2. Amihud illiquidity — negate so higher = more liquid
            amihud = amihud_illiquidity(close, volume, window=self.metric_window, log_transform=True)
            components.append(-amihud)

            # 3. Tick volume hour-of-week z-score — already positive = more liquid
            if isinstance(df.index, pd.DatetimeIndex):
                how_z = tick_volume_how_zscore(volume, min_periods=self.metric_window)
                components.append(how_z)

            # z-score each component before averaging (different scales)
            normed = []
            for c in components:
                mu = c.expanding(min_periods=self.metric_window).mean()
                sigma = c.expanding(min_periods=self.metric_window).std()
                normed.append((c - mu) / (sigma + 1e-10))

            composite = pd.concat(normed, axis=1).mean(axis=1)
            composite.name = "liquidity_composite"
            return composite, None

        raise RuntimeError(f"Unhandled yardstick: {self.yardstick}")  # pragma: no cover

    def _rolling_percentile_rank(self, series: pd.Series) -> pd.Series:
        """Rolling percentile rank [0, 1] over lookback_window."""
        return series.rolling(self.lookback_window, min_periods=self.lookback_window).rank(pct=True)

    def label(self, df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        """
        Generate causal regime labels.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. Must contain price_col. For volatility with
            use_atr=True, must also contain 'high' and 'low'.
        price_col : str, default='close'
            Column name for close prices.

        Returns
        -------
        pd.Series
            Integer labels {-1, 0, 1} aligned to df.index.
            First warmup_bars entries are NaN.
        """
        labels, _ = self._label_internal(df, price_col)
        return labels

    def label_with_diagnostics(self, df: pd.DataFrame, price_col: str = "close") -> tuple[pd.Series, pd.DataFrame]:
        """
        Generate causal regime labels with full per-bar diagnostic trace.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        price_col : str, default='close'
            Column name for close prices.

        Returns
        -------
        labels : pd.Series
            Integer labels {-1, 0, 1} aligned to df.index.
        diagnostics : pd.DataFrame
            Columns: metric_value, [secondary_value], metric_percentile,
            raw_candidate, candidate_state, consecutive_count, label.
        """
        labels, diagnostics = self._label_internal(df, price_col)

        # Attach raw metric values to diagnostics for full transparency
        primary, secondary = self._compute_metric(df, price_col)
        diagnostics.insert(0, "metric_value", primary.values)
        if secondary is not None:
            diagnostics.insert(1, "secondary_value", secondary.values)

        return labels, diagnostics

    def _label_internal(self, df: pd.DataFrame, price_col: str) -> tuple[pd.Series, pd.DataFrame]:
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found. Available: {df.columns.tolist()}")
        if len(df) < self.warmup_bars:
            raise ValueError(
                f"DataFrame has {len(df)} rows but warmup_bars={self.warmup_bars}. "
                f"Provide at least warmup_bars rows."
            )

        primary, secondary = self._compute_metric(df, price_col)
        metric_pct = self._rolling_percentile_rank(primary)

        r2_for_hysteresis = secondary if self.yardstick == MarketPropertyType.DIRECTION else None
        labels, diagnostics = _apply_hysteresis(
            metric_pct=metric_pct,
            upper_pct=self.upper_pct,
            lower_pct=self.lower_pct,
            min_persistence=self.min_persistence,
            r_squared=r2_for_hysteresis,
            min_r_squared=self.min_r_squared,
        )
        return labels, diagnostics

    def __repr__(self) -> str:
        return (
            f"CausalRegimeLabeler("
            f"yardstick={self.yardstick.value!r}, "
            f"metric_window={self.metric_window}, "
            f"lookback_window={self.lookback_window}, "
            f"upper_pct={self.upper_pct}, "
            f"lower_pct={self.lower_pct}, "
            f"min_persistence={self.min_persistence})"
        )

