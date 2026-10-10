from enum import StrEnum
from typing import Optional, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class RenkoAnchor(StrEnum):
    """Where the reference level for the next brick sits once a brick has formed."""

    GRID = "grid"  # classic Renko: bricks sit on the fixed levels k * brick_size
    CLOSE = "close"  # re-anchor at the close that formed the brick (close-only, variable brick height)


class Renko:
    """
    A production-ready Renko chart builder that converts OHLCV data into Renko bricks.

    Renko charts filter out minor price movements and only display price changes
    that exceed a specified brick size, making trend identification clearer.

    Brick construction and its bias
    -------------------------------
    A brick forms one ``brick_size`` above or below the current reference level, so on a driftless random walk the
    next brick's direction should be a fair coin (reversal share 0.5). How close each mode gets to that:

    * ``anchor=GRID, use_high_low=True`` (default) walks every bar along the path its OHLC implies:
      open -> low -> high -> close when close >= open, open -> high -> low -> close otherwise (the previous close
      stands in for the open when there is no open column). The last brick of a bar therefore agrees with where the
      bar closed. A fixed grid on sampled prices keeps a small momentum bias, because the price that crosses a level
      overshoots it and the next brick in the same direction is nearer than a reversal: the reversal share is about
      0.48 at ``brick_size`` = 2 bar sigmas, the same as a grid built tick by tick.
    * ``anchor=GRID, use_high_low=False`` sees closes only, so the overshoot is a whole bar's worth: reversal share
      about 0.39 at 2 bar sigmas and 0.32 at 1. Use it for charts, not for statistics on brick sequences.
    * ``anchor=CLOSE`` (requires ``use_high_low=False``) re-anchors at the close that formed each brick. One brick
      per bar at most, with height >= ``brick_size``. Under symmetric iid increments the next brick's direction is
      independent of every earlier brick, so reversal fractions and run lengths have an exact null. Use this mode for
      path-structure statistics.

    Parameters
    ----------
    brick_size : float
        The size of each brick in price points. Must be positive.
    high_col : str, default='high'
        Column name for high prices
    low_col : str, default='low'
        Column name for low prices
    close_col : str, default='close'
        Column name for closing prices (used for initial price)
    volume_col : str, default='volume'
        Column name for volume data
    use_high_low : bool, default=True
        If True, walks each bar's open/high/low/close path for brick detection.
        If False, only uses close_col.
    open_col : str, default='open'
        Column name for open prices. Optional: when it is absent the previous close is used as the bar's open.
    anchor : RenkoAnchor, default=RenkoAnchor.GRID
        GRID for classic fixed-level bricks, CLOSE for close-anchored bricks (see above).

    Attributes
    ----------
    brick_size : float
        The configured brick size
    renko_df : pd.DataFrame
        The resulting Renko dataframe after calling build()

    Examples
    --------
    >>> ohlcv_df = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'high': [102, 103, 104],
    ...     'low': [99, 100, 101],
    ...     'close': [101, 102, 103],
    ...     'volume': [1000, 1100, 1200]
    ... })
    >>> renko = Renko(brick_size=2.0)
    >>> renko_df = renko.build(ohlcv_df)

    >>> # Custom column names
    >>> custom_df = pd.DataFrame({
    ...     'h': [102, 103, 104],
    ...     'l': [99, 100, 101],
    ...     'c': [101, 102, 103],
    ...     'v': [1000, 1100, 1200]
    ... })
    >>> renko = Renko(brick_size=2.0, high_col='h', low_col='l',
    ...               close_col='c', volume_col='v')
    >>> renko_df = renko.build(custom_df)
    """

    # Class constants for result dataframe column names
    BRICK_NUM_COL = "brick_num"
    OPEN_COL = "open"
    CLOSE_COL = "close"
    DIRECTION_COL = "direction"
    TIMESTAMP_COL = "timestamp"
    VOLUME_COL = "volume"

    UP_DIRECTION = 1
    DOWN_DIRECTION = -1

    def __init__(self, brick_size: float, high_col: str = "high", low_col: str = "low", close_col: str = "close",
                 volume_col: str = "volume", use_high_low: bool = True, open_col: str = "open",
                 anchor: RenkoAnchor = RenkoAnchor.GRID):
        """
        Initialize the Renko chart builder.

        Parameters
        ----------
        brick_size : float
            The size of each brick in price points. Must be positive.
        high_col : str, default='high'
            Column name for high prices
        low_col : str, default='low'
            Column name for low prices
        close_col : str, default='close'
            Column name for closing prices
        volume_col : str, default='volume'
            Column name for volume data
        use_high_low : bool, default=True
            If True, uses the bar's open/high/low/close path for brick detection
        open_col : str, default='open'
            Column name for open prices (optional in the data)
        anchor : RenkoAnchor, default=RenkoAnchor.GRID
            Reference level after a brick: the fixed grid, or the close that formed it

        Raises
        ------
        ValueError
            If brick_size is not positive, column names are empty, or anchor=CLOSE is combined with use_high_low
        TypeError
            If brick_size cannot be converted to float
        """
        try:
            self.brick_size = float(brick_size)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"brick_size must be numeric, got {type(brick_size).__name__}"
            ) from e

        if self.brick_size <= 0:
            raise ValueError(f"brick_size must be positive, got {self.brick_size}")

        # Validate column names
        for col_name, col_value in [
            ("high_col", high_col),
            ("low_col", low_col),
            ("close_col", close_col),
            ("volume_col", volume_col),
            ("open_col", open_col),
        ]:
            if not isinstance(col_value, str) or not col_value.strip():
                raise ValueError(f"{col_name} must be a non-empty string")

        self.anchor = RenkoAnchor(anchor)
        if self.anchor is RenkoAnchor.CLOSE and use_high_low:
            raise ValueError("anchor=RenkoAnchor.CLOSE re-anchors at closes only; pass use_high_low=False")

        # Store configuration
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.open_col = open_col
        self.use_high_low = use_high_low

        # Result storage
        self.renko_df: Optional[pd.DataFrame] = None
        self._source_df: Optional[pd.DataFrame] = None

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Renko chart from OHLCV dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with OHLCV data. Must contain configured columns.

        Returns
        -------
        pd.DataFrame
            Renko dataframe with columns:
            - brick_num: Sequential brick number
            - open: Brick open price
            - close: Brick close price
            - direction: 1 for up brick, -1 for down brick
            - timestamp: Timestamp when brick was formed
            - volume: Accumulated volume during brick formation

        Raises
        ------
        ValueError
            If dataframe is empty or missing required columns
        TypeError
            If df is not a pandas DataFrame
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError("Input dataframe is empty")

        # Create a copy to avoid modifying original
        df = df.copy()
        self._source_df = df

        # Validate required columns exist
        self._validate_columns(df)

        # Check for volume column availability
        has_volume = self.volume_col in df.columns

        # Initialize brick building
        bricks = self._initialize_bricks(df, has_volume)

        # Create result dataframe
        self.renko_df = self._create_result_dataframe(bricks, has_volume)

        return self.renko_df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in dataframe.

        Raises
        ------
        ValueError
            If required columns are missing
        """
        if self.use_high_low:
            required_cols = [self.high_col, self.low_col, self.close_col]
        else:
            required_cols = [self.close_col]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            mode = "high/low" if self.use_high_low else "close-only"
            raise ValueError(
                f"Missing required columns for {mode} mode: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _initialize_bricks(self, df: pd.DataFrame, has_volume: bool) -> List[dict]:
        """
        Process dataframe and build list of bricks.

        Parameters
        ----------
        df : pd.DataFrame
            Validated input dataframe
        has_volume : bool
            Whether volume column is available

        Returns
        -------
        List[dict]
            List of brick dictionaries
        """
        close = df[self.close_col].to_numpy(dtype=float)
        volume = np.nan_to_num(df[self.volume_col].to_numpy(dtype=float)) if has_volume else np.zeros(len(df))
        finite = np.flatnonzero(np.isfinite(close))
        if finite.size == 0:
            return []
        # The first finite close is the reference: price action before it (that bar's own high/low) is history.
        start = int(finite[0])

        if self.anchor is RenkoAnchor.CLOSE:
            bricks = self._close_anchored_bricks(df.index, close, volume, start)
        else:
            bricks = self._grid_bricks(df, close, volume, start)

        for brick_num, brick in enumerate(bricks):
            brick[self.BRICK_NUM_COL] = brick_num
        return bricks

    def _bar_paths(self, df: pd.DataFrame, close: np.ndarray) -> np.ndarray:
        """Per-bar price path, shape (T, 4) in high/low mode and (T, 1) in close-only mode.

        High/low mode orders the extremes by the bar's direction: open -> low -> high -> close for an up bar,
        open -> high -> low -> close for a down bar. Taking the high first on every bar (the old behaviour) leaves the
        last brick pointing away from the close and manufactures reversals.
        """
        if not self.use_high_low:
            return close[:, None]
        high = df[self.high_col].to_numpy(dtype=float)
        low = df[self.low_col].to_numpy(dtype=float)
        if self.open_col in df.columns:
            opens = df[self.open_col].to_numpy(dtype=float)
        else:
            opens = np.r_[close[0], close[:-1]]
        up_bar = close >= opens
        first = np.where(up_bar, low, high)
        second = np.where(up_bar, high, low)
        return np.column_stack([opens, first, second, close])

    def _grid_bricks(self, df: pd.DataFrame, close: np.ndarray, volume: np.ndarray, start: int) -> List[dict]:
        """Classic bricks on the fixed levels k * brick_size, crossed in path order within each bar."""
        paths = self._bar_paths(df, close)
        index = df.index
        b = self.brick_size
        level = int(round(close[start] / b))  # integer level index, so repeated steps never drift
        bricks: List[dict] = []
        accumulated_volume = volume[start]
        for i in range(start + 1, len(close)):
            accumulated_volume += volume[i]
            formed = 0
            for price in paths[i]:  # NaN compares False, so missing prices form no bricks
                while price >= (level + 1) * b:
                    bricks.append(self._brick(level * b, (level + 1) * b, self.UP_DIRECTION, index[i],
                                              accumulated_volume if formed == 0 else 0.0))
                    level += 1
                    formed += 1
                while price <= (level - 1) * b:
                    bricks.append(self._brick(level * b, (level - 1) * b, self.DOWN_DIRECTION, index[i],
                                              accumulated_volume if formed == 0 else 0.0))
                    level -= 1
                    formed += 1
            if formed:
                accumulated_volume = 0.0
        return bricks

    def _close_anchored_bricks(self, index: pd.Index, close: np.ndarray, volume: np.ndarray, start: int) -> List[dict]:
        """One brick per bar whose close is >= brick_size from the last anchor; that close becomes the anchor."""
        anchor = close[start]
        bricks: List[dict] = []
        accumulated_volume = volume[start]
        for i in range(start + 1, len(close)):
            accumulated_volume += volume[i]
            move = close[i] - anchor
            if abs(move) >= self.brick_size:  # False for NaN
                direction = self.UP_DIRECTION if move > 0 else self.DOWN_DIRECTION
                bricks.append(self._brick(anchor, close[i], direction, index[i], accumulated_volume))
                anchor = close[i]
                accumulated_volume = 0.0
        return bricks

    def _brick(self, open_price: float, close_price: float, direction: int, timestamp, volume: float) -> dict:
        return {self.OPEN_COL: open_price, self.CLOSE_COL: close_price, self.DIRECTION_COL: direction,
                self.TIMESTAMP_COL: timestamp, self.VOLUME_COL: volume}

    def _create_result_dataframe(
        self, bricks: List[dict], has_volume: bool
    ) -> pd.DataFrame:
        if not bricks:
            # Return empty dataframe with correct schema
            return pd.DataFrame(
                columns=[
                    self.BRICK_NUM_COL,
                    self.OPEN_COL,
                    self.CLOSE_COL,
                    self.DIRECTION_COL,
                    self.TIMESTAMP_COL,
                    self.VOLUME_COL,
                ]
            )

        result_df = pd.DataFrame(bricks)

        # Ensure volume column exists
        if not has_volume:
            result_df[self.VOLUME_COL] = 0.0

        # Ensure proper column order
        column_order = [
            self.BRICK_NUM_COL,
            self.OPEN_COL,
            self.CLOSE_COL,
            self.DIRECTION_COL,
            self.TIMESTAMP_COL,
            self.VOLUME_COL,
        ]
        return result_df[column_order]

    def get_trend_changes(self) -> pd.DataFrame:
        """
        Identify trend change points in the Renko chart.

        Returns
        -------
        pd.DataFrame
            Subset of renko_df where trend direction changed

        Raises
        ------
        RuntimeError
            If build() hasn't been called yet
        """
        self._check_build_called()

        if self.renko_df.empty:
            return self.renko_df.copy()

        # Find where direction changes
        direction_changes = self.renko_df[self.DIRECTION_COL].diff() != 0
        return self.renko_df[direction_changes].copy()

    def get_statistics(self) -> dict:
        self._check_build_called()

        if self.renko_df.empty:
            return {
                "total_bricks": 0,
                "up_bricks": 0,
                "down_bricks": 0,
                "trend_changes": 0,
                "largest_trend": 0,
                "total_volume": 0.0,
            }

        df = self.renko_df
        up_bricks = (df[self.DIRECTION_COL] == self.UP_DIRECTION).sum()
        down_bricks = (df[self.DIRECTION_COL] == self.DOWN_DIRECTION).sum()

        # Calculate trend changes (subtract 1 because first row always shows change)
        trend_changes = max(0, (df[self.DIRECTION_COL].diff() != 0).sum() - 1)

        # Calculate largest trend
        runs = (df[self.DIRECTION_COL] != df[self.DIRECTION_COL].shift()).cumsum()
        largest_trend = df.groupby(runs).size().max()

        # Total volume
        total_volume = df[self.VOLUME_COL].sum()

        return {
            "total_bricks": len(df),
            "up_bricks": int(up_bricks),
            "down_bricks": int(down_bricks),
            "trend_changes": int(trend_changes),
            "largest_trend": int(largest_trend),
            "total_volume": float(total_volume),
        }

    def _check_build_called(self) -> None:
        if self.renko_df is None:
            raise RuntimeError(
                "Must call build() before accessing results. "
                "Example: renko.build(df)"
            )

    def plot(
        self,
        figsize: Tuple[int, int] = (14, 8),
        up_color: str = "#26a69a",
        down_color: str = "#ef5350",
        edge_color: str = "#1f1f1f",
        title: Optional[str] = None,
        show_volume: bool = True,
        style: str = "classic",
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """
        Plot the Renko chart with bricks.

        Parameters
        ----------
        figsize : Tuple[int, int], default=(14, 8)
            Figure size (width, height) in inches
        up_color : str, default='#26a69a'
            Color for upward bricks (green)
        down_color : str, default='#ef5350'
            Color for downward bricks (red)
        edge_color : str, default='#1f1f1f'
            Color for brick edges
        title : str, optional
            Chart title. If None, auto-generates title
        show_volume : bool, default=True
            If True, shows volume as bar chart below main chart
        style : str, default='classic'
            Matplotlib style ('classic', 'seaborn', 'dark_background', etc.)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If provided, figsize is ignored.

        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes objects for further customization

        Raises
        ------
        RuntimeError
            If build() hasn't been called yet

        Examples
        --------
        >>> renko = Renko(brick_size=2.0)
        >>> renko.build(df)
        >>> fig, ax = renko.plot()
        >>> plt.show()

        >>> # Customize appearance
        >>> fig, ax = renko.plot(
        ...     up_color='blue',
        ...     down_color='orange',
        ...     title='My Custom Renko Chart',
        ...     show_volume=False
        ... )
        """
        self._check_build_called()

        if self.renko_df.empty:
            raise ValueError("Cannot plot empty Renko chart (no bricks formed)")

        # Apply style
        plt.style.use(style)

        # Create figure and axes
        if ax is None:
            if show_volume and self.VOLUME_COL in self.renko_df.columns:
                fig, (ax_main, ax_vol) = plt.subplots(
                    2,
                    1,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                    sharex=True,
                )
            else:
                fig, ax_main = plt.subplots(1, 1, figsize=figsize)
                ax_vol = None
        else:
            ax_main = ax
            fig = ax.get_figure()
            ax_vol = None

        # Plot bricks
        self._plot_bricks(ax_main, up_color, down_color, edge_color)

        # Plot volume if requested
        if show_volume and ax_vol is not None:
            self._plot_volume(ax_vol, up_color, down_color)

        # Set title
        if title is None:
            stats = self.get_statistics()
            title = (
                f"Renko Chart (Brick Size: {self.brick_size} points) - "
                f'{stats["total_bricks"]} bricks, '
                f'{stats["trend_changes"]} trend changes'
            )
        ax_main.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Formatting
        ax_main.set_ylabel("Price", fontsize=12, fontweight="bold")
        ax_main.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax_main.tick_params(labelsize=10)

        if ax_vol is not None:
            ax_vol.set_xlabel("Time", fontsize=12, fontweight="bold")
            ax_vol.set_ylabel("Volume", fontsize=10, fontweight="bold")
            ax_vol.tick_params(labelsize=9)
        else:
            ax_main.set_xlabel("Time", fontsize=12, fontweight="bold")

        # Rotate x-axis labels for better readability
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Adjust layout
        fig.tight_layout()

        return fig, ax_main

    def _plot_bricks(
        self, ax: Axes, up_color: str, down_color: str, edge_color: str
    ) -> None:
        """
        Plot Renko bricks on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        up_color : str
            Color for up bricks
        down_color : str
            Color for down bricks
        edge_color : str
            Color for brick edges
        """
        df = self.renko_df

        # Convert timestamps to numerical positions for plotting
        timestamps = df[self.TIMESTAMP_COL].values
        x_positions = np.arange(len(df))

        # Create brick width (80% of space between bricks)
        brick_width = 0.8

        # Plot each brick
        for idx, row in df.iterrows():
            x_pos = x_positions[idx]
            open_price = row[self.OPEN_COL]
            close_price = row[self.CLOSE_COL]
            direction = row[self.DIRECTION_COL]

            # Determine color based on direction
            color = up_color if direction == self.UP_DIRECTION else down_color

            # Create rectangle for brick
            # Rectangle params: (x, y, width, height)
            brick_height = abs(close_price - open_price)
            brick_bottom = min(open_price, close_price)

            rect = patches.Rectangle(
                (x_pos - brick_width / 2, brick_bottom),
                brick_width,
                brick_height,
                linewidth=1.5,
                edgecolor=edge_color,
                facecolor=color,
                alpha=0.9,
            )
            ax.add_patch(rect)

        # Set x-axis with timestamps
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.set_xticks(x_positions[:: max(1, len(df) // 20)])  # Show ~20 labels max
        ax.set_xticklabels(
            [str(ts) for ts in timestamps[:: max(1, len(df) // 20)]],
            rotation=45,
            ha="right",
        )

        # Set y-axis limits with some padding
        price_min = df[self.OPEN_COL].min()
        price_max = df[self.CLOSE_COL].max()
        price_range = price_max - price_min
        ax.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)

    def _plot_volume(self, ax: Axes, up_color: str, down_color: str) -> None:
        """
        Plot volume bars on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        up_color : str
            Color for up brick volumes
        down_color : str
            Color for down brick volumes
        """
        df = self.renko_df
        x_positions = np.arange(len(df))

        # Color bars based on brick direction
        colors = [
            up_color if d == self.UP_DIRECTION else down_color
            for d in df[self.DIRECTION_COL]
        ]

        # Plot volume bars
        ax.bar(x_positions, df[self.VOLUME_COL], color=colors, alpha=0.6, width=0.8)

        # Format volume axis
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Format y-axis labels for volume (e.g., 1000 -> 1K)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: self._format_volume(x))
        )

    @staticmethod
    def _format_volume(volume: float) -> str:
        """
        Format volume numbers for display (e.g., 1000 -> 1K).

        Parameters
        ----------
        volume : float
            Volume value

        Returns
        -------
        str
            Formatted volume string
        """
        if volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return f"{volume:.0f}"

    def __repr__(self) -> str:
        """String representation of Renko instance."""
        built_status = "built" if self.renko_df is not None else "not built"
        brick_count = len(self.renko_df) if self.renko_df is not None else 0
        mode = "high/low" if self.use_high_low else "close-only"
        return (
            f"Renko(brick_size={self.brick_size}, mode={mode}, anchor={self.anchor.value}, "
            f"status={built_status}, bricks={brick_count})"
        )
