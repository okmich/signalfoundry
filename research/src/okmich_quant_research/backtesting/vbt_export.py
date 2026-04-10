"""VectorBT Portfolio -> QUANTFRAME export mixin.

Converts a VectorBT Portfolio into the folder/file structure that QUANTFRAME's ingestion layer expects:

    {output_dir}/{symbol}_{strategy_slug}/
        metadata.json
        equity_curve.parquet   (or .csv)
        returns.parquet
        trades.parquet
"""

import json
import re
import shutil
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import vectorbt as vbt

# Characters safe in a folder name component (no path separators, no ..)
_SAFE_FOLDER_CHAR = re.compile(r"[^\w]")


class FileFormat(StrEnum):
    PARQUET = "parquet"
    CSV = "csv"



_REQUIRED_VBT_TRADE_COLS = frozenset({
    "Entry Timestamp", "Exit Timestamp", "Avg Entry Price", "Avg Exit Price",
    "Size", "PnL", "Return", "Direction",
})


class QuantframeExportMixin:
    """Mixin that adds a ``export_to_quantframe`` method to any class holding a ``vbt.Portfolio``."""

    def export_to_quantframe(self, portfolio: vbt.Portfolio, output_dir: str | Path, symbol: str, strategy: str,
                             timeframe: str, param_set: dict | None = None,
                             file_format: str | FileFormat = FileFormat.PARQUET) -> Path:
        """Export *portfolio* to a QUANTFRAME-compatible folder structure.

        Parameters
        ----------
        portfolio : vbt.Portfolio
            Single-symbol, single-strategy portfolio. Must not be multi-column or grouped.
        output_dir : str | Path
            Parent directory (must exist). A sub-folder ``{symbol}_{strategy_slug}`` is created inside.
        symbol : str
            Trading symbol, e.g. ``"AAPL"`` or ``"DJ30"``.
        strategy : str
            Human-readable strategy name.
        timeframe : str
            Bar timeframe, e.g. ``"1D"``, ``"4H"``.
        param_set : dict | None
            Optional strategy parameters written into ``metadata.json``.
        file_format : str | FileFormat
            ``"parquet"`` (default) or ``"csv"``.

        Returns
        -------
        Path
            Absolute path to the created backtest sub-folder.
        """
        self._validate_export_inputs(portfolio, output_dir, symbol, strategy, file_format, timeframe)

        output_dir = Path(output_dir)
        file_format = FileFormat(file_format)

        # -- resolve output folder (sanitised) --------------------------------
        strategy_slug = self._slugify(strategy)
        safe_symbol = self._sanitize_path_component(symbol)
        folder_name = f"{safe_symbol}_{strategy_slug}"
        backtest_dir = output_dir / folder_name

        if not backtest_dir.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Constructed output path escapes output_dir: {backtest_dir}")

        # -- extract, validate -------------------------------------------------
        equity_df = self._extract_equity_curve(portfolio)
        returns_df = self._extract_returns(portfolio)
        trades_df = self._extract_trades(portfolio)

        # -- write atomically via staging dir ---------------------------------
        staged_dir = _make_staging_dir(output_dir, folder_name)
        try:
            self._write_metadata(staged_dir, symbol=symbol, strategy=strategy, strategy_id=folder_name,
                                 timeframe=timeframe, param_set=param_set or {})
            self._write_dataframe(equity_df, staged_dir / f"equity_curve.{file_format.value}", file_format)
            self._write_dataframe(returns_df, staged_dir / f"returns.{file_format.value}", file_format)
            self._write_dataframe(trades_df, staged_dir / f"trades.{file_format.value}", file_format)
            _promote_staged_export(staged_dir, backtest_dir)
        except Exception:
            _rmtree_if_exists(staged_dir)
            raise

        return backtest_dir.resolve()

    # -- validation -----------------------------------------------------------

    @staticmethod
    def _validate_export_inputs(portfolio: vbt.Portfolio, output_dir: str | Path, symbol: str, strategy: str,
                                file_format: str | FileFormat, timeframe: str | None = None) -> None:
        if not isinstance(portfolio, vbt.Portfolio):
            raise TypeError(f"portfolio must be a vbt.Portfolio instance, got {type(portfolio)}")

        if timeframe is not None and not isinstance(timeframe, str):
            raise TypeError(f"timeframe must be a string, got {type(timeframe)}")
        if timeframe is not None and not timeframe.strip():
            raise ValueError("timeframe must be a non-empty string")

        file_format = file_format if isinstance(file_format, str) else file_format.value
        if file_format not in (FileFormat.PARQUET, FileFormat.CSV):
            raise ValueError(f"file_format must be 'parquet' or 'csv', got '{file_format}'")

        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        if not strategy or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")

        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise ValueError(f"output_dir does not exist: {output_dir}")
        if not output_dir.is_dir():
            raise ValueError(f"output_dir is not a directory: {output_dir}")

        try:
            if portfolio.wrapper.ndim > 1:
                raise ValueError("portfolio appears to be multi-column. export_to_quantframe only supports "
                                 "single-symbol, single-strategy Portfolio instances.")
        except ValueError:
            raise
        except AttributeError:
            raise ValueError("Cannot verify portfolio dimensionality - portfolio.wrapper.ndim is not accessible. "
                             "Ensure this is a standard single-column vbt.Portfolio.")

    # -- slug helper ----------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\s\-]+", "_", text)
        text = _SAFE_FOLDER_CHAR.sub("", text)
        return text.strip("_")

    @staticmethod
    def _sanitize_path_component(text: str) -> str:
        """Strip path separators and dangerous sequences from a single folder-name component."""
        text = text.replace("/", "_").replace("\\", "_").replace("..", "_")
        text = _SAFE_FOLDER_CHAR.sub("", text)
        return text.strip("_") or "unknown"

    # -- data extraction ------------------------------------------------------

    @staticmethod
    def _extract_equity_curve(portfolio: vbt.Portfolio) -> pd.DataFrame:
        try:
            value_series = portfolio.value()
        except Exception as e:
            raise RuntimeError(f"Failed to extract equity curve via portfolio.value(): {e}") from e

        if value_series is None or value_series.empty:
            raise RuntimeError("portfolio.value() returned empty data. Cannot generate equity_curve file.")
        if value_series.isna().all():
            raise RuntimeError("portfolio.value() returned all-NaN series. Check that the portfolio was run with valid price data.")
        if len(value_series) < 2:
            raise RuntimeError(f"portfolio.value() returned only {len(value_series)} row(s). At least 2 rows are required.")

        idx = _validate_datetime_index(value_series.index)
        equity = value_series.values.astype("float64")
        _guard_numeric(equity, "equity_curve.equity")

        return pd.DataFrame({"date": _timestamps_to_strings(idx), "equity": equity})

    @staticmethod
    def _extract_returns(portfolio: vbt.Portfolio) -> pd.DataFrame:
        try:
            returns_series = portfolio.returns()
        except Exception as e:
            raise RuntimeError(f"Failed to extract returns via portfolio.returns(): {e}") from e

        if returns_series is None or returns_series.empty:
            raise RuntimeError("portfolio.returns() returned empty data. Cannot generate returns file.")

        idx = _validate_datetime_index(returns_series.index)
        values = returns_series.values.astype("float64")
        _guard_numeric(values, "returns.return", allow_nan=True)  # NaN dropped below via dropna

        df = pd.DataFrame({"date": _timestamps_to_strings(idx), "return": values})
        df = df.dropna(subset=["return"])

        if len(df) < 2:
            raise RuntimeError(f"After dropping NaN values, only {len(df)} return row(s) remain. At least 2 valid rows are required.")

        return df.reset_index(drop=True)

    @staticmethod
    def _extract_trades(portfolio: vbt.Portfolio) -> pd.DataFrame:
        try:
            raw = portfolio.trades.records_readable
        except Exception as e:
            raise RuntimeError(f"Failed to extract trades via portfolio.trades.records_readable: {e}") from e

        if raw is None or len(raw) == 0:
            raise RuntimeError("Portfolio has no closed trades. QUANTFRAME requires at least one closed trade.")

        missing = _REQUIRED_VBT_TRADE_COLS - set(raw.columns)
        if missing:
            raise RuntimeError(f"portfolio.trades.records_readable is missing required columns: {sorted(missing)}. "
                               f"Available columns: {sorted(raw.columns.tolist())}")

        entry_ts = _ensure_datetime_series(raw["Entry Timestamp"])
        exit_ts = _ensure_datetime_series(raw["Exit Timestamp"])
        entry_price = raw["Avg Entry Price"].astype("float64")
        exit_price = raw["Avg Exit Price"].astype("float64")
        size = raw["Size"].astype("float64")
        pnl = raw["PnL"].astype("float64")
        pnl_pct = raw["Return"].astype("float64")

        for col_name, arr in [("entry_price", entry_price), ("exit_price", exit_price), ("size", size), ("pnl", pnl), ("pnl_pct", pnl_pct)]:
            _guard_numeric(arr.values, f"trades.{col_name}")

        df = pd.DataFrame({
            "entry_date": _timestamps_to_strings(entry_ts),
            "exit_date": _timestamps_to_strings(exit_ts),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "direction": raw["Direction"].str.lower().str.strip(),
        })
        df = df.dropna(subset=["entry_date", "exit_date"])

        if len(df) == 0:
            raise RuntimeError("All trades had null entry or exit dates after parsing. Cannot generate trades file.")

        return df.reset_index(drop=True)

    # -- file writers ---------------------------------------------------------

    @staticmethod
    def _write_metadata(backtest_dir: Path, *, symbol: str, strategy: str, strategy_id: str, timeframe: str,
                        param_set: dict) -> None:
        metadata = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "strategy": strategy,
            "timeframe": timeframe,
            "param_set": _sanitize_for_json(param_set),
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "source": "vectorbt",
            "vbt_version": vbt.__version__,
        }
        with open(backtest_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)

    @staticmethod
    def _write_dataframe(df: pd.DataFrame, path: Path, file_format: FileFormat) -> None:
        if file_format is FileFormat.PARQUET:
            df.to_parquet(path, index=False, engine="pyarrow")
        else:
            df.to_csv(path, index=False)


# -- module-level utilities ---------------------------------------------------

def _validate_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Validate datetime index and preserve timestamp timezone/offset as-is."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise RuntimeError(f"Expected DatetimeIndex, got {type(idx).__name__}. "
                           "Portfolio index must be datetime-based for export.")
    return idx


def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Ensure datetime-like series, parsing only when needed, with no timezone normalization."""
    if not isinstance(s, pd.Series):
        raise RuntimeError(f"Expected pandas Series, got {type(s).__name__}.")
    if pd.api.types.is_datetime64_any_dtype(s.dtype) or isinstance(s.dtype, pd.DatetimeTZDtype):
        _guard_not_mixed_naive_aware(s)
        return s
    # String/object series — detect mixed tz before pd.to_datetime silently drops offsets
    _guard_not_mixed_naive_aware_strings(s)
    dt_series = pd.to_datetime(s, errors="coerce", utc=False)
    if not (pd.api.types.is_datetime64_any_dtype(dt_series.dtype) or isinstance(dt_series.dtype, pd.DatetimeTZDtype)):
        raise RuntimeError(f"Expected datetime-like Series, got {dt_series.dtype}.")
    return dt_series


def _guard_numeric(arr, label: str, *, allow_nan: bool = False) -> None:
    """Raise RuntimeError if array contains inf/-inf or (optionally) NaN values."""
    if np.isinf(arr).any():
        raise RuntimeError(f"{label} contains inf/-inf values. Clean the portfolio data before exporting.")
    if not allow_nan and np.isnan(arr).any():
        raise RuntimeError(f"{label} contains NaN values. Clean the portfolio data before exporting.")


def _timestamps_to_strings(values) -> list[str | float]:
    """Convert datetime values to ISO8601 strings while preserving original timezone/offset."""
    return [_timestamp_to_string(v) for v in values]


def _timestamp_to_string(value) -> str | float:
    """Convert one datetime value to ISO8601; preserve timezone/offset and keep NaT as NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return pd.Timestamp(value).isoformat()


def _guard_not_mixed_naive_aware(s: pd.Series) -> None:
    """Reject series that mix tz-naive and tz-aware timestamps."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return

    def _is_tz_aware(v) -> bool:
        ts = v if isinstance(v, pd.Timestamp) else pd.Timestamp(v)
        return ts.tzinfo is not None

    first_is_aware = _is_tz_aware(non_null.iloc[0])
    for v in non_null.iloc[1:]:
        if _is_tz_aware(v) != first_is_aware:
            raise RuntimeError("Timestamp series mixes tz-aware and tz-naive values. "
                               "Provide consistent timezone semantics in input data.")


def _guard_not_mixed_naive_aware_strings(s: pd.Series) -> None:
    """Detect mixed tz-naive/tz-aware in raw string values before pd.to_datetime loses offset info."""
    import re
    _tz_offset_pattern = re.compile(r"[+-]\d{2}:\d{2}$|Z$")
    non_null = s.dropna()
    if len(non_null) == 0:
        return
    first_has_tz = bool(_tz_offset_pattern.search(str(non_null.iloc[0])))
    for v in non_null.iloc[1:]:
        if bool(_tz_offset_pattern.search(str(v))) != first_has_tz:
            raise RuntimeError("Timestamp series mixes tz-aware and tz-naive values. "
                               "Provide consistent timezone semantics in input data.")



def _make_staging_dir(parent_dir: Path, folder_name: str) -> Path:
    """Create a unique staging directory under *parent_dir* for atomic export writes."""
    staged_dir = parent_dir / f".{folder_name}.staged-{uuid4().hex}"
    staged_dir.mkdir(parents=True, exist_ok=False)
    return staged_dir


def _promote_staged_export(staged_dir: Path, target_dir: Path) -> None:
    """Promote fully-written staged export into place with rollback on failure."""
    backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid4().hex}"
    had_existing_target = target_dir.exists()

    try:
        if had_existing_target:
            target_dir.replace(backup_dir)
        staged_dir.replace(target_dir)
    except Exception as exc:
        rollback_error = None
        if had_existing_target and backup_dir.exists() and not target_dir.exists():
            try:
                backup_dir.replace(target_dir)
            except Exception as rb_exc:
                rollback_error = rb_exc
        if rollback_error is not None:
            raise RuntimeError(
                f"Failed to finalize export into {target_dir}. Rollback also failed: {rollback_error}"
            ) from exc
        raise RuntimeError(f"Failed to finalize export into {target_dir}: {exc}") from exc
    finally:
        _rmtree_if_exists(staged_dir)
        _rmtree_if_exists(backup_dir)


def _rmtree_if_exists(path: Path) -> None:
    """Best-effort removal for a path that may be a directory or file."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas scalars to JSON-safe types.

    NaN and Inf floats are converted to None (valid JSON null) since
    the JSON spec does not allow NaN/Infinity tokens.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Decimal):
        return None if not obj.is_finite() else float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, Enum):
        return _sanitize_for_json(obj.value)
    return obj
