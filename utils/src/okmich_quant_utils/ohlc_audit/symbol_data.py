"""Everything the detectors need about one symbol, computed once."""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pandas as pd

from .calendar import SessionTemplate, expected_grid
from .findings import Finding
from .profile import BrokerProfile, SymbolSpec
from .enums import Action, FindingKind, Severity, SlotState


#: The only columns the audit requires. Anything else (MT5 tick_volume / spread, IB volume / average / barCount,
#: CCXT volume, ...) is carried along; tick_volume and spread are checked only when present.
BAR_COLUMNS = ("open", "high", "low", "close")


class NotABarSeries(ValueError):
    """The frame is not an OHLC bar series (e.g. CCXT funding_rate / open_interest datasets)."""


def check_bar_schema(df: pd.DataFrame, symbol: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise NotABarSeries(f"{symbol}: index must be a DatetimeIndex (got {type(df.index).__name__})")
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise NotABarSeries(f"{symbol}: not an OHLC bar series, missing {missing}; columns are {list(df.columns)}")
    bad = [c for c in BAR_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise NotABarSeries(f"{symbol}: non-numeric price columns {bad}")
    if len(df) == 0:
        raise NotABarSeries(f"{symbol}: no bars")


@dc.dataclass
class SymbolData:
    profile: BrokerProfile
    spec: SymbolSpec
    template: SessionTemplate
    df: pd.DataFrame                  # as stored, raw wall-clock index, NOT re-sorted (D1 must see order problems)
    raw: pd.DatetimeIndex
    utc: pd.DatetimeIndex             # aligned with raw (profile clock, confirmed overrides only)
    grid: pd.Series                   # SlotState value per UTC slot over the symbol's span
    unrepresentable: list[tuple[pd.Timestamp, pd.Timestamp]]
    blind_after: pd.Timestamp | None = None   # findings reaching past this raw stamp report no evidence (SPEC §8)
    pos: np.ndarray | None = None             # position of each row of ``df`` in the frame as given (rows without a stamp removed)
    missing_stamps: np.ndarray | None = None  # positions (in the frame as given) of rows whose timestamp is NaT

    @property
    def symbol(self) -> str:
        return self.spec.name

    @property
    def bar(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.profile.bar_minutes)

    @property
    def bars_per_day(self) -> int:
        return 1440 // self.profile.bar_minutes

    @classmethod
    def build(cls, profile: BrokerProfile, symbol: str, df: pd.DataFrame | None = None, blind_after: pd.Timestamp | None = None,
              span: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> SymbolData:
        """``span`` (UTC) widens the expected grid beyond the bars present, e.g. to the original file's span when the
        first or last bars were dropped, so the missing head and tail still show as missing."""
        spec = profile.symbol(symbol)
        if df is None:
            df = profile.load_raw(symbol)
        else:
            check_bar_schema(df, symbol)
            df = df.copy()
            df.index = pd.DatetimeIndex(df.index).tz_localize(None) if pd.DatetimeIndex(df.index).tz is not None else pd.DatetimeIndex(df.index)
        check_bar_schema(df, symbol)
        # rows without a timestamp can't be placed on any clock: quarantine them (D1 MISSING_STAMP drops them)
        stamped = ~np.asarray(pd.isna(df.index))
        pos, missing_stamps = np.flatnonzero(stamped), np.flatnonzero(~stamped)
        if len(missing_stamps):
            df = df[stamped]
            if df.empty:
                raise NotABarSeries(f"{symbol}: no bar has a timestamp")
        # one time unit everywhere: parquet indexes arrive in us, Timedelta.value is ns, and asi8 follows the index unit
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        raw = pd.DatetimeIndex(df.index)
        utc = profile.clock.to_utc(raw, symbol).as_unit("ns")
        template = profile.sessions[spec.session]
        lo, hi = utc.min(), utc.max() + pd.Timedelta(minutes=profile.bar_minutes)
        if span is not None:
            lo, hi = min(lo, span[0]), max(hi, span[1])
        grid = expected_grid(template, lo, hi, profile.bar_minutes)
        unrep = profile.clock.unrepresentable_utc(symbol, lo, hi)
        return cls(profile, spec, template, df, raw, utc, grid, unrep, pd.Timestamp(blind_after) if blind_after is not None else None,
                   pos, missing_stamps)

    # ---- finding constructors ---------------------------------------------------------------------------------------

    def _blind(self, start_raw: pd.Timestamp, end_raw: pd.Timestamp) -> bool:
        """Blind when any part of the window reaches past the cutoff: evidence for a window that crosses it would be
        computed from post-cutoff bars."""
        return self.blind_after is not None and pd.notna(end_raw) and end_raw > self.blind_after

    def _rows(self, idx) -> np.ndarray:
        return self.pos[idx] if self.pos is not None else np.arange(len(self.df))[idx]

    def from_positions(self, kind: FindingKind, severity: Severity, detector: str, a: int, b: int, action: Action, **evidence) -> Finding:
        """A finding over bars ``a .. b-1`` (positions in the stored order); it records exactly those rows."""
        raws, utcs = self.raw[a:b], self.utc[a:b]
        s_raw, e_raw = raws.min(), raws.max() + self.bar
        return Finding(self.symbol, kind, severity, detector, s_raw, e_raw, utcs.min(), utcs.max() + self.bar, b - a, action,
                       dict(evidence), blind=self._blind(s_raw, e_raw), rows=self._rows(slice(a, b)))

    def from_rows(self, kind: FindingKind, severity: Severity, detector: str, idx: np.ndarray, action: Action, **evidence) -> Finding:
        """A finding over exactly the bars at positions ``idx`` (any order, e.g. selected in raw-stamp order)."""
        raws, utcs = self.raw[idx], self.utc[idx]
        s_raw, e_raw = raws.min(), raws.max() + self.bar
        return Finding(self.symbol, kind, severity, detector, s_raw, e_raw, utcs.min(), utcs.max() + self.bar, len(idx), action,
                       dict(evidence), blind=self._blind(s_raw, e_raw), rows=self._rows(np.asarray(idx)))

    def missing_stamp_finding(self) -> Finding | None:
        """D1: rows whose timestamp is NaT. They have no window; the finding names them by position."""
        n = 0 if self.missing_stamps is None else len(self.missing_stamps)
        if not n:
            return None
        return Finding(self.symbol, FindingKind.MISSING_STAMP, Severity.CRITICAL, "D1", pd.NaT, pd.NaT, None, None, n, Action.DROP,
                       {"bars": n, "positions": ",".join(map(str, self.missing_stamps[:10].tolist()))}, rows=self.missing_stamps)

    def from_utc(self, kind: FindingKind, severity: Severity, detector: str, u0: pd.Timestamp, u1: pd.Timestamp, n_bars: int, action: Action,
                 **evidence) -> Finding:
        """A finding over the UTC slots [u0, u1) (e.g. missing bars, which have no raw stamp of their own)."""
        r0, r1 = self.raw_of(u0), self.raw_of(u1 - self.bar)
        s_raw = r0 if pd.notna(r0) else u0.tz_localize(None)
        e_raw = (r1 + self.bar) if pd.notna(r1) else u1.tz_localize(None)
        return Finding(self.symbol, kind, severity, detector, s_raw, e_raw, u0, u1, n_bars, action, dict(evidence),
                       blind=self._blind(s_raw, e_raw))

    def raw_of(self, u: pd.Timestamp) -> pd.Timestamp:
        """Raw stamp of a UTC slot: one vectorised clock inverse over the grid, cached; per-call outside the grid."""
        if "_raw_slots" not in self.__dict__:
            self.__dict__["_raw_slots"] = self.profile.clock.to_raw(self.grid.index, self.symbol)
        idx = self.grid.index
        i = idx.searchsorted(u)
        if i < len(idx) and idx[i] == u:
            return self.__dict__["_raw_slots"][i]
        return self.profile.clock.to_raw(pd.DatetimeIndex([u]), self.symbol)[0]

    # ---- shared views -----------------------------------------------------------------------------------------------

    def present_mask(self) -> np.ndarray:
        """Per grid slot: a bar exists there, or no raw stamp can exist there (fold / backward seam)."""
        present = self.grid.index.isin(self.utc)
        for u0, u1 in self.unrepresentable:
            present |= (self.grid.index >= u0) & (self.grid.index < u1)
        return present

    def bar_states(self) -> np.ndarray:
        """SlotState value of each bar's slot (NaN for off-grid stamps), in stored order."""
        return self.grid.reindex(self.utc).astype(object).to_numpy()

    def returns_grid(self) -> pd.Series:
        """Close-to-close returns on the full UTC slot grid (NaN where either bar is missing): log returns, or plain
        differences for series that can sit at or below 0 (``positive_prices=False``, e.g. a premium index)."""
        close = pd.Series(self.df["close"].to_numpy(dtype=float), index=self.utc)
        close = close[~close.index.duplicated(keep="last")].sort_index()
        full = close.reindex(self.grid.index)
        if not self.spec.positive_prices:
            return full.diff()
        with np.errstate(invalid="ignore", divide="ignore"):       # a mis-set positive_prices yields NaN, not a warning
            return np.log(full).diff()

    def open_like(self) -> np.ndarray:
        state = self.grid.to_numpy()
        return (state == SlotState.OPEN.value) | (state == SlotState.THIN.value)
