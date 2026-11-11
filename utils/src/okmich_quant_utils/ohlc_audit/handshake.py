"""Overlap handshake (SPEC §6.4): fetched bars are trusted only if their margins reproduce the stored bars exactly.

A refetch asks for the target window plus >= 12 trusted stored bars on each side. The margin bars are matched to the
fetch by exact OHLC equality:
- An offset k (fetched stamp - stored stamp, whole bars, |k| <= 12 h) qualifies when it reproduces >= 80% of the margin
  bars on **both** sides. Every offset a margin bar votes for is checked.
  - k = 0 wins whenever it qualifies: there is then no evidence that the clock moved.
  - Otherwise exactly one qualifying offset is accepted, and the fetched bars are restamped by -k. This absorbs a
    broker clock change between the export and the refetch, which the SF fetcher's timestamp-only merge could not do.
  - Several qualifying offsets are ambiguous (e.g. flat, repeated bars): the margins can't tell the clock, so reject.
- Otherwise the attempt is rejected: no offset reproduces both margins, or the margins disagree on price.
"""
from __future__ import annotations

import dataclasses as dc
from collections import Counter

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
MIN_MARGIN_BARS = 12
MIN_AGREEMENT = 0.8
MAX_SHIFT = pd.Timedelta(hours=12)


@dc.dataclass(frozen=True)
class Handshake:
    ok: bool
    shift: pd.Timedelta          # k: fetched stamp - stored stamp; fetched bars are restamped by -k
    left: float                  # share of the left margin reproduced at k
    right: float                 # share of the right margin reproduced at k
    reason: str = ""

    @property
    def shift_minutes(self) -> int:
        return int(self.shift / pd.Timedelta(minutes=1))


def _unique(df: pd.DataFrame) -> pd.DataFrame:
    """Unique stamps (first wins), in ns: parquet indexes load in us, and the offset arithmetic below is on integers."""
    out = df[~df.index.duplicated(keep="first")].copy()
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def handshake(stored: pd.DataFrame, fetched: pd.DataFrame, left: pd.DatetimeIndex, right: pd.DatetimeIndex, bar: pd.Timedelta,
              min_margin: int = MIN_MARGIN_BARS, min_agree: float = MIN_AGREEMENT, max_shift: pd.Timedelta = MAX_SHIFT) -> Handshake:
    """Align ``fetched`` to ``stored`` (both on naive raw stamps) using the margin stamps ``left`` / ``right``."""
    zero = pd.Timedelta(0)
    if len(left) < min_margin or len(right) < min_margin:
        return Handshake(False, zero, 0.0, 0.0, f"too few trusted margin bars (left {len(left)}, right {len(right)}; need {min_margin})")
    fetched = _unique(fetched)
    if fetched.empty:
        return Handshake(False, zero, 0.0, 0.0, "the fetch is empty")
    left, right = pd.DatetimeIndex(left).as_unit("ns"), pd.DatetimeIndex(right).as_unit("ns")
    margins = left.append(right)
    m_vals = stored.loc[margins, PRICE_COLUMNS].to_numpy(dtype=float)
    f_vals = fetched[PRICE_COLUMNS].to_numpy(dtype=float)
    m_hash = pd.util.hash_pandas_object(pd.DataFrame(m_vals), index=False).to_numpy()
    f_hash = pd.util.hash_pandas_object(pd.DataFrame(f_vals), index=False).to_numpy()
    order = np.argsort(f_hash, kind="stable")
    fh, fns = f_hash[order], fetched.index.asi8[order]
    votes: Counter[int] = Counter()
    for h, t in zip(m_hash, margins.asi8, strict=True):
        lo, hi = np.searchsorted(fh, h, "left"), np.searchsorted(fh, h, "right")
        d = fns[lo:hi] - t
        votes.update({int(x) for x in d[(np.abs(d) <= max_shift.value) & (d % bar.value == 0)]})

    def reproduced(k: int) -> np.ndarray:
        pos = fetched.index.get_indexer(pd.DatetimeIndex(margins.asi8 + k))
        ok = pos >= 0
        hit = np.zeros(len(margins), dtype=bool)
        hit[ok] = (f_vals[pos[ok]] == m_vals[ok]).all(axis=1)
        return hit

    if not votes:
        present = int((fetched.index.get_indexer(margins) >= 0).sum())
        why = (f"price disagreement: the fetch has {present} of {len(margins)} margin stamps, none with the stored prices" if present
               else "the fetch has none of the margin bars")
        return Handshake(False, zero, 0.0, 0.0, why)
    def shares(v: int) -> tuple[float, float]:
        hit = reproduced(v)
        return float(hit[:len(left)].mean()), float(hit[len(left):].mean())

    scored = {v: shares(v) for v in votes}
    qualifying = sorted((v for v, (lf, rf) in scored.items() if lf >= min_agree and rf >= min_agree), key=lambda v: (abs(v), v))
    if 0 in qualifying:
        return Handshake(True, zero, *scored[0])
    if len(qualifying) == 1:
        return Handshake(True, pd.Timedelta(qualifying[0], unit="ns"), *scored[qualifying[0]])
    if qualifying:
        shown = ", ".join(str(pd.Timedelta(v, unit="ns")) for v in qualifying[:4])
        return Handshake(False, zero, 0.0, 0.0, f"ambiguous: {len(qualifying)} offsets reproduce both margins ({shown}); the margins can't tell the clock")
    k = max(votes, key=lambda v: (votes[v], -abs(v)))
    lf, rf = scored[k]
    shift = pd.Timedelta(k, unit="ns")
    at_zero = fetched.index.get_indexer(margins) >= 0
    differ = int((at_zero & ~reproduced(0)).sum())
    why = f"no single offset reproduces both margins (best {shift}: left {lf:.0%}, right {rf:.0%})"
    if differ:
        why += f"; {differ} margin bars differ in price at their own stamps"
    return Handshake(False, shift, lf, rf, why)


def restamp(fetched: pd.DataFrame, hs: Handshake) -> pd.DataFrame:
    """The fetched bars on the stored clock (stamp - k)."""
    out = _unique(fetched)
    out.index = pd.DatetimeIndex(out.index - hs.shift, name=out.index.name)
    return out
