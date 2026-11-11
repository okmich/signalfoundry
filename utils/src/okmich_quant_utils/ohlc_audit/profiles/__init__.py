"""Broker profiles by name."""
from __future__ import annotations

from pathlib import Path

from ..profile import MARKET_DATA_DIR_VAR, BrokerProfile, resolve_market_data_dir

PROFILES = ("FXPIG-Server",)


def get_profile(name: str, data_dir: str | Path | None = None, require_data: bool = True) -> BrokerProfile:
    """Build a broker profile rooted at the market-data folder (``data_dir`` argument, else OKMICH_QUANT_MARKET_DATA_DIR).

    ``require_data=False`` builds it without a data root (for bars supplied in memory); reading files then fails clearly.
    """
    if name not in PROFILES:
        raise KeyError(f"no profile for broker {name!r}; known: {list(PROFILES)}")
    raw_root = None
    if require_data:
        raw_root = resolve_market_data_dir(data_dir) / "raw"
    elif data_dir:
        raw_root = Path(data_dir) / "raw"
    from . import fxpig
    return fxpig.build(raw_root)


__all__ = ["PROFILES", "get_profile", "MARKET_DATA_DIR_VAR"]
