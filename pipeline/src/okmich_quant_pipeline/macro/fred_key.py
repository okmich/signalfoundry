r"""FRED API key loader.

The keyed FRED / ALFRED endpoints (point-in-time first-print vintages, full-history ICE HY-OAS)
require an API key; the anonymous ``fredgraph.csv`` path does not. The key is a **secret**: it is
read from ``$FRED_API_KEY`` or a file on disk (default ``E:\data_dump\.fred``) and must never be
logged, put in a ``repr``/exception message, or committed. Error messages here name only the
*source* (env var or path), never the value.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_KEY_PATH = Path(r"E:\data_dump\.fred")
ENV_VAR = "FRED_API_KEY"

# FRED API keys are 32 lowercase-hex-ish (alphanumeric) characters.
_KEY_RE = re.compile(r"^[a-z0-9]{32}$")


def load_fred_key(path: Path | str = DEFAULT_KEY_PATH) -> str:
    """Return the FRED API key from ``$FRED_API_KEY`` (preferred) or the key file.

    Raises ``FileNotFoundError`` if neither source is present and ``ValueError`` if the value is
    malformed. Neither the return value nor any exception message contains the key itself.
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        key, source = env, f"${ENV_VAR}"
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"FRED API key not found: set ${ENV_VAR} or create {p}")
        key, source = p.read_text().strip(), str(p)

    if not _KEY_RE.match(key):
        raise ValueError(f"FRED API key from {source} is malformed (expected 32 lowercase alphanumeric chars)")
    return key
