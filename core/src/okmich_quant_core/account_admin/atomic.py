"""Atomic JSON writes and retried JSON reads for files one process writes and others read (spec §7.4, §10.2).

A write goes to a temp file in the same directory, is flushed and fsynced, then replaces the target, so a reader sees
the old file or the new one, never half of one. On Windows the replace fails with a sharing violation while any reader
has the target open, so the writer retries it briefly; a reader that lands on the replace retries too.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

#: Writer: attempts at the final replace, and the growing pause between them (spec §7.4). A reader holds the file for
#: a millisecond or two, so a replace that keeps colliding is rare; ten attempts (about 2.75 s at most) keep it rare
#: under far more read traffic than a fleet produces, well inside the Admin's cycle.
REPLACE_ATTEMPTS = 10
REPLACE_BACKOFF_S = 0.05

#: Reader: attempts at open+parse within about 200 ms before a failure counts (spec §10.2).
READ_ATTEMPTS = 5
READ_BACKOFF_S = 0.04


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, attempts: int = REPLACE_ATTEMPTS,
                      backoff_s: float = REPLACE_BACKOFF_S) -> None:
    """Write ``payload`` to ``path`` atomically and durably. Raises ``OSError`` when every replace attempt failed; the
    temp file is removed in that case so failed cycles do not accumulate litter."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, allow_nan=False, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    last_error: OSError | None = None
    for attempt in range(max(1, attempts)):
        try:
            os.replace(tmp, target)
            return
        except PermissionError as exc:   # Windows sharing violation: a reader holds the target open
            last_error = exc
            time.sleep(backoff_s * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    raise last_error if last_error is not None else OSError(f"could not replace {target}")


def _reject_constant(name: str):
    """Strict JSON: Python's json accepts NaN/Infinity, which the Admin's own writers refuse (allow_nan=False). A file
    carrying them is malformed, not a value to act on."""
    raise ValueError(f"non-standard JSON constant {name}")


def read_json_retry(path: str | Path, *, attempts: int = READ_ATTEMPTS, backoff_s: float = READ_BACKOFF_S) -> Any:
    """Read and parse ``path``, retrying a failed open or parse briefly.

    ``FileNotFoundError`` is raised at once, never retried: an atomic replace never makes the target disappear, so a
    missing file is a real absence. Any other failure is retried, then the last error is raised.
    """
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh, parse_constant=_reject_constant)
        except FileNotFoundError:
            raise
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(backoff_s)
    raise last_error  # type: ignore[misc]
