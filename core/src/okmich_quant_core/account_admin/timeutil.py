"""UTC timestamps as the Admin's files carry them: ISO-8601 with a ``Z`` suffix, never broker server time (spec §4.1)."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime | None) -> str | None:
    """``2026-09-24T13:02:41Z``: second precision, UTC. ``None`` passes through. A naive value is rejected, since
    guessing its zone is exactly the mistake the Admin's files exist to prevent."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime {dt!r}: the Admin's timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 instant that carries its zone (``Z`` or an offset) into an aware UTC datetime."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no zone; write it in UTC with a Z suffix")
    return dt.astimezone(timezone.utc)
