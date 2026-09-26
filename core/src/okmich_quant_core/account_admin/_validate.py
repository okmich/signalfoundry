"""Config validation helpers shared by the host and every task (ACCOUNT_ADMIN_SPEC §6.7).

Validation collects EVERY problem rather than stopping at the first, so an operator fixes a config in one pass. Each
helper records a problem under a dotted path (``tasks[0].policy.daily_loss_pct``) and returns the parsed value, or
``None`` when it was missing or invalid.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Mapping, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .timeutil import parse_utc

E = TypeVar("E", bound=enum.Enum)

_MISSING = object()


class Problems:
    """An accumulating list of config problems, each prefixed with where it was found."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.items: list[str] = []

    def at(self, key: str | int) -> str:
        if isinstance(key, int):
            return f"{self.prefix}[{key}]"
        return f"{self.prefix}.{key}" if self.prefix else str(key)

    def add(self, where: str, message: str) -> None:
        self.items.append(f"{where}: {message}")

    def child(self, key: str | int) -> "Problems":
        """A collector for a nested section that appends to this one's list."""
        nested = Problems(self.at(key))
        nested.items = self.items
        return nested

    def __bool__(self) -> bool:
        return bool(self.items)


def section(raw: Mapping[str, Any], key: str, p: Problems, *, required: bool = True) -> Mapping[str, Any] | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        if required:
            p.add(p.at(key), "missing")
        return None
    if not isinstance(value, Mapping):
        p.add(p.at(key), f"must be an object, got {type(value).__name__}")
        return None
    return value


def integer(raw: Mapping[str, Any], key: str, p: Problems, *, minimum: int | None = None, maximum: int | None = None,
            required: bool = True) -> int | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        if required:
            p.add(p.at(key), "missing")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        p.add(p.at(key), f"must be an integer, got {value!r}")
        return None
    if minimum is not None and value < minimum:
        p.add(p.at(key), f"must be >= {minimum}, got {value}")
        return None
    if maximum is not None and value > maximum:
        p.add(p.at(key), f"must be <= {maximum}, got {value}")
        return None
    return value


def number(raw: Mapping[str, Any], key: str, p: Problems, *, gt: float | None = None, lt: float | None = None,
           required: bool = True, nullable: bool = False) -> float | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING:
        if required:
            p.add(p.at(key), "missing")
        return None
    if value is None:
        if not nullable:
            p.add(p.at(key), "must not be null")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        p.add(p.at(key), f"must be a number, got {value!r}")
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        p.add(p.at(key), "must be finite")
        return None
    if gt is not None and not value > gt:
        p.add(p.at(key), f"must be > {gt}, got {value}")
        return None
    if lt is not None and not value < lt:
        p.add(p.at(key), f"must be < {lt}, got {value}")
        return None
    return value


def boolean(raw: Mapping[str, Any], key: str, p: Problems) -> bool | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        p.add(p.at(key), "missing")
        return None
    if not isinstance(value, bool):
        p.add(p.at(key), f"must be true or false, got {value!r}")
        return None
    return value


def text(raw: Mapping[str, Any], key: str, p: Problems, *, required: bool = True) -> str | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        if required:
            p.add(p.at(key), "missing")
        return None
    if not isinstance(value, str) or not value.strip():
        p.add(p.at(key), f"must be a non-empty string, got {value!r}")
        return None
    return value.strip()


def choice(raw: Mapping[str, Any], key: str, enum_type: type[E], p: Problems, *, required: bool = True,
           allowed: frozenset | set | None = None) -> E | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        if required:
            p.add(p.at(key), "missing")
        return None
    try:
        parsed = enum_type(value)
    except ValueError:
        options = ", ".join(str(m.value) for m in enum_type)
        p.add(p.at(key), f"unknown value {value!r} (one of: {options})")
        return None
    if allowed is not None and parsed not in allowed:
        p.add(p.at(key), f"{parsed.value!r} is not allowed here (one of: {', '.join(sorted(str(a.value) for a in allowed))})")
        return None
    return parsed


def timestamp(raw: Mapping[str, Any], key: str, p: Problems) -> datetime | None:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        p.add(p.at(key), "missing")
        return None
    try:
        return parse_utc(value)
    except (TypeError, ValueError) as exc:
        p.add(p.at(key), f"must be an ISO-8601 UTC timestamp: {exc}")
        return None


def timezone_name(raw: Mapping[str, Any], key: str, p: Problems) -> ZoneInfo | None:
    name = text(raw, key, p)
    if name is None:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        p.add(p.at(key), f"unknown IANA time zone {name!r}")
        return None


def unknown_keys(raw: Mapping[str, Any], known: set[str] | frozenset[str], p: Problems) -> None:
    """A key the schema does not know is a typo until proven otherwise; refuse it rather than ignore a setting."""
    for key in raw:
        if key not in known:
            p.add(p.at(key), "unknown key")
