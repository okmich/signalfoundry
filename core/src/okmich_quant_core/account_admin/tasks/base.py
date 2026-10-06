"""The admin-task framework (ACCOUNT_ADMIN_SPEC §12.1).

The Admin is a host; tasks do the work. A task owns one job: it validates its own ``tasks[]`` entry before the Admin
touches the terminal, keeps its own slot of the state file, publishes its own output every cycle, and acts on the book
only through the :class:`ActionPort` the host hands it, which offers nothing that opens or adds risk.

Adding a task kind: specify it in ACCOUNT_ADMIN_SPEC §12 first, add its :class:`AdminTaskKind`, implement
:class:`AdminTask`, and decorate the class with :func:`register_task`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Protocol

from ..enums import AdminEvent, AdminTaskKind, AlertLevel, RequestOutcome
from ..owners import MagicOwner
from ..requests import AdminRequest
from ..snapshot import AccountSnapshot, BookActionResult, Deal, PendingOrder, Position

#: Kinds that would collide with a governance file's name (spec §3.1).
RESERVED_OUTPUT_STEMS = frozenset({"directive", "state", "writer", "requests"})


@dataclass(frozen=True)
class TaskEvent:
    """An audit record the task asks the host to write (spec §7.7)."""

    event: AdminEvent
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    """A Telegram alert the task asks the host to send (spec §7.8)."""

    level: AlertLevel
    title: str
    body: str = ""


@dataclass
class TaskResult:
    """What one cycle of a task produced. ``slot`` replaces its state slot; ``output`` is written to its output file;
    ``request_outcomes`` maps each routed request id to (outcome, reason); ``audit_ids`` optionally supplies the
    ``sequence``/``episode`` the host stamps on every audit record (only ``prop_guard`` does)."""

    slot: dict[str, Any]
    output: dict[str, Any]
    events: list[TaskEvent] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    request_outcomes: dict[str, tuple[RequestOutcome, str]] = field(default_factory=dict)
    audit_ids: tuple[int | None, int | None] | None = None
    #: A compact summary for the host's per-cycle ``cycle`` audit record.
    summary: dict[str, Any] = field(default_factory=dict)


class ActionPort(Protocol):
    """The ONLY way a task acts on the book (invariant 3). Both actions reduce risk. The host refuses both in a degraded
    cycle, and audits and reports every call. ``reason`` is recorded with the action."""

    def cancel_pending(self, order: PendingOrder, *, reason: str) -> BookActionResult:
        ...

    def close_position(self, position: Position, *, reason: str) -> BookActionResult:
        ...


class DealHistory(Protocol):
    def __call__(self, from_utc: datetime, to_utc: datetime) -> list[Deal]:
        ...


@dataclass
class TaskContext:
    """Everything a task may use in one cycle.

    ``snapshot`` is ``None`` in a degraded cycle (the account could not be read or verified); a task must then keep its
    state, publish its output, and not act. ``deals`` reads deal history on demand (it raises on failure). ``owners``
    lazily maps magic numbers to this account's systems. ``first_cycle`` is true on the Admin's first cycle after a
    start; ``state_load`` is how the state file loaded at that start, and ``previous_output`` the task's output file as
    found at that start (``None`` when absent or unreadable), so a task that lost its slot can tell a first-ever start
    from lost history.
    """

    now: datetime
    account: str
    degraded: bool
    degraded_reason: str | None
    snapshot: AccountSnapshot | None
    slot: dict[str, Any]
    requests: list[AdminRequest]
    actions: ActionPort
    deals: DealHistory
    owners: Callable[[], dict[int, MagicOwner]]
    valid_for_s: int
    broker_label: str
    expected_login: int
    expected_server: str
    first_cycle: bool
    state_load: str
    previous_output: Mapping[str, Any] | None


class AdminTask(ABC):
    """One admin task. Constructed only from an entry that :meth:`validate` accepted."""

    kind: ClassVar[AdminTaskKind]
    #: Whether this task writes the directive, the one output trading systems read (only ``prop_guard``).
    governs_directive: ClassVar[bool] = False
    #: The config keys every task entry may carry besides its own.
    COMMON_KEYS: ClassVar[frozenset[str]] = frozenset({"kind"})

    def __init__(self, entry: Mapping[str, Any]):
        self.entry = entry

    @classmethod
    @abstractmethod
    def validate(cls, entry: Mapping[str, Any], prefix: str) -> list[str]:
        """Every problem in this task's own config entry, each prefixed with ``prefix``; empty when it can run. Called
        in startup stage 1, before any terminal is attached (spec §6.7)."""

    @property
    def output_name(self) -> str:
        """The output file this task publishes each cycle: ``<kind>.json`` unless the task says otherwise."""
        return f"{self.kind.value}.json"

    @abstractmethod
    def on_cycle(self, ctx: TaskContext) -> TaskResult:
        """One cycle's work. May raise: the host isolates the failure and keeps the task's last slot and output."""


#: Registered task classes by kind. Populated by :func:`register_task`.
TASK_REGISTRY: dict[AdminTaskKind, type[AdminTask]] = {}


def register_task(cls: type[AdminTask]) -> type[AdminTask]:
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, AdminTaskKind):
        raise TypeError(f"{cls.__name__}.kind must be an AdminTaskKind")
    if kind.value in RESERVED_OUTPUT_STEMS:
        raise ValueError(f"task kind {kind.value!r} collides with a governance file name")
    if kind in TASK_REGISTRY and TASK_REGISTRY[kind] is not cls:
        raise ValueError(f"task kind {kind.value!r} is already registered by {TASK_REGISTRY[kind].__name__}")
    TASK_REGISTRY[kind] = cls
    return cls


def output_path(admin_dir: Path, task: AdminTask) -> Path:
    return admin_dir / task.output_name
