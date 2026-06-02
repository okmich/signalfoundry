"""Identity value objects for the inference-log envelope (LOGGING_CONTRACT §6).

Identity is split by scope:

* :class:`LogicalSystemIdentity` — the per-strategy unit of bar-freshness and blast
  radius (``<strategy>/<symbol>/<timeframe>``). Known at strategy construction time, so
  the fail-closed default logger's file path is derivable from it alone.
* :class:`RunnerIdentity` — the per-process (one PID) unit of startup/shutdown control:
  ``runner_id`` + ``runner_start_token`` (both process-generated) plus the broker-session
  facts (``broker``/``account_id``/``broker_session_id``) sourced from the live broker at
  startup.

Both are pure value objects with no dependency on the record schema or the logger, so they
can be constructed early (logical) and late (runner) without import cycles.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


class LogRootConfigError(ValueError):
    """Raised when the inference-log root was not supplied by config or environment."""


class IdentityTokenError(ValueError):
    """Raised when a strategy/symbol identity token cannot be a safe single path component."""


#: Characters invalid in a filesystem path component (Windows reserved set + control chars). Path
#: separators are handled by _path_safe; ''/'.'/'..' are rejected explicitly in _validate_identity_token.
_RESERVED_PATH_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def _validate_identity_token(kind: str, value: str) -> str:
    """Reject an identity token that would escape or collapse the log tree as a path component.

    Bars empty / '.' / '..' (which traverse out of or collapse the tree) and reserved filesystem
    characters. A clean failure at identity-construction time beats a silently re-homed log tree —
    e.g. a strategy named '..' resolving to ``<log_base>/../...``. Mid-token dots ('BTCUSDT.r') are fine.
    """
    s = str(value).strip()
    if not s or s in (".", ".."):
        raise IdentityTokenError(f"{kind} identity token {value!r} is empty or a path-traversal component ('.'/'..')")
    if _RESERVED_PATH_CHARS.search(s):
        raise IdentityTokenError(f"{kind} identity token {value!r} contains a reserved filesystem character")
    # NB: path separators ('/' '\\') are intentionally NOT rejected — _path_safe maps them to '_', which is
    # traversal-safe ('../x' -> '.._x'); only the bare '.'/'..'/'' components above can actually escape.
    return s


def _resolve_log_base(log_base: str | Path | None = None) -> Path:
    """Resolve the inference-log root from an explicit value or ``OKMICH_QUANT_LOG_BASE``.

    There is deliberately no hardcoded production fallback: deployment paths such as
    ``D:\\quant_logs`` are ops examples, not portable source-code defaults. Shared by the bar
    logger and the runner status writer so both resolve the root identically.
    """
    source = "explicit log_base"
    raw = log_base
    if raw is None:
        source = "OKMICH_QUANT_LOG_BASE"
        raw = os.environ.get("OKMICH_QUANT_LOG_BASE")
    if raw is None or str(raw).strip() == "":
        raise LogRootConfigError(
            "an explicit log_base or OKMICH_QUANT_LOG_BASE is required; "
            "no hardcoded deployment fallback is used."
        )
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    if expanded.strip() == "":
        raise LogRootConfigError(f"resolved {source} to an empty path")
    return Path(expanded)


def _path_safe(segment: str) -> str:
    """Make a single identity token safe to use as one path component.

    Replaces path separators only — identifiers are expected to already be clean
    (``EURUSD``, ``deriv_hmm_posterior``); this is a guard, not a sanitiser.
    """
    return str(segment).replace("/", "_").replace("\\", "_").strip()


@dataclass(frozen=True)
class LogicalSystemIdentity:
    """The logical-system scope: ``<strategy>/<symbol>/<timeframe-minutes>``.

    ``timeframe_minutes`` is the broker-neutral integer the envelope's ``timeframe`` field
    carries (LOGGING_CONTRACT §6) and the path's ``<timeframe>`` segment uses, so MT5 and IB
    logs for the same cadence sort and compare on one axis regardless of each broker's native
    timeframe encoding (MT5 constant int vs IB ``"5 mins"`` string).
    """

    strategy: str
    symbol: str
    timeframe_minutes: int

    def __post_init__(self):
        # Fail fast: a malformed strategy/symbol must not flow into inference/status paths (where it could
        # escape or collapse the log tree). timeframe_minutes is an int (str-ified to digits) — safe.
        _validate_identity_token("strategy", self.strategy)
        _validate_identity_token("symbol", self.symbol)

    @property
    def logical_system_id(self) -> str:
        return f"{self.strategy}/{self.symbol}/{self.timeframe_minutes}"

    def path_parts(self) -> tuple[str, str, str]:
        """The three path components below ``<log_base>`` (before ``inference/``)."""
        return _path_safe(self.strategy), _path_safe(self.symbol), str(self.timeframe_minutes)

    def with_strategy(self, strategy: str) -> "LogicalSystemIdentity":
        """A copy under a different strategy code — used at the runner phase to apply the runner-root
        (e.g. the statutory ``-multi`` suffix) that is only known once the dispatch type is."""
        return LogicalSystemIdentity(strategy=strategy, symbol=self.symbol, timeframe_minutes=self.timeframe_minutes)


def runner_strategy_root(strategy: str, *, multi: bool) -> str:
    """The runner-root strategy folder: the plain strategy code, with a statutory ``-multi`` suffix for a
    multi-trader (one process, N symbols, one broker session). The hyphen marks the runner-type tag,
    visually distinct from the snake_case strategy code, and keeps a single-trader and a multi-trader of
    the same strategy from colliding in the log tree (LOGGING_CONTRACT §7.1/§10). Idempotent: applying it
    to an already-suffixed code is a no-op (so a re-bind never produces ``-multi-multi``)."""
    if not multi or strategy.endswith("-multi"):
        return strategy
    return f"{strategy}-multi"


@dataclass(frozen=True)
class RunnerIdentity:
    """The runner (OS process) scope. One per PID; shared by every record the runner emits.

    ``runner_id``/``runner_start_token`` are process-generated (the Fleet Supervisor that
    would otherwise hand them down does not exist yet). ``broker``/``account_id`` come from the
    runner's ``.env`` login info; ``broker_session_id`` is read from the live broker session
    (``mt5.terminal_info()`` / IB ``clientId``) and may be ``None`` if unavailable.
    """

    runner_id: str
    runner_start_token: str
    broker: str
    account_id: str
    broker_session_id: str | None = None

    @classmethod
    def generate(cls, *, name: str, broker: str, account_id: str,
                 broker_session_id: str | None = None, pid: int | None = None) -> "RunnerIdentity":
        """Build a runner identity with a process-generated id + start token.

        ``runner_id`` = ``<name>-<pid>`` (stable for the life of the process);
        ``runner_start_token`` = a fresh uuid4 so ops can bind records to one incarnation and
        ignore stale ones across a restart.
        """
        resolved_pid = os.getpid() if pid is None else pid
        return cls(runner_id=f"{name}-{resolved_pid}", runner_start_token=uuid.uuid4().hex,
                   broker=broker, account_id=account_id, broker_session_id=broker_session_id)
