"""The Account Admin, broker-neutral part (docs: signalfoundry-lab ``docs/ops/ACCOUNT_ADMIN_SPEC.md``).

Contracts shared by the Admin and every reader (the directive file, the state file, requests, audit records, enums),
the admin-task framework with its built-in tasks, the host that runs a cycle, and the runner the Supervisor starts. A
broker library supplies the account (:class:`AccountSource`) and the two book actions (:class:`BrokerActions`).
"""

from .clock import AdminClock, BlackoutWindow, ClockConfig, gap_max, longest_blackout_run
from .config import AdminConfig, AdminConfigError, load_admin_config, parse_admin_config
from .directive import (ADMIN_FOLDER, DIRECTIVE_FILE, DIRECTIVE_SCHEMA_VERSION, DirectiveAccount, DirectiveFile,
                        DirectiveFormatError, DirectiveReading, admin_dir, directive_path, read_directive, write_directive)
from .enums import (AccountDirective, AdminEvent, AdminTaskKind, AlertLevel, BookActionKind, BookActionOutcome,
                    CalendarKind, CleanupScope, Condition, DailyBase, DealEntry, DealKind, DirectiveSource, LatchScope,
                    LimitReference, MaxLossMode, PendingOrderType, PositionSide, RequestKind, RequestOutcome,
                    TargetMeasure, most_restrictive)
from .host import AccountIdentity, AdminHost, AdminNotifier, CycleReport, NotifierAlerts
from .lock import WriterLock, WriterLockHeld
from .requests import AdminRequest, RequestInbox, submit_request
from .runner import AdminDeployment, AdminRunLoop, resolve_deployment
from .snapshot import (AccountInfo, AccountSnapshot, AccountSource, BookActionResult, BrokerActions, Deal, PendingOrder,
                       Position)
from .state import StateLoad, load_state, save_state
from .tasks import TASK_REGISTRY, AdminTask, PendingOrderCleanupTask, PropGuardTask, register_task

__all__ = [name for name in dir() if not name.startswith("_")]
