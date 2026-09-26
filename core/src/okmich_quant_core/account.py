"""The account a running system is deployed in (LOGGING_CONTRACT §10, OPS §3.1).

On the live box every system is deployed in an account folder, ``<live_base>\\<account>\\...``, chosen by the
operator when the system is imported. The logs mirror that folder: a system deployed in
``<live_base>\\fxify.demo\\`` logs under ``<log_base>\\fxify.demo\\``. The framework reads the account from where
the running script sits — nothing is configured, passed or injected. A script outside ``<live_base>``, or in a
folder under it that is not an account (``<broker>.<env>``), has no account and logs flat under ``<log_base>``,
exactly as before the account layout (dev runs, flat deployments).

Which broker credentials a system logs in with stays its own ``run.py``'s business (its ``--env-file``); the
account folder decides only where it lives and logs. The Fleet Supervisor checks at run time that the terminal's
login matches the account folder's env file.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

LIVE_BASE_ENV_VAR = "OKMICH_QUANT_LIVE_BASE"

#: ``<broker>.<env>``, lower-case snake tokens: the name of an account folder (and of its ``.env.<account>`` file).
_ACCOUNT_PATTERN = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")


def is_account(name: str) -> bool:
    """Whether ``name`` is an account folder name, ``<broker>.<env>``."""
    return bool(_ACCOUNT_PATTERN.match(str(name)))


def deployment_account(script: str | Path | None = None, live_base: str | Path | None = None) -> str | None:
    """The account folder the running system is deployed in, or ``None`` when it is not deployed in one.

    ``script`` defaults to the running main script (the system's ``run.py``) and ``live_base`` to
    ``OKMICH_QUANT_LIVE_BASE``. The account is the first folder of the script's path below ``live_base``, when that
    folder is named like an account.
    """
    base = live_base if live_base is not None else os.environ.get(LIVE_BASE_ENV_VAR)
    path = Path(script) if script is not None else _main_script()
    if not base or not str(base).strip() or path is None:
        return None
    base_path = Path(os.path.expanduser(os.path.expandvars(str(base).strip()))).resolve()
    script_path = path.resolve()
    base_parts = [os.path.normcase(p) for p in base_path.parts]
    parts = script_path.parts
    if len(parts) <= len(base_parts) + 1 or [os.path.normcase(p) for p in parts[:len(base_parts)]] != base_parts:
        return None  # outside <live_base>, or directly in it
    first = parts[len(base_parts)]
    return first if is_account(first) else None


def _main_script() -> Path | None:
    """The running main script: ``__main__.__file__``, else ``sys.argv[0]``."""
    main = sys.modules.get("__main__")
    name = getattr(main, "__file__", None) or (sys.argv[0] if sys.argv and sys.argv[0] else None)
    return Path(name) if name else None
