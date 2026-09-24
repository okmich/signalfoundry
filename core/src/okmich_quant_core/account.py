"""The account a runner trades (LOGGING_CONTRACT §10, OPS §3.1).

An account is named by its broker env-file stem, ``<broker>.<env>`` (``fxify.demo``, ``deriv.live``,
``ib.paper``): one env file = one terminal + one login = one account. It is the first path level under
both ``OKMICH_QUANT_LIVE_BASE`` and ``OKMICH_QUANT_LOG_BASE``. The Fleet Supervisor injects it per process
as ``OKMICH_QUANT_ACCOUNT`` from the system's account folder, so the folder is the single source of truth:
the runner loads ``<OKMICH_QUANT_ENV_DIR>\\.env.<account>`` from it and the framework logs under
``<log_base>\\<account>``. There is no default account.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ACCOUNT_ENV_VAR = "OKMICH_QUANT_ACCOUNT"
ENV_DIR_ENV_VAR = "OKMICH_QUANT_ENV_DIR"

#: ``<broker>.<env>``, lower-case snake tokens: a safe single path component by construction.
_ACCOUNT_PATTERN = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")


class AccountConfigError(ValueError):
    """Raised when the account is missing or malformed, or its env file cannot be found."""


def validate_account(account: str) -> str:
    """Return ``account`` stripped, or raise if it is not a ``<broker>.<env>`` stem."""
    s = str(account).strip()
    if not _ACCOUNT_PATTERN.match(s):
        raise AccountConfigError(f"account {account!r} is not an env-file stem '<broker>.<env>' (e.g. 'fxify.demo')")
    return s


def resolve_account(account: str | None = None) -> str:
    """Resolve the account from an explicit value or ``OKMICH_QUANT_ACCOUNT``. Required: there is no default."""
    raw = account if account is not None else os.environ.get(ACCOUNT_ENV_VAR)
    if raw is None or str(raw).strip() == "":
        raise AccountConfigError(f"an explicit account or {ACCOUNT_ENV_VAR} is required (e.g. 'fxify.demo'); "
                                 "the Fleet Supervisor sets it from the system's account folder")
    return validate_account(raw)


def account_env_path(account: str | None = None, env_dir: str | Path | None = None) -> Path:
    """The account's broker env file: ``<env_dir>\\.env.<account>``, ``env_dir`` defaulting to ``OKMICH_QUANT_ENV_DIR``."""
    acct = resolve_account(account)
    raw_dir = env_dir if env_dir is not None else os.environ.get(ENV_DIR_ENV_VAR)
    if raw_dir is None or str(raw_dir).strip() == "":
        raise AccountConfigError(f"an explicit env_dir or {ENV_DIR_ENV_VAR} is required to locate .env.{acct}")
    return Path(os.path.expanduser(os.path.expandvars(str(raw_dir)))) / f".env.{acct}"


def load_account_env(account: str | None = None, env_dir: str | Path | None = None, *, env_file: str | Path | None = None,
                     override: bool = True) -> Path:
    """Load the account's broker env file into ``os.environ`` and return its path. Raises if it does not exist,
    so a runner can never start against a different account's credentials by falling back to a default.

    ``env_file`` is a manual-run override (a runner's ``--env-file``): that file is loaded instead. The account
    is still required for the log path, and the Supervisor's account check flags an override that disagrees."""
    path = Path(env_file) if env_file is not None else account_env_path(account, env_dir)
    if not path.is_file():
        raise AccountConfigError(f"broker env file not found: {path}")
    load_dotenv(dotenv_path=path, override=override)
    return path
