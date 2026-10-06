"""Process-level console control for runners the Fleet Supervisor stops with a targeted Ctrl+C (FLEET_SUPERVISOR_SPEC §9).

On Windows a process can be started with Ctrl+C *ignored*, and every child inherits that flag, including one given a
new console. A runner launched from such a parent (a service wrapper, some shells and IDEs) never sees the Supervisor's
Ctrl+C: the graceful stop silently degrades to a hard kill after the stop timeout, with no stopped status and no
proven disconnect. Found while testing the Account Admin against the Supervisor's own stop path.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def reenable_ctrl_c() -> bool:
    """Clear an inherited "ignore Ctrl+C" flag so a console Ctrl+C reaches this process's SIGINT handling. Returns
    whether it did anything (always ``False`` off Windows). Never raises."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False))
    except Exception:
        logger.debug("could not re-enable Ctrl+C handling", exc_info=True)
        return False
