"""The Admin's writer lock (ACCOUNT_ADMIN_SPEC §7.4): an exclusive OS lock on ``writer.lock``, held for the Admin's
lifetime, so a second Admin for the same account fails to start. It is the Admin's only lock; readers never lock."""

from __future__ import annotations

import os
from pathlib import Path

WRITER_LOCK_FILE = "writer.lock"


class WriterLockHeld(RuntimeError):
    """Another process holds the writer lock for this account."""


class WriterLock:
    """Non-blocking exclusive lock. The OS releases it if the process dies, so a crashed Admin never blocks a restart."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise WriterLockHeld(f"{self.path} is held by another Account Admin for this account") from exc
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()).encode("ascii"))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()

    @property
    def held(self) -> bool:
        return self._fh is not None

    def __enter__(self) -> "WriterLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
