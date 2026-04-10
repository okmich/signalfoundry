import logging
import sys
from pathlib import Path
from typing import Optional


class EnvironmentDetector:
    """Automatically detects runtime environment."""

    @staticmethod
    def is_colab() -> bool:
        try:
            import google.colab

            return True
        except ImportError:
            return False

    @staticmethod
    def is_jupyter() -> bool:
        try:
            from IPython import get_ipython

            return get_ipython() is not None
        except ImportError:
            return False

    @staticmethod
    def is_interactive() -> bool:
        return EnvironmentDetector.is_colab() or EnvironmentDetector.is_jupyter()

    @staticmethod
    def get_default_checkpoint_dir(
        folder_name: Optional[str] = "wfa_checkpoints",
    ) -> str:
        folder_name = "wfa_checkpoints" if folder_name is None else folder_name
        if EnvironmentDetector.is_colab():
            drive_path = Path("/content/drive/MyDrive")
            if drive_path.exists():
                return f"/content/drive/MyDrive/{folder_name}"
            return f"/content/{folder_name}"
        return f"./{folder_name}"


class UniversalLogger:
    """Environment-aware logging."""

    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.is_interactive = EnvironmentDetector.is_interactive()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self._get_level(verbose))
        self.logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(message)s"
            if self.is_interactive
            else "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _get_level(self, verbose: int) -> int:
        return (
            logging.WARNING
            if verbose == 0
            else (logging.DEBUG if verbose == 2 else logging.INFO)
        )

    def _format_msg(self, msg: str, emoji: str = "") -> str:
        return f"{emoji} {msg}" if self.is_interactive and emoji else msg

    def info(self, msg: str, emoji: str = ""):
        self.logger.info(self._format_msg(msg, emoji))

    def warning(self, msg: str, emoji: str = "⚠"):
        self.logger.warning(self._format_msg(msg, emoji))

    def debug(self, msg: str, emoji: str = ""):
        self.logger.debug(self._format_msg(msg, emoji))
