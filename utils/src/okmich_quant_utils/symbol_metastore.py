import json
import os
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np


class SymbolMetastore:

    def __init__(self, **kwargs):
        self._metadata = None
        self.metastore_file = os.getenv("SYMBOL_METASTORE_FILE")
        if self.metastore_file is None:
            raise KeyError("SYMBOL_METASTORE_FILE environment variable not set")
        if not os.path.exists(self.metastore_file):
            raise FileNotFoundError(f"{self.metastore_file} does not exist")
        self._lock_file = self.metastore_file + ".lock"
        self._read_metastore_file()

    def get_servers(self):
        return list(self._metadata.keys())

    def get_property_value(self, server: str, timeframe: int, symbol: str, key: str):
        symbol_body = self._metadata.get(server, {}).get(str(timeframe), {}).get(symbol, {})
        return symbol_body.get(key)

    def get_symbol_properties(self, server: str, timeframe: int, symbol):
        return self._metadata.get(server, {}).get(str(timeframe), {}).get(symbol)

    def put_property_value(self, server: str, timeframe: int, symbol: str, key: str, value: Any):
        symbol_body = self._metadata.get(server, {}).get(str(timeframe), {}).get(symbol, {})
        symbol_body.setdefault(key, value)
        self.put_symbol_properties(server, timeframe, symbol, symbol_body)

    def put_symbol_properties(self, server: str, timeframe: int, symbol: str, key_values: Dict[str, Any]):
        with self._file_lock():
            self._read_metastore_file()
            symbol_body = self._metadata.setdefault(server, {}).setdefault(str(timeframe), {}).setdefault(symbol, {})
            symbol_body.update(key_values)
            self._write_metastore_file()

    def add_server(self, server: str) -> Dict[str, Any]:
        server_body = self._metadata.setdefault(server, {})
        self._write_metastore_file()
        return server_body

    def add_server_symbol(self, server: str, timeframe: int, symbol):
        sym_body = (
            self.add_server(server).setdefault(str(timeframe), {}).setdefault(symbol, {})
        )
        self._write_metastore_file()
        return sym_body

    @contextmanager
    def _file_lock(self, timeout: float = 30.0):
        """Cross-process file lock for safe concurrent writes."""
        import time
        import msvcrt
        lock_fd = None
        start = time.monotonic()
        while True:
            try:
                lock_fd = open(self._lock_file, "w")
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except (OSError, IOError):
                if lock_fd:
                    lock_fd.close()
                    lock_fd = None
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Could not acquire metastore lock after {timeout}s")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                lock_fd.close()

    def _read_metastore_file(self):
        with open(self.metastore_file, "r") as f:
            self._metadata = json.load(f)

    def _write_metastore_file(self):
        with open(self.metastore_file, "w") as f:
            f.writelines(json.dumps(self._metadata, indent=2, ensure_ascii=False))
