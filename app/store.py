from __future__ import annotations

import threading
from pathlib import Path

from app.jsonfile import read_json, write_json_atomic


class MessageStore:
    """Messages persisted as a JSON array in a single file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        with self._lock:
            return read_json(self.path, [])

    def add(self, message: dict) -> None:
        with self._lock:
            messages = read_json(self.path, [])
            messages.append(message)
            write_json_atomic(self.path, messages)
