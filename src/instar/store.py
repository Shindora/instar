from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from instar.events import Event


class JsonlSink:
    """Appends each event to a JSONL file as one line, tagged with its type."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: Event) -> None:
        self._file.write(json.dumps({"type": type(event).__name__, **asdict(event)}) + "\n")

    def close(self) -> None:
        self._file.close()
