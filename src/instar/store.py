from __future__ import annotations

import json
import queue
import sqlite3
import threading
import types
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import cache, singledispatchmethod
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

from instar.events import CheckResult, Execution, LLMCall, Metric, Run, Span

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TextIO

    from _typeshed import DataclassInstance

    from instar.events import Event

DB_NAME = "run.sqlite"
JSONL_NAME = "events.jsonl"
BATCH = 500

TABLES: dict[type[DataclassInstance], str] = {
    Run: "runs",
    LLMCall: "llm_calls",
    Execution: "executions",
    Metric: "metrics",
    Span: "spans",
    CheckResult: "check_results",
}
UNIQUE: dict[type[DataclassInstance], str] = {Run: "run_id"}
SQL_TYPES: dict[object, str] = {int: "INTEGER", float: "REAL", str: "TEXT"}


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    sql_type: str
    decode: Callable[[Any], object]
    as_json: bool = False


def _maybe(convert: Callable[[Any], object]) -> Callable[[Any], object]:
    return lambda value: None if value is None else convert(value)


def _column(name: str, hint: object) -> Column:
    """Scalars keep their SQL type, bools and enums are restored on read, the rest is JSON."""
    if isinstance(hint, types.UnionType):
        options = [arg for arg in get_args(hint) if arg is not type(None)]
    else:
        options = [hint]
    if len(options) == 1:
        [base] = options
        if base is bool:
            return Column(name, "INTEGER", _maybe(bool))
        if isinstance(base, type) and issubclass(base, Enum):
            return Column(name, "TEXT", _maybe(base))
        if base in SQL_TYPES:
            return Column(name, SQL_TYPES[base], lambda value: value)
    return Column(name, "TEXT", _maybe(json.loads), as_json=True)


@cache
def columns(kind: type[DataclassInstance]) -> tuple[Column, ...]:
    hints = get_type_hints(kind)
    return tuple(_column(f.name, hints[f.name]) for f in fields(kind))


def _quote(name: str) -> str:
    return f'"{name}"'


def schema() -> list[str]:
    statements = []
    for kind, table in TABLES.items():
        cols = [f"{_quote(c.name)} {c.sql_type}" for c in columns(kind)]
        if kind in UNIQUE:
            cols.append(f"UNIQUE ({_quote(UNIQUE[kind])})")
        statements.append(
            f"CREATE TABLE IF NOT EXISTS {table} (seq INTEGER PRIMARY KEY, {', '.join(cols)})"
        )
        if any(c.name == "session" for c in columns(kind)):
            statements.append(f"CREATE INDEX IF NOT EXISTS {table}_session ON {table} (session)")
    return statements


@cache
def _insert_sql(kind: type[DataclassInstance], verb: str) -> str:
    names = [_quote(c.name) for c in columns(kind)]
    marks = ", ".join("?" * len(names))
    return f"{verb} INTO {TABLES[kind]} ({', '.join(names)}) VALUES ({marks})"


def _row(event: DataclassInstance) -> list[object]:
    return [
        json.dumps(getattr(event, c.name), default=str) if c.as_json else getattr(event, c.name)
        for c in columns(type(event))
    ]


def jsonl_line(event: Event) -> str:
    return json.dumps({"type": type(event).__name__, **asdict(event)}, default=str) + "\n"


class Store:
    """The run store: `run.sqlite` (WAL) and `events.jsonl`, both written by one thread.

    `emit` only enqueues, so it is safe from any thread or coroutine and never blocks. `close`
    drains the queue, leaves `run.sqlite` as one self-contained file, and raises if any event
    could not be stored.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._queue: queue.SimpleQueue[Event | None] = queue.SimpleQueue()
        self._failures: list[str] = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="instar-store", daemon=True)
        self._thread.start()

    def emit(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("store is closed")
        self._queue.put(event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if self._failures:
            first = self._failures[0]
            raise RuntimeError(f"{len(self._failures)} events were not stored; first: {first}")

    def _run(self) -> None:
        try:
            db = sqlite3.connect(self.directory / DB_NAME)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                for statement in schema():
                    db.execute(statement)
                with (self.directory / JSONL_NAME).open("a", encoding="utf-8") as jsonl:
                    while not self._write_batch(db, jsonl):
                        pass
                db.execute("PRAGMA journal_mode=DELETE")
            finally:
                db.close()
        except Exception as exc:  # surfaced by close(), never lost silently
            self._failures.append(f"store writer stopped: {exc!r}")

    def _write_batch(self, db: sqlite3.Connection, jsonl: TextIO) -> bool:
        """Write everything queued so far; return True once close() has asked to stop."""
        batch = [self._queue.get()]
        while len(batch) < BATCH:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        events = [e for e in batch if e is not None]
        with db:
            for event in events:
                try:
                    self._write(event, db)
                except Exception as exc:  # keep storing the rest; close() reports it
                    self._failures.append(f"{type(event).__name__}: {exc!r}")
        jsonl.writelines(jsonl_line(e) for e in events)
        jsonl.flush()
        return len(events) < len(batch)

    @singledispatchmethod
    def _write(self, event: DataclassInstance, db: sqlite3.Connection) -> None:
        db.execute(_insert_sql(type(event), "INSERT"), _row(event))

    @_write.register(Run)
    def _(self, event: Run, db: sqlite3.Connection) -> None:
        db.execute(_insert_sql(Run, "INSERT OR REPLACE"), _row(event))


class StoreReader:
    """A read-only view of a run store, safe to open while the run is still writing."""

    def __init__(self, directory: Path) -> None:
        uri = f"{(directory / DB_NAME).resolve().as_uri()}?mode=ro"
        self.db = sqlite3.connect(uri, uri=True)
        self.db.row_factory = sqlite3.Row

    def select[E: DataclassInstance](self, kind: type[E], **where: object) -> list[E]:
        """Events of one type in the order they were written, filtered by field equality."""
        cols = columns(kind)
        if unknown := set(where) - {c.name for c in cols}:
            raise ValueError(f"{kind.__name__} has no field {', '.join(sorted(unknown))}")
        sql = f"SELECT * FROM {TABLES[kind]}"
        if where:
            sql += " WHERE " + " AND ".join(f"{_quote(name)} = ?" for name in where)
        rows = self.db.execute(f"{sql} ORDER BY seq", list(where.values()))
        return [kind(**{c.name: c.decode(row[c.name]) for c in cols}) for row in rows]

    def close(self) -> None:
        self.db.close()


class JsonlSink:
    """Appends each event to a JSONL file as one line, tagged with its type."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: Event) -> None:
        self._file.write(jsonl_line(event))

    def close(self) -> None:
        self._file.close()
