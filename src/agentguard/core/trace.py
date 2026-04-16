"""Execution trace capture, storage backends, and session recording helpers.

agentguard supports two first-party persistence backends:

- ``SQLiteTraceStore`` for durable, queryable production storage.
- ``JsonlTraceStore`` for legacy compatibility and simple file export flows.

The ``TraceStore`` facade preserves the historic public API while routing to
the selected backend.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Iterable, Optional

from agentguard.core.types import ToolCallStatus, TraceEntry

TRACE_BACKEND_SQLITE = "sqlite"
TRACE_BACKEND_JSONL = "jsonl"
DEFAULT_SQLITE_FILENAME = "agentguard_traces.db"


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _normalise_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _isoformat(value: datetime) -> str:
    return _normalise_dt(value).isoformat()


def _parse_timestamp(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return _utcnow()


def _resolve_sqlite_path(storage: str | Path) -> Path:
    path = Path(storage)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / DEFAULT_SQLITE_FILENAME


class BaseTraceStore(ABC):
    """Abstract trace store API shared by all backends."""

    backend: str

    @abstractmethod
    def write(self, entry: TraceEntry, session_id: Optional[str] = None) -> Path:
        """Persist a single trace entry."""

    @abstractmethod
    def read_session(self, session_id: str) -> list[TraceEntry]:
        """Read all entries for one session."""

    @abstractmethod
    def read_all(self) -> list[TraceEntry]:
        """Read every stored entry."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all known session ids."""

    @abstractmethod
    def filter(
        self,
        *,
        tool_name: Optional[str] = None,
        status: Optional[ToolCallStatus] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> list[TraceEntry]:
        """Filter stored entries."""

    @abstractmethod
    def stats(self, session_id: Optional[str] = None) -> dict[str, object]:
        """Summarise stored entries."""

    def load_session(self, session_id: str) -> list[TraceEntry]:
        """Backward-compatible alias for :meth:`read_session`."""
        return self.read_session(session_id)

    def load_all(self) -> list[TraceEntry]:
        """Backward-compatible alias for :meth:`read_all`."""
        return self.read_all()

    def export_jsonl(
        self,
        output_dir: str,
        *,
        session_id: Optional[str] = None,
    ) -> list[Path]:
        """Export entries to JSONL files grouped by session."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        entries = self.read_session(session_id) if session_id else self.read_all()
        grouped: dict[str, list[TraceEntry]] = {}
        for entry in entries:
            sid = entry.call.session_id or "default"
            grouped.setdefault(sid, []).append(entry)

        written: list[Path] = []
        for sid, session_entries in grouped.items():
            path = target / f"{sid}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for entry in session_entries:
                    fh.write(entry.model_dump_json() + "\n")
            written.append(path)
        return written

    def import_jsonl(
        self,
        source_dir: str,
        *,
        dedupe: bool = True,
    ) -> int:
        """Import JSONL traces from *source_dir* into this store."""
        source_store = JsonlTraceStore(directory=source_dir)
        count = 0
        existing_ids: set[str] = set()
        if dedupe:
            existing_ids = {entry.call_id for entry in self.read_all()}
        for entry in source_store.read_all():
            if dedupe and entry.call_id in existing_ids:
                continue
            self.write(entry, session_id=entry.call.session_id)
            count += 1
        return count

    def session_summaries(self) -> list[dict[str, Any]]:
        """Return session-level summary rows used by CLI and dashboard."""
        summaries: list[dict[str, Any]] = []
        for session_id in self.list_sessions():
            entries = self.read_session(session_id)
            if not entries:
                continue
            entries.sort(key=lambda e: _normalise_dt(e.call.timestamp))
            start = entries[0].call.timestamp
            end = entries[-1].result.timestamp
            failures = sum(1 for e in entries if e.result.failed)
            total_cost = sum(e.result.cost or 0.0 for e in entries if e.result.cost is not None)
            summaries.append(
                {
                    "session_id": session_id,
                    "calls": len(entries),
                    "failures": failures,
                    "started_at": _isoformat(start),
                    "ended_at": _isoformat(end),
                    "duration_ms": max(
                        (_normalise_dt(end) - _normalise_dt(start)).total_seconds() * 1000,
                        0.0,
                    ),
                    "total_cost_usd": round(total_cost, 6) if total_cost else None,
                    "tool_names": sorted({entry.tool_name for entry in entries}),
                }
            )
        summaries.sort(key=lambda item: item["started_at"], reverse=True)
        return summaries


class JsonlTraceStore(BaseTraceStore):
    """Thread-safe append-only trace store backed by JSONL files."""

    backend = TRACE_BACKEND_JSONL

    def __init__(self, directory: str = "./traces") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, entry: TraceEntry, session_id: Optional[str] = None) -> Path:
        sid = session_id or entry.call.session_id or _utcnow().strftime("%Y%m%d")
        path = self.directory / f"{sid}.jsonl"
        line = entry.model_dump_json() + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        return path

    def read_session(self, session_id: str) -> list[TraceEntry]:
        return self._load_file(self.directory / f"{session_id}.jsonl")

    def read_all(self) -> list[TraceEntry]:
        entries: list[TraceEntry] = []
        for path in sorted(self.directory.glob("*.jsonl")):
            entries.extend(self._load_file(path))
        entries.sort(key=lambda e: _normalise_dt(e.call.timestamp))
        return entries

    def list_sessions(self) -> list[str]:
        return sorted(path.stem for path in self.directory.glob("*.jsonl"))

    def filter(
        self,
        *,
        tool_name: Optional[str] = None,
        status: Optional[ToolCallStatus] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> list[TraceEntry]:
        entries = self.read_session(session_id) if session_id else self.read_all()
        filtered: list[TraceEntry] = []
        for entry in entries:
            timestamp = _normalise_dt(entry.call.timestamp)
            if tool_name and entry.tool_name != tool_name:
                continue
            if status and entry.result.status != status:
                continue
            if since and timestamp < _normalise_dt(since):
                continue
            if until and timestamp > _normalise_dt(until):
                continue
            filtered.append(entry)
        return filtered

    def stats(self, session_id: Optional[str] = None) -> dict[str, object]:
        return _stats_from_entries(self.read_session(session_id) if session_id else self.read_all())

    @staticmethod
    def _load_file(path: Path) -> list[TraceEntry]:
        if not path.exists():
            return []
        entries: list[TraceEntry] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entries.append(TraceEntry.model_validate(json.loads(raw)))
                except (json.JSONDecodeError, Exception):
                    continue
        return entries


class SQLiteTraceStore(BaseTraceStore):
    """Durable SQLite-backed trace store for production observability."""

    backend = TRACE_BACKEND_SQLITE

    def __init__(self, db_path: str = "./traces/agentguard_traces.db") -> None:
        self.db_path = _resolve_sqlite_path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    call_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS trace_entries (
                    call_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    call_timestamp TEXT NOT NULL,
                    result_timestamp TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    retry_count INTEGER NOT NULL,
                    cost REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trace_entries_session_id
                    ON trace_entries(session_id);
                CREATE INDEX IF NOT EXISTS idx_trace_entries_tool_name
                    ON trace_entries(tool_name);
                CREATE INDEX IF NOT EXISTS idx_trace_entries_status
                    ON trace_entries(status);
                CREATE INDEX IF NOT EXISTS idx_trace_entries_call_timestamp
                    ON trace_entries(call_timestamp);
                CREATE INDEX IF NOT EXISTS idx_trace_entries_result_timestamp
                    ON trace_entries(result_timestamp);
                """
            )

    def write(self, entry: TraceEntry, session_id: Optional[str] = None) -> Path:
        sid = session_id or entry.call.session_id or "default"
        payload = entry.model_dump_json()
        call_ts = _isoformat(entry.call.timestamp)
        result_ts = _isoformat(entry.result.timestamp)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sessions(session_id, started_at, last_seen_at, call_count)
                    VALUES (?, ?, ?, 0)
                    """,
                    (sid, call_ts, result_ts),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trace_entries(
                        call_id, session_id, tool_name, status, call_timestamp,
                        result_timestamp, execution_time_ms, retry_count, cost, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.call_id,
                        sid,
                        entry.tool_name,
                        entry.result.status.value,
                        call_ts,
                        result_ts,
                        entry.result.execution_time_ms,
                        entry.result.retry_count,
                        entry.result.cost,
                        payload,
                    ),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET started_at = MIN(started_at, ?),
                        last_seen_at = MAX(last_seen_at, ?),
                        call_count = (
                            SELECT COUNT(*) FROM trace_entries WHERE session_id = ?
                        )
                    WHERE session_id = ?
                    """,
                    (call_ts, result_ts, sid, sid),
                )
        return self.db_path

    def read_session(self, session_id: str) -> list[TraceEntry]:
        return self._query_entries("WHERE session_id = ? ORDER BY call_timestamp ASC", (session_id,))

    def read_all(self) -> list[TraceEntry]:
        return self._query_entries("ORDER BY call_timestamp ASC", ())

    def list_sessions(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def filter(
        self,
        *,
        tool_name: Optional[str] = None,
        status: Optional[ToolCallStatus] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> list[TraceEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if since:
            clauses.append("call_timestamp >= ?")
            params.append(_isoformat(since))
        if until:
            clauses.append("call_timestamp <= ?")
            params.append(_isoformat(until))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query_entries(f"{where} ORDER BY call_timestamp ASC", tuple(params))

    def stats(self, session_id: Optional[str] = None) -> dict[str, object]:
        entries = self.read_session(session_id) if session_id else self.read_all()
        return _stats_from_entries(entries)

    def session_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.started_at,
                    s.last_seen_at,
                    s.call_count,
                    SUM(CASE WHEN t.status NOT IN ('success', 'retried') THEN 1 ELSE 0 END) AS failures,
                    SUM(t.cost) AS total_cost
                FROM sessions s
                LEFT JOIN trace_entries t ON t.session_id = s.session_id
                GROUP BY s.session_id, s.started_at, s.last_seen_at, s.call_count
                ORDER BY s.started_at DESC
                """
            ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            started = _parse_timestamp(str(row["started_at"]))
            ended = _parse_timestamp(str(row["last_seen_at"]))
            summaries.append(
                {
                    "session_id": str(row["session_id"]),
                    "calls": int(row["call_count"] or 0),
                    "failures": int(row["failures"] or 0),
                    "started_at": str(row["started_at"]),
                    "ended_at": str(row["last_seen_at"]),
                    "duration_ms": max(
                        (_normalise_dt(ended) - _normalise_dt(started)).total_seconds() * 1000,
                        0.0,
                    ),
                    "total_cost_usd": (
                        round(float(row["total_cost"]), 6) if row["total_cost"] is not None else None
                    ),
                    "tool_names": [],
                }
            )
        return summaries

    def _query_entries(self, clause: str, params: tuple[Any, ...]) -> list[TraceEntry]:
        query = f"SELECT payload_json FROM trace_entries {clause}"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        entries: list[TraceEntry] = []
        for row in rows:
            try:
                entries.append(TraceEntry.model_validate(json.loads(str(row["payload_json"]))))
            except Exception:
                continue
        return entries


def _stats_from_entries(entries: Iterable[TraceEntry]) -> dict[str, object]:
    rows = list(entries)
    if not rows:
        return {"total_calls": 0}

    total = len(rows)
    successes = sum(1 for entry in rows if entry.result.status == ToolCallStatus.SUCCESS)
    failures = sum(1 for entry in rows if entry.result.failed)
    latencies = [entry.result.execution_time_ms for entry in rows]
    costs = [entry.result.cost for entry in rows if entry.result.cost is not None]
    tool_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    retries = 0
    for entry in rows:
        tool_counts[entry.tool_name] = tool_counts.get(entry.tool_name, 0) + 1
        status = entry.result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        retries += entry.result.retry_count

    return {
        "total_calls": total,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / total,
        "avg_latency_ms": sum(latencies) / total,
        "max_latency_ms": max(latencies),
        "min_latency_ms": min(latencies),
        "total_cost_usd": sum(costs) if costs else None,
        "calls_per_tool": tool_counts,
        "status_counts": status_counts,
        "total_retries": retries,
    }


def create_trace_store(
    *,
    backend: Optional[str] = None,
    directory: Optional[str] = None,
    db_path: Optional[str] = None,
) -> BaseTraceStore:
    """Create a concrete trace store from backend hints."""
    resolved = backend
    if resolved is None:
        if db_path is not None:
            resolved = TRACE_BACKEND_SQLITE
        elif directory is not None:
            candidate = Path(directory)
            if candidate.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                resolved = TRACE_BACKEND_SQLITE
            elif (candidate / DEFAULT_SQLITE_FILENAME).exists():
                resolved = TRACE_BACKEND_SQLITE
            else:
                resolved = TRACE_BACKEND_JSONL
        else:
            resolved = TRACE_BACKEND_SQLITE

    if resolved == TRACE_BACKEND_JSONL:
        return JsonlTraceStore(directory=directory or "./traces")
    if resolved == TRACE_BACKEND_SQLITE:
        return SQLiteTraceStore(db_path=db_path or str(_resolve_sqlite_path(directory or "./traces")))
    raise ValueError(f"Unsupported trace backend: {resolved!r}")


class TraceStore(BaseTraceStore):
    """Facade preserving the historic TraceStore API while selecting a backend."""

    def __init__(
        self,
        directory: str = "./traces",
        *,
        backend: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self._store = create_trace_store(backend=backend, directory=directory, db_path=db_path)
        self.backend = self._store.backend

    @property
    def store(self) -> BaseTraceStore:
        """Expose the concrete backend instance."""
        return self._store

    def write(self, entry: TraceEntry, session_id: Optional[str] = None) -> Path:
        return self._store.write(entry, session_id=session_id)

    def read_session(self, session_id: str) -> list[TraceEntry]:
        return self._store.read_session(session_id)

    def read_all(self) -> list[TraceEntry]:
        return self._store.read_all()

    def list_sessions(self) -> list[str]:
        return self._store.list_sessions()

    def filter(
        self,
        *,
        tool_name: Optional[str] = None,
        status: Optional[ToolCallStatus] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> list[TraceEntry]:
        return self._store.filter(
            tool_name=tool_name,
            status=status,
            since=since,
            until=until,
            session_id=session_id,
        )

    def stats(self, session_id: Optional[str] = None) -> dict[str, object]:
        return self._store.stats(session_id)

    def load_session(self, session_id: str) -> list[TraceEntry]:
        return self._store.load_session(session_id)

    def load_all(self) -> list[TraceEntry]:
        return self._store.load_all()

    def export_jsonl(
        self,
        output_dir: str,
        *,
        session_id: Optional[str] = None,
    ) -> list[Path]:
        return self._store.export_jsonl(output_dir, session_id=session_id)

    def import_jsonl(
        self,
        source_dir: str,
        *,
        dedupe: bool = True,
    ) -> int:
        return self._store.import_jsonl(source_dir, dedupe=dedupe)

    def session_summaries(self) -> list[dict[str, Any]]:
        return self._store.session_summaries()


class TraceRecorder:
    """Context manager and start/stop API for recording tool call traces."""

    def __init__(
        self,
        storage: str = "./traces",
        session_id: Optional[str] = None,
        *,
        backend: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.backend = backend or TRACE_BACKEND_SQLITE
        self.db_path = db_path
        self.session_id = session_id or str(uuid.uuid4())
        self._store = TraceStore(directory=storage, backend=self.backend, db_path=db_path)
        self._active = False
        self._local = threading.local()

    def __enter__(self) -> "TraceRecorder":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        self._active = True
        _register_active_recorder(self)

    def stop(self) -> None:
        self._active = False
        _unregister_active_recorder(self)

    @property
    def is_active(self) -> bool:
        return self._active

    def record(self, entry: TraceEntry) -> Path:
        return self._store.write(entry, session_id=self.session_id)

    def entries(self) -> list[TraceEntry]:
        return self._store.read_session(self.session_id)

    def stats(self) -> dict[str, object]:
        return self._store.stats(self.session_id)

    @property
    def store(self) -> TraceStore:
        return self._store


_active_recorders: list[TraceRecorder] = []
_active_recorders_lock = threading.Lock()


def _register_active_recorder(recorder: TraceRecorder) -> None:
    with _active_recorders_lock:
        _active_recorders.append(recorder)


def _unregister_active_recorder(recorder: TraceRecorder) -> None:
    with _active_recorders_lock:
        try:
            _active_recorders.remove(recorder)
        except ValueError:
            pass


def get_active_recorders() -> list[TraceRecorder]:
    """Return all currently active TraceRecorder instances."""
    with _active_recorders_lock:
        return list(_active_recorders)


@contextmanager
def record_session(
    storage: str = "./traces",
    session_id: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Generator[TraceRecorder, None, None]:
    """Convenience context manager for one-off recording sessions."""
    recorder = TraceRecorder(storage=storage, session_id=session_id, backend=backend, db_path=db_path)
    with recorder:
        yield recorder


@asynccontextmanager
async def async_record_session(
    storage: str = "./traces",
    session_id: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    db_path: Optional[str] = None,
) -> AsyncGenerator[TraceRecorder, None]:
    """Async-compatible context manager for one-off recording sessions."""
    recorder = TraceRecorder(storage=storage, session_id=session_id, backend=backend, db_path=db_path)
    recorder.start()
    try:
        yield recorder
    finally:
        recorder.stop()
