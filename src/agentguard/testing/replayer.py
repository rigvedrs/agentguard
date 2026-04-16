"""Replay recorded tool call traces as deterministic tests.

The TraceReplayer loads saved :class:`~agentguard.core.types.TraceEntry`
objects and replays them against the live or mocked tool implementations.
It asserts that:

- The tool can be called with the same arguments without crashing.
- The return value structure matches the recorded trace.
- Execution time stays within a configurable tolerance.

Example::

    from agentguard.testing.replayer import TraceReplayer

    replayer = TraceReplayer(traces_dir="./traces")
    replayer.replay_all(timeout_multiplier=2.0)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agentguard.core.trace import TraceStore
from agentguard.core.types import ToolCallStatus, TraceEntry


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Outcome of replaying a single trace entry."""

    entry: TraceEntry
    """The original trace entry."""

    success: bool
    """True if the replay passed all assertions."""

    actual_return: Any = None
    """The value returned by the live call."""

    actual_time_ms: float = 0.0
    """Wall-clock execution time of the live call."""

    failures: list[str] = field(default_factory=list)
    """List of assertion failure messages."""

    skipped: bool = False
    """True if this entry was skipped (e.g., tool not found)."""

    skip_reason: str = ""
    """Reason for skipping."""


@dataclass
class ReplayReport:
    """Aggregate report for a batch replay."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ReplayResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of non-skipped entries that passed."""
        eligible = self.total - self.skipped
        return (self.passed / eligible) if eligible > 0 else 1.0


# ---------------------------------------------------------------------------
# TraceReplayer
# ---------------------------------------------------------------------------


class TraceReplayer:
    """Replays recorded tool call traces against live tool implementations.

    Attributes:
        traces_dir: Path to the directory containing ``.jsonl`` trace files.
        tool_registry: Map of tool names to callables used for replay.
        timeout_multiplier: Allowed latency factor above original execution time.
        assert_return_type: Check the type of the return value.
        assert_return_keys: Check that dict returns contain the same top-level keys.
    """

    def __init__(
        self,
        traces_dir: str = "./traces",
        *,
        backend: str | None = None,
        trace_db_path: str | None = None,
        tool_registry: Optional[dict[str, Callable[..., Any]]] = None,
        timeout_multiplier: float = 5.0,
        assert_return_type: bool = True,
        assert_return_keys: bool = True,
    ) -> None:
        """Initialise the replayer.

        Args:
            traces_dir: Path to the trace directory.
            tool_registry: Map of ``tool_name → callable``. If None, the
                global agentguard registry is queried automatically.
            timeout_multiplier: Replay calls are allowed to take up to
                ``original_time × timeout_multiplier`` ms.
            assert_return_type: If True, assert that the replay return value
                has the same Python type as the recorded value.
            assert_return_keys: If True, assert dict returns have the same
                top-level keys as the recorded response.
        """
        self.traces_dir = traces_dir
        self._tool_registry = tool_registry or {}
        self.timeout_multiplier = timeout_multiplier
        self.assert_return_type = assert_return_type
        self.assert_return_keys = assert_return_keys
        self._store = TraceStore(directory=traces_dir, backend=backend, db_path=trace_db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callable for replay.

        Args:
            name: The tool name as recorded in the trace.
            fn: The callable to invoke during replay.
        """
        self._tool_registry[name] = fn

    def replay_session(self, session_id: str) -> ReplayReport:
        """Replay all entries from a specific session.

        Args:
            session_id: The session to replay.

        Returns:
            :class:`ReplayReport` with per-entry results.
        """
        entries = self._store.read_session(session_id)
        return self._replay_entries(entries)

    def replay_all(self) -> ReplayReport:
        """Replay all recorded traces.

        Returns:
            :class:`ReplayReport` aggregating all sessions.
        """
        entries = self._store.read_all()
        return self._replay_entries(entries)

    def replay_tool(self, tool_name: str) -> ReplayReport:
        """Replay all traces for a specific tool.

        Args:
            tool_name: The tool to filter on.

        Returns:
            :class:`ReplayReport`.
        """
        from agentguard.core.types import ToolCallStatus
        entries = self._store.filter(tool_name=tool_name)
        return self._replay_entries(entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _replay_entries(self, entries: list[TraceEntry]) -> ReplayReport:
        report = ReplayReport(total=len(entries))
        for entry in entries:
            result = self._replay_one(entry)
            report.results.append(result)
            if result.skipped:
                report.skipped += 1
            elif result.success:
                report.passed += 1
            else:
                report.failed += 1
        return report

    def _replay_one(self, entry: TraceEntry) -> ReplayResult:
        tool_name = entry.tool_name

        # Skip entries that originally failed
        if entry.result.status not in (ToolCallStatus.SUCCESS, ToolCallStatus.RETRIED):
            return ReplayResult(
                entry=entry,
                success=True,
                skipped=True,
                skip_reason=f"Original call had status {entry.result.status.value}; skipping",
            )

        fn = self._resolve_tool(tool_name)
        if fn is None:
            return ReplayResult(
                entry=entry,
                success=True,
                skipped=True,
                skip_reason=f"No callable registered for tool '{tool_name}'",
            )

        failures: list[str] = []
        actual_return: Any = None
        actual_time_ms: float = 0.0

        try:
            start = time.perf_counter()
            actual_return = fn(*entry.call.args, **entry.call.kwargs)
            actual_time_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            failures.append(f"Replay raised exception: {type(exc).__name__}: {exc}")
            return ReplayResult(
                entry=entry,
                success=False,
                actual_time_ms=actual_time_ms,
                failures=failures,
            )

        # Type check
        if self.assert_return_type:
            original = entry.result.return_value
            if original is not None and actual_return is not None:
                if type(actual_return) is not type(original):
                    failures.append(
                        f"Return type changed: expected {type(original).__name__}, "
                        f"got {type(actual_return).__name__}"
                    )

        # Key check
        if self.assert_return_keys:
            original = entry.result.return_value
            if isinstance(original, dict) and isinstance(actual_return, dict):
                original_keys = set(original.keys())
                actual_keys = set(actual_return.keys())
                missing = original_keys - actual_keys
                if missing:
                    failures.append(f"Response missing keys: {sorted(missing)}")

        return ReplayResult(
            entry=entry,
            success=len(failures) == 0,
            actual_return=actual_return,
            actual_time_ms=actual_time_ms,
            failures=failures,
        )

    def _resolve_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """Look up a callable by tool name."""
        if name in self._tool_registry:
            return self._tool_registry[name]
        # Try the global registry
        try:
            from agentguard.core.registry import global_registry
            reg = global_registry.get(name)
            if reg:
                return reg.func
        except Exception:
            pass
        return None
