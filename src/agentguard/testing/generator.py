"""Auto-generate pytest test cases from recorded traces.

The TestGenerator reads :class:`~agentguard.core.types.TraceEntry` objects
from a trace directory and emits a ``.py`` file containing complete pytest
test functions — one per recorded tool call.

Generated tests:
- Call the tool with the recorded arguments.
- Assert the return type matches.
- Assert dict return values contain the same top-level keys.
- Assert list return values are non-empty (if the original was non-empty).
- Include a ``# Source trace: …`` comment for traceability.

Example::

    from agentguard.testing import TestGenerator

    gen = TestGenerator(traces_dir="./traces")
    gen.generate_tests(output="tests/test_generated.py")
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Optional

from agentguard.core.trace import TraceStore
from agentguard.core.types import ToolCallStatus, TraceEntry


class TestGenerator:  # noqa: N801
    """Generates pytest test files from recorded agentguard traces.

    .. note:: This class has an ``__init__`` constructor and will not be collected by pytest.
    
    :meta private:

    Example::

        generator = TestGenerator(traces_dir="./traces")
        test_code = generator.generate_tests(output="tests/test_generated.py")
        print(test_code)
    """

    # Minimum and maximum number of tests to generate
    MAX_TESTS_PER_TOOL: int = 10
    MAX_TOTAL_TESTS: int = 200

    def __init__(
        self,
        traces_dir: str = "./traces",
        *,
        backend: str | None = None,
        trace_db_path: str | None = None,
        include_failing: bool = False,
        max_per_tool: int = 10,
        import_prefix: str = "",
    ) -> None:
        """Initialise the generator.

        Args:
            traces_dir: Directory containing ``.jsonl`` trace files.
            include_failing: If True, also generate tests for trace entries
                that originally failed (as ``xfail`` tests).
            max_per_tool: Maximum tests to generate per tool name.
            import_prefix: Python import prefix for the module containing
                the tools (e.g. ``"from mypackage.tools import "``).
        """
        self.traces_dir = traces_dir
        self.include_failing = include_failing
        self.max_per_tool = max_per_tool
        self.import_prefix = import_prefix
        self._store = TraceStore(directory=traces_dir, backend=backend, db_path=trace_db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_tests(
        self,
        output: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Generate pytest test code from traces.

        Args:
            output: If provided, write the generated code to this file path.
            session_id: Restrict generation to a specific session.

        Returns:
            The generated Python source code as a string.
        """
        if session_id:
            entries = self._store.read_session(session_id)
        else:
            entries = self._store.read_all()

        if not entries:
            code = self._empty_module()
        else:
            code = self._build_module(entries)

        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")

        return code

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _build_module(self, entries: list[TraceEntry]) -> str:
        parts: list[str] = [self._module_header()]

        # Group entries by tool
        by_tool: dict[str, list[TraceEntry]] = {}
        for e in entries:
            by_tool.setdefault(e.tool_name, []).append(e)

        # Collect tool imports
        tool_names = sorted(by_tool.keys())
        if self.import_prefix:
            for name in tool_names:
                parts.append(f"from {self.import_prefix} import {name}")
            parts.append("")

        # Generate one test per entry
        total = 0
        for tool_name in tool_names:
            tool_entries = by_tool[tool_name][: self.max_per_tool]
            for i, entry in enumerate(tool_entries):
                if total >= self.MAX_TOTAL_TESTS:
                    break
                if entry.result.status not in (ToolCallStatus.SUCCESS, ToolCallStatus.RETRIED):
                    if not self.include_failing:
                        continue
                    parts.append(self._generate_xfail_test(entry, i))
                else:
                    parts.append(self._generate_test(entry, i))
                total += 1

        return "\n".join(parts)

    def _generate_test(self, entry: TraceEntry, idx: int) -> str:
        """Generate a single passing pytest test function."""
        fn_name = _safe_identifier(entry.tool_name)
        args_repr = _repr_args(entry.call.args)
        kwargs_repr = _repr_kwargs(entry.call.kwargs)
        call_repr = _build_call(entry.tool_name, args_repr, kwargs_repr)
        return_value = entry.result.return_value
        assertions = _build_assertions(return_value)
        timestamp = entry.call.timestamp.isoformat(timespec="seconds")

        lines = [
            f"def test_{fn_name}_{idx:03d}():",
            f'    """Auto-generated from trace {timestamp} (call_id={entry.call_id[:8]})."""',
        ]
        if entry.call.args or entry.call.kwargs:
            lines.append(f"    result = {call_repr}")
        else:
            lines.append(f"    result = {entry.tool_name}()")
        for assertion in assertions:
            lines.append(f"    {assertion}")
        lines.append("")

        return textwrap.indent("\n".join(lines), "")

    def _generate_xfail_test(self, entry: TraceEntry, idx: int) -> str:
        """Generate an xfail test for a previously failing call."""
        fn_name = _safe_identifier(entry.tool_name)
        args_repr = _repr_args(entry.call.args)
        kwargs_repr = _repr_kwargs(entry.call.kwargs)
        call_repr = _build_call(entry.tool_name, args_repr, kwargs_repr)
        timestamp = entry.call.timestamp.isoformat(timespec="seconds")
        reason = entry.result.exception or entry.result.status.value

        return (
            f"@pytest.mark.xfail(reason={reason!r})\n"
            f"def test_{fn_name}_{idx:03d}_failing():\n"
            f'    """Auto-generated xfail from trace {timestamp}."""\n'
            f"    {call_repr}\n"
        )

    def _module_header(self) -> str:
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        return (
            f'"""Auto-generated test suite from agentguard traces.\n'
            f"\nGenerated at: {now} UTC\n"
            f'Do not edit manually — regenerate with TestGenerator.\n"""\n'
            f"from __future__ import annotations\n\n"
            f"import pytest\n\n"
        )

    def _empty_module(self) -> str:
        return (
            '"""Auto-generated test suite — no traces found."""\n\n'
            "import pytest\n\n\n"
            "def test_no_traces():\n"
            '    """Placeholder: no traces were recorded yet."""\n'
            "    pytest.skip(\"No traces recorded yet\")\n"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_identifier(name: str) -> str:
    """Convert an arbitrary string to a safe Python identifier."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "tool_" + safe
    return safe or "tool"


def _repr_args(args: tuple[Any, ...]) -> str:
    if not args:
        return ""
    parts = []
    for a in args:
        try:
            r = repr(a)
            # Truncate very long args
            if len(r) > 200:
                r = r[:197] + "..."
            parts.append(r)
        except Exception:
            parts.append("...")
    return ", ".join(parts)


def _repr_kwargs(kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return ""
    parts = []
    for k, v in kwargs.items():
        try:
            r = repr(v)
            if len(r) > 200:
                r = r[:197] + "..."
            parts.append(f"{k}={r}")
        except Exception:
            parts.append(f"{k}=...")
    return ", ".join(parts)


def _build_call(tool_name: str, args_repr: str, kwargs_repr: str) -> str:
    all_args = ", ".join(filter(None, [args_repr, kwargs_repr]))
    return f"{tool_name}({all_args})"


def _build_assertions(return_value: Any) -> list[str]:
    """Build a list of assertion strings for the given return value."""
    if return_value is None:
        return ["assert result is None"]

    assertions: list[str] = []

    if isinstance(return_value, dict):
        assertions.append("assert isinstance(result, dict)")
        for key in list(return_value.keys())[:5]:  # limit to 5 keys
            assertions.append(f"assert {key!r} in result")

    elif isinstance(return_value, list):
        assertions.append("assert isinstance(result, list)")
        if return_value:
            assertions.append("assert len(result) > 0")

    elif isinstance(return_value, str):
        assertions.append("assert isinstance(result, str)")
        if return_value:
            assertions.append("assert len(result) > 0")

    elif isinstance(return_value, (int, float)):
        assertions.append(f"assert isinstance(result, ({type(return_value).__name__},))")

    elif isinstance(return_value, bool):
        assertions.append(f"assert result is {return_value}")

    else:
        assertions.append(f"assert result is not None")

    return assertions or ["assert result is not None"]
