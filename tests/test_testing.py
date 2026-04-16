"""Tests for the testing module: recorder, replayer, generator, assertions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentguard import guard
from agentguard.core.registry import global_registry
from agentguard.core.types import ToolCallStatus
from agentguard.testing import (
    TestGenerator,
    TraceRecorder,
    TraceReplayer,
    assert_all_succeeded,
    assert_latency_budget,
    assert_no_hallucinations,
    assert_tool_call,
    record_session,
)
from agentguard.core.trace import TraceStore
from agentguard.core.types import ToolCall, ToolResult, TraceEntry


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    global_registry.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    tool_name: str = "test_tool",
    status: ToolCallStatus = ToolCallStatus.SUCCESS,
    return_value: object = None,
    execution_time_ms: float = 100.0,
    retry_count: int = 0,
) -> TraceEntry:
    call = ToolCall(tool_name=tool_name, args=("arg1",), kwargs={"key": "val"})
    result = ToolResult(
        call_id=call.call_id,
        tool_name=tool_name,
        status=status,
        return_value=return_value,
        execution_time_ms=execution_time_ms,
        retry_count=retry_count,
    )
    return TraceEntry(call=call, result=result)


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------


class TestTraceStore:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            entry = make_entry()
            store.write(entry, session_id="s1")
            entries = store.read_session("s1")
            assert len(entries) == 1
            assert entries[0].tool_name == "test_tool"

    def test_read_all_across_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            for session in ["s1", "s2", "s3"]:
                store.write(make_entry(tool_name=f"tool_{session}"), session_id=session)
            all_entries = store.read_all()
            assert len(all_entries) == 3

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            store.write(make_entry(), session_id="session_a")
            store.write(make_entry(), session_id="session_b")
            sessions = store.list_sessions()
            assert "session_a" in sessions
            assert "session_b" in sessions

    def test_filter_by_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            store.write(make_entry(tool_name="search"), session_id="s")
            store.write(make_entry(tool_name="db_query"), session_id="s")
            results = store.filter(tool_name="search")
            assert all(e.tool_name == "search" for e in results)

    def test_filter_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            store.write(make_entry(status=ToolCallStatus.SUCCESS), session_id="s")
            store.write(make_entry(status=ToolCallStatus.FAILURE), session_id="s")
            results = store.filter(status=ToolCallStatus.FAILURE)
            assert all(e.result.status == ToolCallStatus.FAILURE for e in results)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            for i in range(5):
                store.write(make_entry(execution_time_ms=float(i * 10 + 10)), session_id="s")
            stats = store.stats("s")
            assert stats["total_calls"] == 5
            assert "avg_latency_ms" in stats


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


class TestTraceRecorder:
    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            @guard(record=True, trace_dir=tmpdir)
            def tool(x: str) -> str:
                return x

            with TraceRecorder(storage=tmpdir) as rec:
                tool("hello")

            entries = rec.entries()
            assert len(entries) >= 1

    def test_record_session_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            @guard(record=True, trace_dir=tmpdir)
            def tool2(x: str) -> str:
                return x

            with record_session(tmpdir) as rec:
                tool2("world")
            assert len(rec.entries()) >= 1

    def test_is_active_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = TraceRecorder(storage=tmpdir)
            assert not rec.is_active
            rec.start()
            assert rec.is_active
            rec.stop()
            assert not rec.is_active

    def test_recorded_traces_redact_secret_like_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            @guard(record=True, trace_dir=tmpdir)
            def tool(api_key: str, query: str) -> str:
                return query.upper()

            tool("sk-live-super-secret", "hello")

            store = TraceStore(directory=tmpdir)
            entries = store.read_all()
            assert len(entries) == 1
            assert entries[0].call.args[0] == "[REDACTED]"
            assert entries[0].call.args[1] == "hello"


# ---------------------------------------------------------------------------
# TraceReplayer
# ---------------------------------------------------------------------------


class TestTraceReplayer:
    def test_replay_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            entry = make_entry(tool_name="echo", return_value={"msg": "hi"})
            store.write(entry, session_id="s1")

            def echo_impl(*args: object, key: str = "") -> dict:
                return {"msg": key}

            replayer = TraceReplayer(
                traces_dir=tmpdir,
                tool_registry={"echo": echo_impl},
            )
            report = replayer.replay_session("s1")
            assert report.total == 1
            assert report.skipped == 0

    def test_skips_failed_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            entry = make_entry(tool_name="broken", status=ToolCallStatus.FAILURE)
            store.write(entry, session_id="s")

            replayer = TraceReplayer(traces_dir=tmpdir)
            report = replayer.replay_session("s")
            assert report.skipped == 1


# ---------------------------------------------------------------------------
# TestGenerator
# ---------------------------------------------------------------------------


class TestTestGenerator:
    def test_generates_test_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(directory=tmpdir)
            entry = make_entry(
                tool_name="my_tool",
                return_value={"status": 200, "data": [1, 2, 3]},
            )
            store.write(entry, session_id="s")

            gen = TestGenerator(traces_dir=tmpdir)
            out = Path(tmpdir) / "test_gen.py"
            code = gen.generate_tests(output=str(out))

            assert "def test_my_tool" in code
            assert "assert isinstance(result, dict)" in code
            assert out.exists()

    def test_empty_traces_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TestGenerator(traces_dir=tmpdir)
            code = gen.generate_tests()
            assert "pytest.skip" in code


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class TestAssertions:
    def test_succeeded_passes(self):
        entry = make_entry(status=ToolCallStatus.SUCCESS, return_value={"a": 1})
        assert_tool_call(entry).succeeded()

    def test_succeeded_fails_on_failure(self):
        entry = make_entry(status=ToolCallStatus.FAILURE)
        with pytest.raises(AssertionError):
            assert_tool_call(entry).succeeded()

    def test_returned_dict(self):
        entry = make_entry(return_value={"key": "val"})
        assert_tool_call(entry).returned_dict()

    def test_has_keys(self):
        entry = make_entry(return_value={"a": 1, "b": 2})
        assert_tool_call(entry).has_keys("a", "b")

    def test_has_keys_fails(self):
        entry = make_entry(return_value={"a": 1})
        with pytest.raises(AssertionError):
            assert_tool_call(entry).has_keys("missing_key")

    def test_executed_within_ms(self):
        entry = make_entry(execution_time_ms=50.0)
        assert_tool_call(entry).executed_within_ms(100.0)

    def test_executed_within_ms_fails(self):
        entry = make_entry(execution_time_ms=200.0)
        with pytest.raises(AssertionError):
            assert_tool_call(entry).executed_within_ms(100.0)

    def test_chaining(self):
        entry = make_entry(
            status=ToolCallStatus.SUCCESS,
            return_value={"status": 200, "data": []},
            execution_time_ms=50.0,
        )
        (
            assert_tool_call(entry)
            .succeeded()
            .executed_within_ms(100.0)
            .returned_dict()
            .has_keys("status", "data")
        )

    def test_assert_all_succeeded(self):
        entries = [make_entry(status=ToolCallStatus.SUCCESS) for _ in range(3)]
        assert_all_succeeded(entries)

    def test_assert_all_succeeded_fails(self):
        entries = [
            make_entry(status=ToolCallStatus.SUCCESS),
            make_entry(status=ToolCallStatus.FAILURE),
        ]
        with pytest.raises(AssertionError):
            assert_all_succeeded(entries)

    def test_assert_latency_budget(self):
        entries = [make_entry(execution_time_ms=50.0) for _ in range(3)]
        assert_latency_budget(entries, max_ms=100.0)

    def test_field_equals(self):
        entry = make_entry(return_value={"code": 42})
        assert_tool_call(entry).field_equals("code", 42)

    def test_field_equals_fails(self):
        entry = make_entry(return_value={"code": 42})
        with pytest.raises(AssertionError):
            assert_tool_call(entry).field_equals("code", 99)

    def test_returned_non_empty(self):
        entry = make_entry(return_value=[1, 2, 3])
        assert_tool_call(entry).returned_non_empty()

    def test_returned_non_empty_fails(self):
        entry = make_entry(return_value=[])
        with pytest.raises(AssertionError):
            assert_tool_call(entry).returned_non_empty()

    def test_was_retried(self):
        entry = make_entry(retry_count=2)
        assert_tool_call(entry).was_retried(times=2)

    def test_was_not_retried(self):
        entry = make_entry(retry_count=0)
        assert_tool_call(entry).was_not_retried()
