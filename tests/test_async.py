"""Tests for async context manager support: GuardedTool.session() and async_record_session()."""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from agentguard import async_record_session, guard
from agentguard.core.guard import GuardedTool
from agentguard.core.registry import global_registry
from agentguard.core.types import ToolCallStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global registry between tests."""
    yield
    global_registry.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_async_tool(fail: bool = False, value: object = None):
    """Return a simple async guarded tool."""

    @guard
    async def async_tool(x: str) -> dict:
        """A simple async test tool."""
        if fail:
            raise ValueError("simulated async failure")
        return {"result": x, "echo": value}

    return async_tool


# ---------------------------------------------------------------------------
# GuardedTool.session() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_returns_same_tool():
    """session().__aenter__ must return the GuardedTool itself."""
    tool = make_async_tool()
    async with tool.session(session_id="s1") as t:
        assert t is tool


@pytest.mark.asyncio
async def test_session_pins_session_id():
    """While inside the block the tool's config.session_id is set to the given id."""
    tool = make_async_tool()
    assert tool.config.session_id is None
    async with tool.session(session_id="my-session") as t:
        assert t.config.session_id == "my-session"
    # Restored after exit
    assert tool.config.session_id is None


@pytest.mark.asyncio
async def test_session_restores_original_session_id():
    """session() restores a pre-existing session_id after exit."""
    @guard(session_id="original")
    async def my_tool(x: str) -> dict:
        return {"x": x}

    assert my_tool.config.session_id == "original"
    async with my_tool.session(session_id="temp-session") as t:
        assert t.config.session_id == "temp-session"
    assert my_tool.config.session_id == "original"
    global_registry.clear()


@pytest.mark.asyncio
async def test_session_auto_generates_session_id():
    """session() auto-generates a session_id when none is provided."""
    tool = make_async_tool()
    async with tool.session() as t:
        # Should have a non-None session_id inside the block
        assert t.config.session_id is not None
        assert len(t.config.session_id) > 0


@pytest.mark.asyncio
async def test_session_collects_trace_entries(tmp_path):
    """Calls made inside session() are captured as trace entries."""
    @guard(record=True, trace_dir=str(tmp_path))
    async def traced_tool(x: str) -> dict:
        return {"x": x}

    async with traced_tool.session(session_id="collect-test") as t:
        await t.acall("hello")
        await t.acall("world")

    # Entries are exposed on _session_entries after exit
    assert hasattr(traced_tool, "_session_entries")
    assert len(traced_tool._session_entries) == 2
    names = [e.call.args[0] for e in traced_tool._session_entries]
    assert "hello" in names
    assert "world" in names
    global_registry.clear()


@pytest.mark.asyncio
async def test_session_entries_have_correct_session_id(tmp_path):
    """Trace entries collected in a session block carry the pinned session_id."""
    @guard(record=True, trace_dir=str(tmp_path))
    async def sid_tool(x: str) -> dict:
        return {"x": x}

    async with sid_tool.session(session_id="pinned-id") as t:
        await t.acall("check")

    for entry in sid_tool._session_entries:
        assert entry.call.session_id == "pinned-id"
    global_registry.clear()


@pytest.mark.asyncio
async def test_session_exits_cleanly_on_exception():
    """session() restores state even when an exception is raised inside the block."""
    tool = make_async_tool()

    with pytest.raises(RuntimeError):
        async with tool.session(session_id="error-session") as t:
            raise RuntimeError("intentional error")

    # session_id is restored
    assert tool.config.session_id is None


@pytest.mark.asyncio
async def test_session_multiple_calls_return_values():
    """Calls inside session() return correct values."""
    tool = make_async_tool(value=42)
    results = []
    async with tool.session(session_id="multi-call") as t:
        r1 = await t.acall("a")
        r2 = await t.acall("b")
        results.extend([r1, r2])

    assert results[0]["result"] == "a"
    assert results[1]["result"] == "b"


# ---------------------------------------------------------------------------
# async_record_session() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_record_session_basic(tmp_path):
    """async_record_session collects entries from async acall() calls."""
    @guard
    async def my_tool(x: str) -> dict:
        return {"x": x}

    async with async_record_session(storage=str(tmp_path)) as recorder:
        await my_tool.acall("foo")
        await my_tool.acall("bar")

    entries = recorder.entries()
    assert len(entries) == 2
    args = [e.call.args[0] for e in entries]
    assert "foo" in args
    assert "bar" in args
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_explicit_session_id(tmp_path):
    """async_record_session respects an explicit session_id."""
    @guard
    async def another_tool(x: str) -> dict:
        return {"x": x}

    sid = "explicit-async-session"
    async with async_record_session(storage=str(tmp_path), session_id=sid) as recorder:
        await another_tool.acall("test")

    assert recorder.session_id == sid
    entries = recorder.entries()
    assert len(entries) == 1
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_auto_session_id(tmp_path):
    """async_record_session generates a session_id when not provided."""
    @guard
    async def auto_sid_tool(x: str) -> dict:
        return {"x": x}

    async with async_record_session(storage=str(tmp_path)) as recorder:
        await auto_sid_tool.acall("hi")

    assert recorder.session_id is not None
    assert len(recorder.session_id) > 0
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_stats(tmp_path):
    """async_record_session.stats() returns aggregated info."""
    @guard
    async def stats_tool(x: str) -> dict:
        return {"x": x}

    async with async_record_session(storage=str(tmp_path)) as recorder:
        await stats_tool.acall("a")
        await stats_tool.acall("b")
        await stats_tool.acall("c")

    stats = recorder.stats()
    assert stats["total_calls"] == 3
    assert stats["successes"] == 3
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_multiple_tools(tmp_path):
    """async_record_session collects entries from multiple guarded tools."""
    @guard
    async def tool_one(x: str) -> dict:
        return {"from": "one", "x": x}

    @guard
    async def tool_two(x: str) -> dict:
        return {"from": "two", "x": x}

    async with async_record_session(storage=str(tmp_path)) as recorder:
        await tool_one.acall("ping")
        await tool_two.acall("pong")

    entries = recorder.entries()
    assert len(entries) == 2
    tool_names = {e.tool_name for e in entries}
    assert "tool_one" in tool_names
    assert "tool_two" in tool_names
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_exits_on_exception(tmp_path):
    """async_record_session cleans up even when an exception occurs."""
    from agentguard.core.trace import get_active_recorders

    @guard
    async def exc_tool(x: str) -> dict:
        return {"x": x}

    with pytest.raises(ValueError):
        async with async_record_session(storage=str(tmp_path)) as recorder:
            await exc_tool.acall("before")
            raise ValueError("boom")

    # Recorder should no longer be active
    assert recorder not in get_active_recorders()
    # Entry recorded before exception should still be accessible
    entries = recorder.entries()
    assert len(entries) == 1
    global_registry.clear()


@pytest.mark.asyncio
async def test_async_record_session_is_inactive_after_exit(tmp_path):
    """After the async with block the recorder is no longer in active recorders."""
    from agentguard.core.trace import get_active_recorders

    @guard
    async def inactive_tool(x: str) -> dict:
        return {"x": x}

    async with async_record_session(storage=str(tmp_path)) as recorder:
        assert recorder in get_active_recorders()

    assert recorder not in get_active_recorders()
    global_registry.clear()


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_async_record_session_importable():
    """async_record_session is importable from top-level agentguard package."""
    from agentguard import async_record_session as ars  # noqa: F401
    assert ars is not None


def test_version_updated():
    """__version__ should reflect 0.2.0."""
    import agentguard
    assert agentguard.__version__ == "0.2.0"
