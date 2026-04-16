"""Trace recorder for production tool calls.

Re-exports TraceRecorder from agentguard.core.trace for use via the
testing sub-package, and adds a higher-level SessionRecorder that
groups multiple tool calls under one session.

Example::

    from agentguard.testing import TraceRecorder

    with TraceRecorder(storage="./traces") as recorder:
        result = my_guarded_tool("hello")

    print(recorder.stats())
    for entry in recorder.entries():
        print(entry.tool_name, entry.result.status)
"""

from __future__ import annotations

# Re-export the core recorder so users can import from agentguard.testing
from agentguard.core.trace import TraceRecorder, TraceStore, get_active_recorders, record_session

__all__ = [
    "TraceRecorder",
    "TraceStore",
    "get_active_recorders",
    "record_session",
]
