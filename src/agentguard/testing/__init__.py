"""agentguard testing utilities.

Provides trace recording, test generation, replay, and assertion helpers.
"""

from agentguard.testing.assertions import (
    AssertionBuilder,
    assert_all_succeeded,
    assert_latency_budget,
    assert_no_hallucinations,
    assert_tool_call,
)
from agentguard.testing.generator import TestGenerator
from agentguard.testing.recorder import TraceRecorder, record_session
from agentguard.testing.replayer import ReplayReport, ReplayResult, TraceReplayer

__all__ = [
    "assert_all_succeeded",
    "assert_latency_budget",
    "assert_no_hallucinations",
    "assert_tool_call",
    "AssertionBuilder",
    "record_session",
    "TestGenerator",
    "TraceRecorder",
    "TraceReplayer",
    "ReplayReport",
    "ReplayResult",
]
