"""Custom pytest-style assertions for tool calls.

Provides a fluent assertion API that integrates naturally with pytest,
giving clear failure messages about what a tool call returned.

Example::

    from agentguard.testing.assertions import assert_tool_call

    entry = recorder.entries()[-1]

    assert_tool_call(entry).succeeded()
    assert_tool_call(entry).returned_dict().has_keys("results", "count")
    assert_tool_call(entry).executed_within_ms(500)
    assert_tool_call(entry).not_hallucinated()
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agentguard.core.types import ToolCallStatus, TraceEntry


# ---------------------------------------------------------------------------
# AssertionBuilder — fluent API
# ---------------------------------------------------------------------------


class AssertionBuilder:
    """Fluent assertion builder for a single :class:`~agentguard.core.types.TraceEntry`.

    Methods return ``self`` to allow chaining::

        assert_tool_call(entry) \\
            .succeeded() \\
            .executed_within_ms(200) \\
            .returned_dict() \\
            .has_keys("status", "data")
    """

    def __init__(self, entry: TraceEntry) -> None:
        self._entry = entry

    # ------------------------------------------------------------------
    # Status assertions
    # ------------------------------------------------------------------

    def succeeded(self) -> "AssertionBuilder":
        """Assert the tool call completed successfully."""
        status = self._entry.result.status
        assert status == ToolCallStatus.SUCCESS, (
            f"Expected tool '{self._entry.tool_name}' to succeed, "
            f"but got status {status.value!r}"
            + (f" ({self._entry.result.exception})" if self._entry.result.exception else "")
        )
        return self

    def failed(self) -> "AssertionBuilder":
        """Assert the tool call resulted in a failure."""
        assert self._entry.result.failed, (
            f"Expected tool '{self._entry.tool_name}' to fail, "
            f"but it succeeded with status {self._entry.result.status.value!r}"
        )
        return self

    def had_status(self, status: ToolCallStatus) -> "AssertionBuilder":
        """Assert the tool call had a specific status.

        Args:
            status: The expected :class:`~agentguard.core.types.ToolCallStatus`.
        """
        actual = self._entry.result.status
        assert actual == status, (
            f"Expected tool '{self._entry.tool_name}' to have status {status.value!r}, "
            f"but got {actual.value!r}"
        )
        return self

    # ------------------------------------------------------------------
    # Timing assertions
    # ------------------------------------------------------------------

    def executed_within_ms(self, max_ms: float) -> "AssertionBuilder":
        """Assert the tool completed within *max_ms* milliseconds.

        Args:
            max_ms: Maximum allowed execution time in milliseconds.
        """
        actual = self._entry.result.execution_time_ms
        assert actual <= max_ms, (
            f"Tool '{self._entry.tool_name}' took {actual:.1f}ms, "
            f"exceeding the {max_ms:.1f}ms limit"
        )
        return self

    def executed_at_least_ms(self, min_ms: float) -> "AssertionBuilder":
        """Assert the tool took at least *min_ms* milliseconds (real I/O check).

        Args:
            min_ms: Minimum expected execution time in milliseconds.
        """
        actual = self._entry.result.execution_time_ms
        assert actual >= min_ms, (
            f"Tool '{self._entry.tool_name}' completed in {actual:.2f}ms, "
            f"which is suspiciously faster than the {min_ms:.1f}ms minimum"
        )
        return self

    # ------------------------------------------------------------------
    # Return value assertions
    # ------------------------------------------------------------------

    def returned(self, expected: Any) -> "AssertionBuilder":
        """Assert the tool returned exactly *expected*.

        Args:
            expected: The expected return value.
        """
        actual = self._entry.result.return_value
        assert actual == expected, (
            f"Tool '{self._entry.tool_name}' returned {actual!r}, expected {expected!r}"
        )
        return self

    def returned_type(self, expected_type: type) -> "AssertionBuilder":
        """Assert the return value is an instance of *expected_type*.

        Args:
            expected_type: The expected Python type.
        """
        actual = self._entry.result.return_value
        assert isinstance(actual, expected_type), (
            f"Tool '{self._entry.tool_name}' returned {type(actual).__name__!r}, "
            f"expected {expected_type.__name__!r}"
        )
        return self

    def returned_dict(self) -> "AssertionBuilder":
        """Assert the return value is a dict."""
        return self.returned_type(dict)

    def returned_list(self) -> "AssertionBuilder":
        """Assert the return value is a list."""
        return self.returned_type(list)

    def returned_str(self) -> "AssertionBuilder":
        """Assert the return value is a string."""
        return self.returned_type(str)

    def returned_non_empty(self) -> "AssertionBuilder":
        """Assert the return value is non-empty (string, list, dict, etc.)."""
        actual = self._entry.result.return_value
        assert actual, (
            f"Tool '{self._entry.tool_name}' returned an empty or falsy value: {actual!r}"
        )
        return self

    # ------------------------------------------------------------------
    # Dict-specific assertions
    # ------------------------------------------------------------------

    def has_keys(self, *keys: str) -> "AssertionBuilder":
        """Assert the return value (dict) contains all specified keys.

        Args:
            keys: Key names that must be present.
        """
        actual = self._entry.result.return_value
        assert isinstance(actual, dict), (
            f"Tool '{self._entry.tool_name}' did not return a dict "
            f"(got {type(actual).__name__!r}); cannot check keys"
        )
        missing = [k for k in keys if k not in actual]
        assert not missing, (
            f"Tool '{self._entry.tool_name}' response missing keys: {missing}. "
            f"Available keys: {sorted(actual.keys())}"
        )
        return self

    def has_key(self, key: str) -> "AssertionBuilder":
        """Assert the return value (dict) contains a specific key.

        Convenience alias for ``has_keys(key)``.

        Args:
            key: Key name that must be present.
        """
        return self.has_keys(key)

    def lacks_keys(self, *keys: str) -> "AssertionBuilder":
        """Assert the return value (dict) does NOT contain any of the specified keys.

        Args:
            keys: Key names that must be absent.
        """
        actual = self._entry.result.return_value
        assert isinstance(actual, dict), (
            f"Tool '{self._entry.tool_name}' did not return a dict"
        )
        present = [k for k in keys if k in actual]
        assert not present, (
            f"Tool '{self._entry.tool_name}' response contains unexpected keys: {present}"
        )
        return self

    def field_equals(self, key: str, expected: Any) -> "AssertionBuilder":
        """Assert ``result[key] == expected``.

        Args:
            key: The dict key to check.
            expected: The expected value.
        """
        actual = self._entry.result.return_value
        assert isinstance(actual, dict), (
            f"Tool '{self._entry.tool_name}' did not return a dict"
        )
        assert key in actual, (
            f"Key {key!r} not in result. Available: {sorted(actual.keys())}"
        )
        assert actual[key] == expected, (
            f"result[{key!r}] = {actual[key]!r}, expected {expected!r}"
        )
        return self

    def field_matches(self, key: str, pattern: str) -> "AssertionBuilder":
        """Assert ``result[key]`` matches a regex pattern.

        Args:
            key: The dict key to check.
            pattern: Regex pattern.
        """
        actual = self._entry.result.return_value
        assert isinstance(actual, dict)
        value = str(actual.get(key, ""))
        assert re.search(pattern, value), (
            f"result[{key!r}] = {value!r} does not match pattern {pattern!r}"
        )
        return self

    # ------------------------------------------------------------------
    # Hallucination assertions
    # ------------------------------------------------------------------

    def not_hallucinated(self) -> "AssertionBuilder":
        """Assert that the tool call was not detected as hallucinated."""
        hall = self._entry.result.hallucination
        if hall is not None:
            assert not hall.is_hallucinated, (
                f"Tool '{self._entry.tool_name}' was detected as hallucinated "
                f"(confidence={hall.confidence:.2f}): {hall.reason}"
            )
        return self

    def was_hallucinated(self) -> "AssertionBuilder":
        """Assert that the tool call WAS detected as hallucinated."""
        hall = self._entry.result.hallucination
        assert hall is not None, (
            f"No hallucination analysis was performed for '{self._entry.tool_name}'"
        )
        assert hall.is_hallucinated, (
            f"Tool '{self._entry.tool_name}' was NOT detected as hallucinated "
            f"(confidence={hall.confidence:.2f})"
        )
        return self

    # ------------------------------------------------------------------
    # Retry assertions
    # ------------------------------------------------------------------

    def was_retried(self, times: Optional[int] = None) -> "AssertionBuilder":
        """Assert the tool call was retried.

        Args:
            times: Exact number of retries expected. If None, just check > 0.
        """
        actual = self._entry.result.retry_count
        if times is not None:
            assert actual == times, (
                f"Tool '{self._entry.tool_name}' was retried {actual} time(s), "
                f"expected {times}"
            )
        else:
            assert actual > 0, (
                f"Tool '{self._entry.tool_name}' was not retried (retry_count=0)"
            )
        return self

    def was_not_retried(self) -> "AssertionBuilder":
        """Assert the tool call succeeded on the first attempt."""
        actual = self._entry.result.retry_count
        assert actual == 0, (
            f"Tool '{self._entry.tool_name}' was retried {actual} time(s)"
        )
        return self

    # ------------------------------------------------------------------
    # Validation assertions
    # ------------------------------------------------------------------

    def all_validations_passed(self) -> "AssertionBuilder":
        """Assert all validation checks passed."""
        failed = [v for v in self._entry.result.validations if not v.valid]
        assert not failed, (
            f"Tool '{self._entry.tool_name}' had validation failures: "
            + "; ".join(v.message for v in failed)
        )
        return self

    def __repr__(self) -> str:
        return (
            f"AssertionBuilder(tool={self._entry.tool_name!r}, "
            f"status={self._entry.result.status.value!r})"
        )


# ---------------------------------------------------------------------------
# Entry-point function
# ---------------------------------------------------------------------------


def assert_tool_call(entry: TraceEntry) -> AssertionBuilder:
    """Create a fluent :class:`AssertionBuilder` for *entry*.

    Example::

        assert_tool_call(entry).succeeded().returned_dict().has_keys("data")

    Args:
        entry: A :class:`~agentguard.core.types.TraceEntry` from a trace recorder.

    Returns:
        :class:`AssertionBuilder` ready for chaining.
    """
    return AssertionBuilder(entry)


# ---------------------------------------------------------------------------
# Standalone assertion helpers
# ---------------------------------------------------------------------------


def assert_no_hallucinations(entries: list[TraceEntry]) -> None:
    """Assert none of the entries were detected as hallucinated.

    Args:
        entries: List of trace entries to check.

    Raises:
        AssertionError: If any entry has a positive hallucination result.
    """
    for entry in entries:
        assert_tool_call(entry).not_hallucinated()


def assert_all_succeeded(entries: list[TraceEntry]) -> None:
    """Assert all entries have SUCCESS status.

    Args:
        entries: List of trace entries to check.
    """
    for entry in entries:
        assert_tool_call(entry).succeeded()


def assert_latency_budget(entries: list[TraceEntry], max_ms: float) -> None:
    """Assert every entry completed within *max_ms* milliseconds.

    Args:
        entries: List of trace entries.
        max_ms: Maximum allowed execution time.
    """
    for entry in entries:
        assert_tool_call(entry).executed_within_ms(max_ms)
