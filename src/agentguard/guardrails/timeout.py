"""Timeout enforcement for tool calls.

Provides both a decorator-based and context-manager-based timeout,
implemented via threading (for sync functions) and asyncio.wait_for
(for async functions). Does not depend on UNIX signals so it works
on all platforms including Windows.

Usage::

    from agentguard.guardrails.timeout import timeout, with_timeout

    @timeout(seconds=5.0)
    def slow_api(endpoint: str) -> dict:
        ...

    # Context manager
    with with_timeout(5.0):
        result = slow_api("/data")

    # One-shot
    from agentguard.guardrails.timeout import run_with_timeout
    result = run_with_timeout(my_func, args=("x",), timeout_seconds=5.0)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
from typing import Any, Callable, Generator, Optional, TypeVar

from agentguard.core.types import ToolTimeoutError

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def timeout(
    seconds: float,
    *,
    tool_name: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator that enforces a wall-clock timeout on a function.

    Works for both synchronous and asynchronous functions.

    Args:
        seconds: Maximum execution time in seconds.
        tool_name: Override the function name used in error messages.

    Returns:
        A wrapped callable that raises :class:`ToolTimeoutError` if the
        function does not complete within *seconds*.
    """
    def decorator(fn: F) -> F:
        name = tool_name or fn.__name__

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise ToolTimeoutError(name, seconds)
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return run_with_timeout(fn, args=args, kwargs=kwargs, timeout_seconds=seconds, tool_name=name)
            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# run_with_timeout (used internally by guard.py)
# ---------------------------------------------------------------------------


def run_with_timeout(
    func: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: Optional[dict[str, Any]] = None,
    timeout_seconds: float,
    tool_name: str = "unknown",
) -> Any:
    """Execute *func* synchronously, raising ToolTimeoutError if it exceeds *timeout_seconds*.

    Uses a daemon thread so the call does not block indefinitely.

    Args:
        func: Callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        timeout_seconds: Maximum allowed execution time.
        tool_name: Used in the error message if timeout occurs.

    Returns:
        The return value of *func*.

    Raises:
        ToolTimeoutError: If the timeout elapses.
        Exception: Any exception raised by *func* is re-raised as-is.
    """
    if kwargs is None:
        kwargs = {}

    result_holder: list[Any] = [None, None]  # [return_value, exception]
    done = threading.Event()

    def _run() -> None:
        try:
            result_holder[0] = func(*args, **kwargs)
        except Exception as exc:
            result_holder[1] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    if not done.wait(timeout=timeout_seconds):
        raise ToolTimeoutError(tool_name, timeout_seconds)
    if result_holder[1] is not None:
        raise result_holder[1]
    return result_holder[0]


# ---------------------------------------------------------------------------
# Context manager (sync)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def with_timeout(
    seconds: float,
    *,
    tool_name: str = "block",
) -> Generator[None, None, None]:
    """Context manager that raises ToolTimeoutError if the block takes too long.

    .. note::
        This implementation uses a background thread timer and a threading.Event.
        The timeout is detected *after* the block completes (i.e., it cannot
        interrupt a running C extension). For hard interruption, use the
        :func:`run_with_timeout` function which isolates the call in a thread.

    Args:
        seconds: Maximum execution time in seconds.
        tool_name: Used in the error message.

    Raises:
        ToolTimeoutError: If the block exceeds *seconds*.
    """
    timed_out = threading.Event()
    timer: Optional[threading.Timer] = None

    def _trigger() -> None:
        timed_out.set()

    try:
        timer = threading.Timer(seconds, _trigger)
        timer.daemon = True
        timer.start()
        yield
    finally:
        if timer:
            timer.cancel()

    if timed_out.is_set():
        raise ToolTimeoutError(tool_name, seconds)
