"""Composable middleware pipeline for agentguard-protected tool calls.

Inspired by Express.js / Koa, this module lets users chain middleware
functions that run before and/or after each tool call.  Each middleware
receives a :class:`MiddlewareContext` and a ``next`` callable that invokes
the remainder of the chain (ultimately executing the tool itself).

Usage::

    from agentguard.middleware import Middleware, MiddlewareChain

    async def logging_middleware(ctx, next):
        print(f"Calling {ctx.tool_name} with args={ctx.args}")
        result = await next(ctx)
        print(f"  -> status={result.status}")
        return result

    async def auth_middleware(ctx, next):
        if not ctx.metadata.get("api_key"):
            raise PermissionError("No API key provided")
        return await next(ctx)

    chain = MiddlewareChain()
    chain.use(logging_middleware)
    chain.use(auth_middleware)

    @guard(middleware=chain)
    def my_tool(x: str) -> str:
        return x

Both sync and async middleware are supported.  Sync middleware is wrapped
transparently so that it may be mixed freely with async middleware.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Union

from agentguard.core.redaction import sanitize_value
from agentguard.core.types import GuardConfig, ToolCallStatus, ToolResult

__all__ = [
    "Middleware",
    "MiddlewareContext",
    "MiddlewareChain",
    "NextFunc",
    "logging_middleware",
    "metadata_middleware",
    "timing_middleware",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A ``next`` callable that advances the middleware chain.
NextFunc = Callable[["MiddlewareContext"], Awaitable[ToolResult]]

#: A middleware function: ``async (ctx, next) -> ToolResult``.
#: Sync callables are accepted and promoted to coroutines automatically.
Middleware = Callable[["MiddlewareContext", NextFunc], Awaitable[ToolResult]]


# ---------------------------------------------------------------------------
# MiddlewareContext
# ---------------------------------------------------------------------------


@dataclass
class MiddlewareContext:
    """Contextual information passed through the middleware chain.

    Attributes:
        tool_name: The name of the guarded tool being called.
        args: Positional arguments for the tool call.
        kwargs: Keyword arguments for the tool call.
        config: The :class:`~agentguard.core.types.GuardConfig` for this tool.
        metadata: Arbitrary caller-supplied metadata (e.g. ``api_key``,
            ``session_id``, ``user_id``).
        timestamps: Dict with ``"start"`` key set to the call start time
            (UTC).  Middleware may add additional timestamps (e.g.
            ``"auth_checked"``) for tracing purposes.
        call_id: Unique identifier for this specific invocation.
    """

    tool_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    config: GuardConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, datetime] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: _new_call_id())

    def __post_init__(self) -> None:
        if "start" not in self.timestamps:
            self.timestamps["start"] = datetime.now(tz=timezone.utc)

    # Convenience helpers --------------------------------------------------

    def mark(self, label: str) -> None:
        """Record a named timestamp in :attr:`timestamps`.

        Args:
            label: Key to store under (e.g. ``"validated"``).
        """
        self.timestamps[label] = datetime.now(tz=timezone.utc)

    def elapsed_ms(self, *, since: str = "start") -> float:
        """Return milliseconds elapsed since a named timestamp.

        Args:
            since: Key in :attr:`timestamps` to measure from.

        Returns:
            Elapsed time in milliseconds, or 0.0 if the key is missing.
        """
        ts = self.timestamps.get(since)
        if ts is None:
            return 0.0
        delta = datetime.now(tz=timezone.utc) - ts
        return delta.total_seconds() * 1000.0


# ---------------------------------------------------------------------------
# MiddlewareChain
# ---------------------------------------------------------------------------


class MiddlewareChain:
    """An ordered list of middleware functions to run around every tool call.

    Middleware is invoked in the order it was added via :meth:`use`.  Each
    middleware *must* call ``await next(ctx)`` to pass control to subsequent
    middleware (and eventually the tool itself), unless it intentionally
    short-circuits the chain.

    Example::

        chain = MiddlewareChain()
        chain.use(logging_middleware)
        chain.use(auth_middleware)

        @guard(middleware=chain)
        def search(q: str) -> list[str]: ...
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def use(self, middleware: Middleware) -> "MiddlewareChain":
        """Register a middleware function.

        Args:
            middleware: A callable with signature
                ``(ctx: MiddlewareContext, next: NextFunc) -> ToolResult``.
                May be a regular function or a coroutine function.

        Returns:
            ``self`` so calls can be chained fluently::

                chain.use(logging_mw).use(auth_mw).use(tracing_mw)
        """
        self._middleware.append(_ensure_async(middleware))
        return self

    def __len__(self) -> int:
        return len(self._middleware)

    def __repr__(self) -> str:
        names = [getattr(m, "__name__", repr(m)) for m in self._middleware]
        return f"MiddlewareChain([{', '.join(names)}])"

    # ------------------------------------------------------------------
    # Internal: build the composed async handler
    # ------------------------------------------------------------------

    def build(self, terminal: NextFunc) -> NextFunc:
        """Compose all registered middleware around *terminal*.

        Args:
            terminal: The innermost async callable that actually executes the
                tool (i.e. returns a :class:`~agentguard.core.types.ToolResult`).

        Returns:
            A single async callable that, when called with a
            :class:`MiddlewareContext`, runs the entire stack.
        """
        # Build the chain from the inside out (right-to-left)
        handler: NextFunc = terminal
        for mw in reversed(self._middleware):
            handler = _make_next(mw, handler)
        return handler

    async def run(
        self,
        ctx: MiddlewareContext,
        terminal: NextFunc,
    ) -> ToolResult:
        """Execute the full middleware stack and return the final result.

        Args:
            ctx: The middleware context for this invocation.
            terminal: The innermost callable that executes the actual tool.

        Returns:
            The :class:`~agentguard.core.types.ToolResult` produced by the
            pipeline.
        """
        handler = self.build(terminal)
        return await handler(ctx)

    # ------------------------------------------------------------------
    # Copy helpers
    # ------------------------------------------------------------------

    def copy(self) -> "MiddlewareChain":
        """Return a shallow copy of this chain.

        Useful when different tools share most but not all middleware.
        """
        new = MiddlewareChain()
        new._middleware = list(self._middleware)
        return new

    def prepend(self, middleware: Middleware) -> "MiddlewareChain":
        """Add *middleware* at the front of the chain (runs first).

        Args:
            middleware: Middleware to prepend.

        Returns:
            ``self`` for fluent chaining.
        """
        self._middleware.insert(0, _ensure_async(middleware))
        return self


# ---------------------------------------------------------------------------
# Convenience middleware factories
# ---------------------------------------------------------------------------


def logging_middleware(
    *,
    log_args: bool = False,
    log_result: bool = True,
    redact_fields: tuple[str, ...] = (),
) -> Middleware:
    """Return a middleware that prints a log line before and after each call.

    Args:
        log_args: Include positional/keyword arguments in the log output.
        log_result: Include the result status in the post-call log output.

    Returns:
        An async middleware function.
    """
    async def _logging(ctx: MiddlewareContext, next: NextFunc) -> ToolResult:
        prefix = f"[agentguard] {ctx.tool_name}"
        if log_args:
            safe_args = sanitize_value(ctx.args, extra_fields=redact_fields)
            safe_kwargs = sanitize_value(ctx.kwargs, extra_fields=redact_fields)
            print(f"{prefix} args={safe_args!r} kwargs={safe_kwargs!r}")
        else:
            print(f"{prefix} called")
        result = await next(ctx)
        if log_result:
            print(f"{prefix} -> {result.status.value} ({result.execution_time_ms:.1f}ms)")
        return result

    _logging.__name__ = "logging_middleware"
    return _logging


def metadata_middleware(**static_metadata: Any) -> Middleware:
    """Return a middleware that injects static key/value pairs into ``ctx.metadata``.

    Args:
        **static_metadata: Key/value pairs to inject before calling ``next``.

    Returns:
        An async middleware function.
    """
    async def _inject(ctx: MiddlewareContext, next: NextFunc) -> ToolResult:
        ctx.metadata.update(static_metadata)
        return await next(ctx)

    _inject.__name__ = "metadata_middleware"
    return _inject


def timing_middleware() -> Middleware:
    """Return a middleware that measures and stores wall-clock execution time.

    The elapsed time (in milliseconds) is stored in ``ctx.metadata["elapsed_ms"]``
    after the tool returns.

    Returns:
        An async middleware function.
    """
    async def _timing(ctx: MiddlewareContext, next: NextFunc) -> ToolResult:
        t0 = time.perf_counter()
        result = await next(ctx)
        ctx.metadata["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    _timing.__name__ = "timing_middleware"
    return _timing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_async(fn: Middleware) -> Middleware:
    """Wrap a sync middleware so it behaves like an async one.

    Sync middleware may call ``next(ctx)`` and return the resulting coroutine,
    or return a concrete :class:`~agentguard.core.types.ToolResult` directly.
    Both cases are handled transparently.
    """
    if asyncio.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _wrapper(ctx: MiddlewareContext, next: NextFunc) -> ToolResult:
        result = fn(ctx, next)  # type: ignore[call-arg]
        # If the sync middleware returned a coroutine (e.g. return next(ctx)),
        # await it so the caller always receives a ToolResult.
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    return _wrapper


def _make_next(mw: Middleware, downstream: NextFunc) -> NextFunc:
    """Return a ``next`` function that calls *mw* with *downstream* as its next."""

    async def _next(ctx: MiddlewareContext) -> ToolResult:
        return await mw(ctx, downstream)

    return _next


def _new_call_id() -> str:
    import uuid
    return str(uuid.uuid4())
