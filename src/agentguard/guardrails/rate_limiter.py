"""Token-bucket rate limiter for tool calls.

Implements a thread-safe token bucket algorithm that supports per-second,
per-minute, and per-hour rate limits with configurable burst capacity.

The token bucket works like this:
- Tokens accumulate at *rate* tokens/second up to *burst* maximum.
- Each call consumes one token.
- When the bucket is empty, the call is rate-limited.

Usage::

    from agentguard.guardrails.rate_limiter import RateLimiter

    rl = RateLimiter(calls_per_minute=60, burst=10)

    @guard(rate_limit=rl.config)
    def search_web(query: str) -> dict:
        ...

    # Standalone usage
    allowed, retry_after = rl.acquire("my_tool")
    if not allowed:
        print(f"Rate limited, retry in {retry_after:.1f}s")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from agentguard.core.types import GuardAction, RateLimitConfig, RateLimitError


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------


@dataclass
class RateLimiterStats:
    """Snapshot of rate limiter statistics for a single tool."""

    tool_name: str
    current_tokens: float
    max_tokens: int
    rate_per_second: float
    total_allowed: int
    total_rejected: int


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token-bucket rate limiter for agentguard-protected tools.

    This class can be used standalone or via the ``@guard`` decorator by
    passing its :attr:`config` property.

    Example::

        rl = RateLimiter(calls_per_second=10, burst=20)

        @guard(rate_limit=rl.config)
        def my_tool(x: str) -> dict:
            ...
    """

    def __init__(
        self,
        *,
        calls_per_second: Optional[float] = None,
        calls_per_minute: Optional[float] = None,
        calls_per_hour: Optional[float] = None,
        burst: int = 1,
        on_limit: GuardAction = GuardAction.BLOCK,
    ) -> None:
        """Initialise the rate limiter.

        Supply exactly one of *calls_per_second*, *calls_per_minute*, or
        *calls_per_hour*. They are additive if multiple are given.

        Args:
            calls_per_second: Allowed calls per second.
            calls_per_minute: Allowed calls per minute.
            calls_per_hour: Allowed calls per hour.
            burst: Maximum token bucket size (initial burst allowance).
            on_limit: Action when the rate limit is hit.
        """
        self._cfg = RateLimitConfig(
            calls_per_second=calls_per_second,
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            burst=burst,
            on_limit=on_limit,
        )
        # Compute effective tokens/second
        self._rate = 0.0
        if calls_per_second:
            self._rate += calls_per_second
        if calls_per_minute:
            self._rate += calls_per_minute / 60.0
        if calls_per_hour:
            self._rate += calls_per_hour / 3600.0

        self._buckets: dict[str, _Bucket] = {}
        self._global_bucket: Optional[_Bucket] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(
        self,
        tool_name: str = "global",
        *,
        per_tool: bool = False,
    ) -> tuple[bool, float]:
        """Try to consume one token for *tool_name*.

        Args:
            tool_name: The tool being called. Used to key per-tool buckets
                when *per_tool* is True.
            per_tool: If True, maintain separate buckets per tool name.
                If False (default), share a single global bucket.

        Returns:
            ``(allowed, retry_after_seconds)`` — if not allowed, retry_after
            is the number of seconds until the next token is available.
        """
        bucket = self._get_bucket(tool_name, per_tool=per_tool)
        allowed, retry_after = bucket.consume()
        if allowed:
            bucket.total_allowed += 1
        else:
            bucket.total_rejected += 1
        return allowed, retry_after

    def require(self, tool_name: str = "global", *, per_tool: bool = False) -> None:
        """Acquire a token or raise :class:`~agentguard.core.types.RateLimitError`.

        Args:
            tool_name: Tool being called.
            per_tool: Maintain separate per-tool bucket.

        Raises:
            RateLimitError: If the rate limit is exceeded and on_limit=BLOCK.
        """
        allowed, retry_after = self.acquire(tool_name, per_tool=per_tool)
        if not allowed:
            if self._cfg.on_limit == GuardAction.BLOCK:
                raise RateLimitError(tool_name, retry_after)
            if self._cfg.on_limit == GuardAction.WARN:
                import warnings
                warnings.warn(
                    f"Rate limit exceeded for '{tool_name}'; "
                    f"retry after {retry_after:.2f}s",
                    stacklevel=3,
                )

    def reset(self, tool_name: Optional[str] = None, *, per_tool: bool = False) -> None:
        """Reset token bucket(s) to full capacity.

        Args:
            tool_name: Specific tool to reset, or None to reset all.
            per_tool: Whether per-tool buckets are in use.
        """
        with self._lock:
            if tool_name and per_tool:
                if tool_name in self._buckets:
                    self._buckets[tool_name].reset(self._cfg.burst)
            elif self._global_bucket:
                self._global_bucket.reset(self._cfg.burst)
            else:
                for b in self._buckets.values():
                    b.reset(self._cfg.burst)

    def stats(self, tool_name: str = "global", *, per_tool: bool = False) -> RateLimiterStats:
        """Return a stats snapshot.

        Args:
            tool_name: Tool to query (or ``"global"`` for the shared bucket).
            per_tool: Whether per-tool buckets are in use.
        """
        bucket = self._get_bucket(tool_name, per_tool=per_tool)
        return RateLimiterStats(
            tool_name=tool_name,
            current_tokens=bucket.tokens,
            max_tokens=self._cfg.burst,
            rate_per_second=self._rate,
            total_allowed=bucket.total_allowed,
            total_rejected=bucket.total_rejected,
        )

    @property
    def config(self) -> RateLimitConfig:
        """Return the :class:`~agentguard.core.types.RateLimitConfig`."""
        return self._cfg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_bucket(self, tool_name: str, *, per_tool: bool) -> "_Bucket":
        with self._lock:
            if per_tool:
                if tool_name not in self._buckets:
                    self._buckets[tool_name] = _Bucket(self._rate, self._cfg.burst)
                return self._buckets[tool_name]
            if self._global_bucket is None:
                self._global_bucket = _Bucket(self._rate, self._cfg.burst)
            return self._global_bucket

    def __repr__(self) -> str:
        return (
            f"RateLimiter(rate={self._rate:.3f}/s, burst={self._cfg.burst})"
        )


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _Bucket:
    """Thread-safe token bucket."""

    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = rate
        self._capacity = capacity
        self.tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self.total_allowed = 0
        self.total_rejected = 0

    def consume(self) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after)."""
        if self._rate <= 0:
            return True, 0.0
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(
                float(self._capacity),
                self.tokens + elapsed * self._rate,
            )
            self._last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            retry_after = (1.0 - self.tokens) / self._rate
            return False, retry_after

    def reset(self, capacity: int) -> None:
        with self._lock:
            self.tokens = float(capacity)
            self._last_refill = time.monotonic()
