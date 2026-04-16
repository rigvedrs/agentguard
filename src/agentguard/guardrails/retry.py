"""Smart retry with exponential backoff and jitter.

Provides a flexible retry mechanism for tool calls that fail transiently.
Supports:
- Exponential backoff with configurable multiplier
- Full jitter (randomised delay to prevent thundering herd)
- Per-exception-type filtering (only retry on certain errors)
- Custom retry predicates
- Async support

Usage::

    from agentguard.guardrails.retry import retry, RetryPolicy

    policy = RetryPolicy(max_retries=3, initial_delay=1.0, backoff_factor=2.0)

    @retry(policy)
    def flaky_api(endpoint: str) -> dict:
        ...

    # Or with guard
    @guard(max_retries=3)
    def call_api(endpoint: str) -> dict:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, Type, TypeVar

from agentguard.core.types import RetryConfig

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Configurable retry policy with exponential backoff and jitter.

    Example::

        policy = RetryPolicy(
            max_retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
            max_delay=30.0,
            jitter=True,
            retryable_exceptions=(IOError, TimeoutError),
        )
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
        retry_predicate: Optional[Callable[[Exception], bool]] = None,
    ) -> None:
        """Initialise the retry policy.

        Args:
            max_retries: Maximum number of retry attempts after the first failure.
            initial_delay: Delay (seconds) before the first retry.
            backoff_factor: Multiplier applied to the delay on each attempt.
            max_delay: Maximum delay (seconds) between retries.
            jitter: Add randomised jitter (0.5×–1.5× range) to prevent thundering herd.
            retryable_exceptions: Exception types to retry on. If None (default),
                retry on all exceptions.
            retry_predicate: Custom callable ``(exc) -> bool`` that decides whether
                to retry. Takes precedence over *retryable_exceptions*.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.retry_predicate = retry_predicate

    def should_retry(self, exc: Exception) -> bool:
        """Return True if this exception should trigger a retry.

        Args:
            exc: The exception that was raised.
        """
        if self.retry_predicate is not None:
            return self.retry_predicate(exc)
        if self.retryable_exceptions:
            return isinstance(exc, self.retryable_exceptions)
        return True

    def delay_for(self, attempt: int) -> float:
        """Return the delay in seconds before *attempt* (0-based).

        Args:
            attempt: The attempt number (0 = first retry).
        """
        delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random()  # ±50% jitter (0.5x–1.5x)
        return delay

    def to_config(self) -> RetryConfig:
        """Convert to a :class:`~agentguard.core.types.RetryConfig`."""
        return RetryConfig(
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            backoff_factor=self.backoff_factor,
            jitter=self.jitter,
            retryable_exceptions=self.retryable_exceptions or (),
        )

    def __repr__(self) -> str:
        return (
            f"RetryPolicy(max_retries={self.max_retries}, "
            f"initial_delay={self.initial_delay}, "
            f"backoff_factor={self.backoff_factor})"
        )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def retry(
    policy: Optional[RetryPolicy] = None,
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator that retries a function on failure.

    Can accept a :class:`RetryPolicy` or keyword arguments directly.

    Args:
        policy: Pre-built :class:`RetryPolicy`. Keyword args are ignored if provided.
        max_retries: Max retry attempts.
        initial_delay: Initial delay in seconds.
        backoff_factor: Exponential multiplier.
        max_delay: Maximum delay cap.
        jitter: Add random jitter.
        retryable_exceptions: Only retry these exception types.

    Returns:
        Decorator that wraps the function with retry logic.
    """
    if policy is None:
        policy = RetryPolicy(
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
        )

    def decorator(fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Optional[Exception] = None
                for attempt in range(policy.max_retries + 1):  # type: ignore[union-attr]
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt < policy.max_retries and policy.should_retry(exc):  # type: ignore[union-attr]
                            delay = policy.delay_for(attempt)  # type: ignore[union-attr]
                            await asyncio.sleep(delay)
                        else:
                            raise
                raise last_exc  # type: ignore[misc]
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Optional[Exception] = None
                for attempt in range(policy.max_retries + 1):  # type: ignore[union-attr]
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt < policy.max_retries and policy.should_retry(exc):  # type: ignore[union-attr]
                            delay = policy.delay_for(attempt)  # type: ignore[union-attr]
                            time.sleep(delay)
                        else:
                            raise
                raise last_exc  # type: ignore[misc]
            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Utility: compute delay from a RetryConfig
# ---------------------------------------------------------------------------


def compute_retry_delay(config: RetryConfig, attempt: int) -> float:
    """Compute the delay before the next retry using *config*.

    Args:
        config: The retry configuration.
        attempt: Zero-based retry attempt number.

    Returns:
        Delay in seconds.
    """
    delay = min(
        config.initial_delay * (config.backoff_factor ** attempt),
        config.max_delay,
    )
    if config.jitter:
        delay *= 0.5 + random.random() * 0.5
    return delay
