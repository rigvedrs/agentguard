"""Semantic output validation.

Checks that a tool's output is *meaningful* given the input — beyond pure
schema conformance. For example:

- A weather tool called for "New York" should return a location containing
  "New York" somewhere, not "Tokyo".
- A search tool should return results that contain at least one word from the query.
- A translation tool's output should not be identical to its input (unless
  source == target language).

Semantic validators are registered per tool and run as part of the guard
pipeline when ``validate_output=True``.

Example::

    from agentguard.validators.semantic import SemanticValidator

    sv = SemanticValidator()

    @sv.validator("search_web")
    def check_search_results(query: str, result: dict) -> str | None:
        # Return a non-empty string on failure, or None/empty string on success
        if not result.get("results"):
            return "search returned no results"
        return None

    # Apply manually
    violations = sv.validate("search_web", args=("python tutorials",), result={"results": [...]})
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agentguard.core.types import ValidationResult, ValidatorKind


# ---------------------------------------------------------------------------
# Semantic validator registry
# ---------------------------------------------------------------------------


class SemanticValidator:
    """Registry of semantic validators for guarded tools.

    A semantic validator is a callable that accepts the tool's input arguments
    and its return value, and returns either ``None`` (validation passed) or
    a non-empty string describing the failure.

    Example::

        sv = SemanticValidator()

        @sv.validator("get_weather")
        def location_match(city: str, result: dict) -> str | None:
            if city.lower() not in str(result).lower():
                return f"Response does not mention input city '{city}'"
    """

    def __init__(self) -> None:
        self._validators: dict[str, list[Callable[..., Optional[str]]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def validator(
        self,
        tool_name: str,
    ) -> Callable[[Callable[..., Optional[str]]], Callable[..., Optional[str]]]:
        """Decorator that registers a semantic validator for *tool_name*.

        The decorated function receives the tool's positional arguments
        (unpacked) followed by the result as ``result=<value>``.

        Args:
            tool_name: The tool this validator applies to.

        Returns:
            Decorator that registers and returns the function unchanged.
        """
        def decorator(fn: Callable[..., Optional[str]]) -> Callable[..., Optional[str]]:
            self.register(tool_name, fn)
            return fn
        return decorator

    def register(
        self,
        tool_name: str,
        fn: Callable[..., Optional[str]],
    ) -> None:
        """Register a semantic validator function for *tool_name*.

        Args:
            tool_name: The tool this validator applies to.
            fn: Callable ``(*args, result=<value>) -> str | None``.
                Return None to indicate pass; return a non-empty string to indicate failure.
        """
        if tool_name not in self._validators:
            self._validators[tool_name] = []
        self._validators[tool_name].append(fn)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        tool_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> list[ValidationResult]:
        """Run all semantic validators registered for *tool_name*.

        Args:
            tool_name: The tool that was called.
            args: Positional arguments the tool was called with.
            kwargs: Keyword arguments the tool was called with.
            result: The value the tool returned.

        Returns:
            List of :class:`~agentguard.core.types.ValidationResult` objects.
        """
        validators = self._validators.get(tool_name, [])
        if not validators:
            return []

        results: list[ValidationResult] = []
        for fn in validators:
            try:
                failure_msg = fn(*args, **kwargs, result=result)
                if failure_msg:
                    results.append(ValidationResult(
                        valid=False,
                        kind=ValidatorKind.SEMANTIC,
                        message=failure_msg,
                    ))
                else:
                    results.append(ValidationResult(
                        valid=True,
                        kind=ValidatorKind.SEMANTIC,
                    ))
            except Exception as exc:
                results.append(ValidationResult(
                    valid=False,
                    kind=ValidatorKind.SEMANTIC,
                    message=f"Semantic validator '{fn.__name__}' raised: {exc}",
                ))
        return results

    def has_validators(self, tool_name: str) -> bool:
        """Return True if any validators are registered for *tool_name*."""
        return bool(self._validators.get(tool_name))


# ---------------------------------------------------------------------------
# Built-in semantic checks
# ---------------------------------------------------------------------------


def check_non_empty(result: Any, *, field: Optional[str] = None) -> Optional[str]:
    """Verify the result (or a specific field) is non-empty.

    Args:
        result: The tool's return value.
        field: If given, check ``result[field]`` instead of ``result`` itself.

    Returns:
        Failure message or None.
    """
    target = result.get(field) if field and isinstance(result, dict) else result
    if not target:
        label = f"result['{field}']" if field else "result"
        return f"Expected non-empty {label}, got {target!r}"
    return None


def check_key_present(result: Any, *, keys: list[str]) -> Optional[str]:
    """Verify *result* (dict) contains all required *keys*.

    Args:
        result: The tool's return value (expected to be a dict).
        keys: Keys that must be present.

    Returns:
        Failure message or None.
    """
    if not isinstance(result, dict):
        return f"Expected dict result for key check, got {type(result).__name__}"
    missing = [k for k in keys if k not in result]
    if missing:
        return f"Result missing required keys: {missing}"
    return None


def check_no_error_field(result: Any) -> Optional[str]:
    """Verify the result does not contain an ``error`` key with a truthy value.

    Args:
        result: The tool's return value.

    Returns:
        Failure message or None.
    """
    if isinstance(result, dict) and result.get("error"):
        return f"Result contains error field: {result['error']!r}"
    return None


def check_status_ok(result: Any, *, ok_values: Optional[list[Any]] = None) -> Optional[str]:
    """Verify a ``status`` or ``status_code`` field indicates success.

    Args:
        result: The tool's return value (dict).
        ok_values: Acceptable status values. Defaults to ``[200, "ok", "success", True]``.

    Returns:
        Failure message or None.
    """
    if not isinstance(result, dict):
        return None
    ok = ok_values or [200, "ok", "success", True, "OK", "200"]
    for key in ("status", "status_code", "statusCode"):
        if key in result and result[key] not in ok:
            return f"Result {key}={result[key]!r} is not in expected ok values {ok}"
    return None
