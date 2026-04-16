"""Input/output schema validation using type hints and Pydantic.

Validates function arguments against their declared type annotations and
validates return values against the function's return type annotation.
Supports both native Python types and Pydantic models.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, get_type_hints

from agentguard.core.types import ValidationResult, ValidatorKind


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_inputs(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[ValidationResult]:
    """Validate positional and keyword arguments against *func*'s type hints.

    Args:
        func: The original (unwrapped) callable whose annotations to use.
        args: Positional arguments being passed to the function.
        kwargs: Keyword arguments being passed to the function.

    Returns:
        List of :class:`~agentguard.core.types.ValidationResult` objects.
        An empty list means all checks passed.
    """
    results: list[ValidationResult] = []
    hints = _safe_get_hints(func)
    if not hints:
        return results

    sig = inspect.signature(func)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError as exc:
        results.append(ValidationResult(
            valid=False,
            kind=ValidatorKind.SCHEMA,
            message=f"Argument binding failed: {exc}",
        ))
        return results

    for param_name, value in bound.arguments.items():
        if param_name not in hints:
            continue
        expected_type = hints[param_name]
        vr = _check_type(param_name, value, expected_type)
        results.append(vr)

    return results


def validate_output(
    func: Callable[..., Any],
    return_value: Any,
) -> list[ValidationResult]:
    """Validate *return_value* against *func*'s return type annotation.

    Args:
        func: The callable whose return annotation to check against.
        return_value: The value the tool returned.

    Returns:
        List of :class:`~agentguard.core.types.ValidationResult` objects.
    """
    hints = _safe_get_hints(func)
    return_hint = hints.get("return")
    if return_hint is None:
        return []

    # Ignore None return annotations
    if return_hint is type(None):
        if return_value is not None:
            return [ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Expected None return, got {type(return_value).__name__}",
            )]
        return [ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)]

    return [_check_type("return", return_value, return_hint)]


# ---------------------------------------------------------------------------
# Type checking helpers
# ---------------------------------------------------------------------------


def _check_type(name: str, value: Any, expected: Any) -> ValidationResult:
    """Check that *value* is an instance of *expected* type.

    Handles Optional, Union, List, Dict, Tuple, generic types, and Pydantic models.
    """
    if expected is Any or expected is inspect.Parameter.empty:
        return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)

    origin = getattr(expected, "__origin__", None)

    # Optional[X] / Union[X, None]
    if origin is typing.Union:
        args = expected.__args__
        none_type = type(None)
        # Check None explicitly
        if value is None and none_type in args:
            return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)
        non_none_args = [a for a in args if a is not none_type]
        for arg in non_none_args:
            vr = _check_type(name, value, arg)
            if vr.valid:
                return vr
        return ValidationResult(
            valid=False,
            kind=ValidatorKind.SCHEMA,
            message=f"Parameter '{name}': expected {expected}, got {type(value).__name__!r}",
        )

    # list[X]
    if origin is list:
        if not isinstance(value, list):
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Parameter '{name}': expected list, got {type(value).__name__!r}",
            )
        if expected.__args__:
            item_type = expected.__args__[0]
            for i, item in enumerate(value):
                vr = _check_type(f"{name}[{i}]", item, item_type)
                if not vr.valid:
                    return vr
        return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)

    # dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Parameter '{name}': expected dict, got {type(value).__name__!r}",
            )
        return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)

    # tuple
    if origin is tuple:
        if not isinstance(value, tuple):
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Parameter '{name}': expected tuple, got {type(value).__name__!r}",
            )
        return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel
        if isinstance(expected, type) and issubclass(expected, BaseModel):
            if isinstance(value, expected):
                return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)
            if isinstance(value, dict):
                try:
                    expected.model_validate(value)
                    return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)
                except Exception as exc:
                    return ValidationResult(
                        valid=False,
                        kind=ValidatorKind.SCHEMA,
                        message=f"Parameter '{name}': Pydantic validation failed: {exc}",
                    )
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Parameter '{name}': expected {expected.__name__}, got {type(value).__name__!r}",
            )
    except ImportError:
        pass

    # Fallback: plain isinstance check
    if isinstance(expected, type):
        if not isinstance(value, expected):
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=(
                    f"Parameter '{name}': expected {expected.__name__}, "
                    f"got {type(value).__name__!r} (value={value!r})"
                ),
                details={"expected": expected.__name__, "actual": type(value).__name__},
            )
        return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)

    # Unrecognised generic — skip
    return ValidationResult(valid=True, kind=ValidatorKind.SCHEMA)


def _safe_get_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Return type hints, returning an empty dict on any error."""
    try:
        return get_type_hints(func)
    except Exception:
        return {}
