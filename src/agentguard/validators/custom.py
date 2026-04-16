"""Custom validator support.

Allows users to define their own validators and plug them into the guard
pipeline. Custom validators follow a simple protocol: they receive a
:class:`~agentguard.core.types.ToolCall` and an optional return value,
and produce a :class:`~agentguard.core.types.ValidationResult`.

Example::

    from agentguard.validators.custom import CustomValidator, validator_fn
    from agentguard.core.types import ValidationResult, ValidatorKind

    # Using the decorator
    @validator_fn(name="no_sql_injection")
    def check_no_injection(call, result=None) -> ValidationResult:
        sql = call.kwargs.get("sql", "")
        forbidden = ["DROP", "DELETE", "--", ";--"]
        for token in forbidden:
            if token.upper() in sql.upper():
                return ValidationResult(
                    valid=False,
                    kind=ValidatorKind.CUSTOM,
                    message=f"Potential SQL injection detected: {token!r}",
                )
        return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

    # Register with a guard
    from agentguard import guard
    @guard(custom_validators=[check_no_injection])
    def query_db(sql: str) -> list[dict]:
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind


# ---------------------------------------------------------------------------
# Protocol / type alias
# ---------------------------------------------------------------------------

#: Type signature for a custom validator callable.
ValidatorCallable = Callable[..., ValidationResult]


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------


def validator_fn(
    name: Optional[str] = None,
) -> Callable[[ValidatorCallable], ValidatorCallable]:
    """Decorator that marks a function as an agentguard custom validator.

    Decorated functions must accept ``(call: ToolCall, result: Any = None)``
    and return a :class:`~agentguard.core.types.ValidationResult`.

    Args:
        name: Optional display name for the validator. Defaults to
            ``func.__name__``.

    Returns:
        Decorator that annotates and returns the function unchanged.
    """
    def decorator(fn: ValidatorCallable) -> ValidatorCallable:
        fn._agentguard_validator = True  # type: ignore[attr-defined]
        fn._agentguard_name = name or fn.__name__  # type: ignore[attr-defined]
        return fn
    return decorator


# ---------------------------------------------------------------------------
# CustomValidator class
# ---------------------------------------------------------------------------


class CustomValidator:
    """Wraps a callable into a named, reusable custom validator.

    Example::

        from agentguard.validators.custom import CustomValidator

        def my_check(call, result=None):
            if not result:
                return ValidationResult(valid=False, kind=ValidatorKind.CUSTOM,
                                        message="Empty result")
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        cv = CustomValidator("non_empty_result", my_check)
        vr = cv.validate(call, result=some_value)
    """

    def __init__(
        self,
        name: str,
        fn: ValidatorCallable,
        *,
        description: str = "",
        apply_to: Optional[list[str]] = None,
    ) -> None:
        """Initialise the validator wrapper.

        Args:
            name: Identifier for this validator.
            fn: The validation callable ``(call, result=None) -> ValidationResult``.
            description: Human-readable description.
            apply_to: Optional list of tool names this validator applies to.
                If None, it applies to all tools.
        """
        self.name = name
        self._fn = fn
        self.description = description
        self.apply_to = apply_to

    def validate(
        self,
        call: ToolCall,
        result: Any = None,
    ) -> ValidationResult:
        """Run the validator against *call* and optional *result*.

        Args:
            call: The tool call being validated.
            result: The tool's return value (None for input-only validators).

        Returns:
            :class:`~agentguard.core.types.ValidationResult`.
        """
        if self.apply_to and call.tool_name not in self.apply_to:
            return ValidationResult(
                valid=True,
                kind=ValidatorKind.CUSTOM,
                message=f"Validator '{self.name}' not applicable to '{call.tool_name}'",
            )
        try:
            return self._fn(call, result=result)
        except Exception as exc:
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.CUSTOM,
                message=f"Validator '{self.name}' raised an exception: {exc}",
                details={"exception": str(exc), "validator": self.name},
            )

    def __repr__(self) -> str:
        scope = f"tools={self.apply_to}" if self.apply_to else "all tools"
        return f"CustomValidator(name={self.name!r}, scope={scope!r})"


# ---------------------------------------------------------------------------
# Built-in custom validators
# ---------------------------------------------------------------------------


@validator_fn(name="no_empty_string_args")
def no_empty_string_args(call: ToolCall, result: Any = None) -> ValidationResult:
    """Reject calls where any string argument is empty or whitespace-only.

    This catches common bugs where an LLM passes an empty string for a
    required parameter.
    """
    for key, val in call.kwargs.items():
        if isinstance(val, str) and not val.strip():
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.CUSTOM,
                message=f"Parameter '{key}' is an empty string",
                details={"param": key},
            )
    for i, val in enumerate(call.args):
        if isinstance(val, str) and not val.strip():
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.CUSTOM,
                message=f"Positional argument {i} is an empty string",
                details={"position": i},
            )
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)


@validator_fn(name="no_none_required_kwargs")
def no_none_required_kwargs(call: ToolCall, result: Any = None) -> ValidationResult:
    """Reject calls where a keyword argument is unexpectedly None.

    Useful for catching LLMs that omit required values.
    """
    none_keys = [k for k, v in call.kwargs.items() if v is None]
    if none_keys:
        return ValidationResult(
            valid=False,
            kind=ValidatorKind.CUSTOM,
            message=f"Keyword argument(s) are None: {none_keys}",
            details={"none_params": none_keys},
        )
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)


def run_custom_validators(
    validators: list[Any],
    call: ToolCall,
    result: Any = None,
) -> list[ValidationResult]:
    """Run a list of custom validators and return all results.

    Accepts both :class:`CustomValidator` instances and raw callables.

    Args:
        validators: List of validators (``CustomValidator`` or callable).
        call: The tool call to validate.
        result: The tool's return value (optional).

    Returns:
        List of :class:`~agentguard.core.types.ValidationResult`.
    """
    outputs: list[ValidationResult] = []
    for v in validators:
        if isinstance(v, CustomValidator):
            outputs.append(v.validate(call, result=result))
        elif callable(v):
            try:
                vr = v(call, result=result)
                if isinstance(vr, ValidationResult):
                    outputs.append(vr)
                else:
                    outputs.append(ValidationResult(
                        valid=bool(vr),
                        kind=ValidatorKind.CUSTOM,
                        message=str(vr) if not bool(vr) else "",
                    ))
            except Exception as exc:
                outputs.append(ValidationResult(
                    valid=False,
                    kind=ValidatorKind.CUSTOM,
                    message=f"Custom validator raised: {exc}",
                ))
    return outputs
