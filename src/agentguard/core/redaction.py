"""Helpers for redacting sensitive values before persistence or logging."""

from __future__ import annotations

import re
from typing import Any, Iterable

from agentguard.core.types import ToolCall

REDACTED = "[REDACTED]"

DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "pwd",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "session_key",
    "access_key",
    "private_key",
    "client_secret",
)

_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}\b", re.IGNORECASE),
)


def sensitive_field_names(extra_fields: Iterable[str] = ()) -> tuple[str, ...]:
    """Return the normalized set of field names treated as sensitive."""
    normalized = {field.strip().lower() for field in DEFAULT_REDACT_FIELDS}
    normalized.update(field.strip().lower() for field in extra_fields if field.strip())
    return tuple(sorted(normalized))


def is_sensitive_field(field_name: str | None, *, extra_fields: Iterable[str] = ()) -> bool:
    """Return whether *field_name* should be redacted."""
    if not field_name:
        return False
    candidate = field_name.strip().lower()
    return any(name in candidate for name in sensitive_field_names(extra_fields))


def sanitize_value(value: Any, *, field_name: str | None = None, extra_fields: Iterable[str] = ()) -> Any:
    """Recursively redact secret-like values from arbitrary data structures."""
    if is_sensitive_field(field_name, extra_fields=extra_fields):
        return REDACTED

    if isinstance(value, dict):
        return {
            key: sanitize_value(
                item,
                field_name=str(key),
                extra_fields=extra_fields,
            )
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(sanitize_value(item, extra_fields=extra_fields) for item in value)

    if isinstance(value, list):
        return [sanitize_value(item, extra_fields=extra_fields) for item in value]

    if isinstance(value, set):
        return {sanitize_value(item, extra_fields=extra_fields) for item in value}

    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _TOKEN_PATTERNS):
            return REDACTED

    return value


def sanitize_tool_call(call: ToolCall, *, extra_fields: Iterable[str] = ()) -> ToolCall:
    """Return a copy of *call* with secret-like data redacted."""
    return call.model_copy(
        update={
            "args": sanitize_value(call.args, extra_fields=extra_fields),
            "kwargs": sanitize_value(call.kwargs, extra_fields=extra_fields),
            "metadata": sanitize_value(call.metadata, extra_fields=extra_fields),
        }
    )
