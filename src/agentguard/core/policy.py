"""Policy-as-Code: define agentguard configurations in YAML or TOML files.

This module allows teams to store guard configurations in version-controlled
policy files rather than scattering them across application code.

Supported formats
-----------------
* **YAML** — ``agentguard.yaml`` / ``agentguard.yml`` (requires PyYAML,
  which is an optional dependency).
* **TOML** — ``agentguard.toml`` (uses stdlib ``tomllib`` on Python ≥ 3.11).

Policy file format (YAML example)::

    version: "1"
    defaults:
      validate_input: true
      max_retries: 2
      record: true
      trace_dir: "./traces"

    tools:
      search_web:
        timeout: 10.0
        rate_limit:
          calls_per_minute: 60
          burst: 10
        hallucination:
          expected_latency_ms: [100, 5000]
          required_fields: ["results", "total"]

      query_database:
        timeout: 30.0
        budget:
          max_cost_per_session: 1.00
          cost_per_call: 0.05
        circuit_breaker:
          failure_threshold: 5
          recovery_timeout: 60

Public API
----------
* :func:`load_policy` — parse a policy file into ``dict[tool_name, GuardConfig]``.
* :func:`apply_policy` — decorate a list of functions with their matching configs.
* :func:`validate_policy` — check a policy file for structural errors.

CLI commands (registered in :mod:`agentguard.cli.main`)::

    agentguard policy validate agentguard.yaml
    agentguard policy apply    agentguard.yaml
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from agentguard.core.types import (
    BudgetConfig,
    CircuitBreakerConfig,
    GuardAction,
    GuardConfig,
    RateLimitConfig,
    RetryConfig,
    TimeoutConfig,
)

__all__ = [
    "load_policy",
    "apply_policy",
    "validate_policy",
    "PolicyError",
    "PolicyValidationError",
    "policy_summary",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PolicyError(Exception):
    """Base class for policy-related errors."""


class PolicyValidationError(PolicyError):
    """Raised when a policy file fails structural validation.

    Attributes:
        errors: List of human-readable error descriptions.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Policy validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_policy(path: str | Path) -> dict[str, GuardConfig]:
    """Parse a policy file and return a mapping of tool name → :class:`~agentguard.core.types.GuardConfig`.

    Args:
        path: Path to the policy file.  Must end with ``.yaml``, ``.yml``,
            or ``.toml``.

    Returns:
        A dict where each key is a tool name and the value is the fully
        resolved :class:`~agentguard.core.types.GuardConfig` for that tool
        (defaults merged with per-tool overrides).

    Raises:
        FileNotFoundError: If the file does not exist.
        PolicyError: If the file cannot be parsed or has an unsupported format.
        PolicyValidationError: If the file fails structural validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    raw = _parse_file(path)
    errors = _validate_raw(raw)
    if errors:
        raise PolicyValidationError(errors)

    return _build_configs(raw)


def apply_policy(
    policy: dict[str, GuardConfig],
    tools: list[Callable[..., Any]],
    *,
    missing_ok: bool = True,
) -> dict[str, Any]:
    """Wrap each function in *tools* with its matching policy config.

    Tools that have no matching entry in *policy* are left unwrapped (when
    ``missing_ok=True``) or raise a :class:`PolicyError` (when
    ``missing_ok=False``).

    Args:
        policy: Mapping returned by :func:`load_policy`.
        tools: List of callables to guard.
        missing_ok: If ``False``, raise :class:`PolicyError` for any tool
            that has no entry in *policy*.

    Returns:
        A dict of ``{tool_name: guarded_tool}``.  Tools without a matching
        policy entry appear under their original name with their original
        callable.

    Raises:
        PolicyError: If ``missing_ok=False`` and a tool has no policy entry.
    """
    from agentguard.core.guard import GuardedTool

    result: dict[str, Any] = {}
    for fn in tools:
        name = getattr(fn, "__name__", repr(fn))
        if name in policy:
            guarded = GuardedTool(fn, config=policy[name])
            result[name] = guarded
        elif not missing_ok:
            raise PolicyError(
                f"Tool '{name}' has no entry in the policy.  "
                "Pass missing_ok=True to skip unmatched tools."
            )
        else:
            result[name] = fn
    return result


def validate_policy(path: str | Path) -> list[str]:
    """Validate a policy file and return a list of error strings.

    An empty list means the policy is valid.

    Args:
        path: Path to the policy file.

    Returns:
        A list of human-readable error descriptions (empty = valid).

    Raises:
        FileNotFoundError: If the file does not exist.
        PolicyError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    try:
        raw = _parse_file(path)
    except PolicyError as exc:
        return [str(exc)]

    return _validate_raw(raw)


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def _parse_file(path: Path) -> dict[str, Any]:
    """Load *path* into a raw Python dict, dispatching by extension."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _parse_yaml(path)
    if suffix == ".toml":
        return _parse_toml(path)
    raise PolicyError(
        f"Unsupported policy file format: '{suffix}'.  "
        "Use .yaml, .yml, or .toml."
    )


def _parse_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML policy file."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return _parse_simple_yaml(path)

    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise PolicyError(f"Failed to parse YAML policy file: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PolicyError("Policy file must be a YAML mapping at the top level.")
    return data


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse a small YAML subset used by agentguard policy files."""
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise PolicyError(f"Unsupported YAML syntax in policy file: {raw_line.strip()}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_yaml_scalar(value)

    return root


def _parse_yaml_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return ast.literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception as exc:
            raise PolicyError(f"Unsupported YAML list value: {value}") from exc
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML policy file."""
    # tomllib (stdlib) reads bytes; tomli (third-party) has same API.
    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[import]
    else:
        try:
            import tomllib  # type: ignore[import]
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[import]
            except ImportError as exc:
                raise PolicyError(
                    "tomllib (Python ≥ 3.11) or tomli is required to load "
                    "TOML policy files.  Install tomli with: pip install tomli"
                ) from exc

    with path.open("rb") as fh:
        try:
            return tomllib.load(fh)
        except Exception as exc:
            raise PolicyError(f"Failed to parse TOML policy file: {exc}") from exc


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

_VALID_TOP_KEYS = {"version", "defaults", "tools"}
_VALID_GUARD_KEYS = {
    "validate_input",
    "validate_output",
    "detect_hallucination",
    "max_retries",
    "timeout",
    "budget",
    "rate_limit",
    "circuit_breaker",
    "record",
    "trace_dir",
    "trace_backend",
    "trace_db_path",
    "session_id",
    "hallucination",  # convenience alias for hallucination config
    "retry",
}
_VALID_BUDGET_KEYS = {
    "max_cost_per_call",
    "max_cost_per_session",
    "max_calls_per_session",
    "alert_threshold",
    "on_exceed",
    "cost_per_call",
    "use_dynamic_llm_costs",
    "model_pricing_overrides",
    "record_llm_spend",
}
_VALID_RATE_LIMIT_KEYS = {
    "calls_per_second",
    "calls_per_minute",
    "calls_per_hour",
    "burst",
    "on_limit",
}
_VALID_CIRCUIT_BREAKER_KEYS = {
    "failure_threshold",
    "recovery_timeout",
    "success_threshold",
    "on_open",
}
_VALID_RETRY_KEYS = {
    "max_retries",
    "initial_delay",
    "max_delay",
    "backoff_factor",
    "jitter",
}


def _validate_raw(raw: dict[str, Any]) -> list[str]:
    """Return a list of error strings for a parsed policy dict."""
    errors: list[str] = []

    # Top-level keys
    unknown = set(raw) - _VALID_TOP_KEYS
    if unknown:
        errors.append(f"Unknown top-level keys: {sorted(unknown)}")

    # Version check
    version = raw.get("version")
    if version is not None and str(version) not in {"1", "1.0"}:
        errors.append(f"Unsupported policy version: {version!r}.  Only '1' is supported.")

    # defaults section
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        errors.append("'defaults' must be a mapping.")
    else:
        errors.extend(_validate_guard_section(defaults, context="defaults"))

    # tools section
    tools = raw.get("tools", {})
    if not isinstance(tools, dict):
        errors.append("'tools' must be a mapping of tool_name -> config.")
    else:
        for tool_name, tool_cfg in tools.items():
            if not isinstance(tool_cfg, dict):
                errors.append(f"tools.{tool_name}: value must be a mapping.")
                continue
            errors.extend(
                _validate_guard_section(tool_cfg, context=f"tools.{tool_name}")
            )

    return errors


def _validate_guard_section(cfg: dict[str, Any], *, context: str) -> list[str]:
    errors: list[str] = []
    unknown = set(cfg) - _VALID_GUARD_KEYS
    if unknown:
        errors.append(f"{context}: unknown keys {sorted(unknown)}")

    # Sub-sections
    for sub_key, valid_keys, label in [
        ("budget", _VALID_BUDGET_KEYS, "budget"),
        ("rate_limit", _VALID_RATE_LIMIT_KEYS, "rate_limit"),
        ("circuit_breaker", _VALID_CIRCUIT_BREAKER_KEYS, "circuit_breaker"),
        ("retry", _VALID_RETRY_KEYS, "retry"),
    ]:
        sub = cfg.get(sub_key)
        if sub is not None:
            if not isinstance(sub, dict):
                errors.append(f"{context}.{label}: must be a mapping.")
                continue
            unknown_sub = set(sub) - valid_keys
            if unknown_sub:
                errors.append(
                    f"{context}.{label}: unknown keys {sorted(unknown_sub)}"
                )

    # timeout must be numeric
    timeout = cfg.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        errors.append(f"{context}.timeout: must be a number.")

    # max_retries must be integer
    mr = cfg.get("max_retries")
    if mr is not None and not isinstance(mr, int):
        errors.append(f"{context}.max_retries: must be an integer.")

    return errors


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def _build_configs(raw: dict[str, Any]) -> dict[str, GuardConfig]:
    """Merge defaults with per-tool overrides and build GuardConfig objects."""
    defaults_raw = raw.get("defaults", {}) or {}
    tools_raw = raw.get("tools", {}) or {}

    configs: dict[str, GuardConfig] = {}
    for tool_name, tool_raw in tools_raw.items():
        merged = {**defaults_raw, **tool_raw}
        configs[tool_name] = _raw_to_guard_config(merged)

    return configs


def _raw_to_guard_config(raw: dict[str, Any]) -> GuardConfig:
    """Convert a raw (merged defaults + tool) dict into a :class:`~agentguard.core.types.GuardConfig`."""
    kwargs: dict[str, Any] = {}

    # Booleans / scalars
    for simple_key in (
        "validate_input",
        "validate_output",
        "detect_hallucination",
        "max_retries",
        "record",
        "trace_dir",
        "trace_backend",
        "trace_db_path",
        "session_id",
    ):
        if simple_key in raw:
            kwargs[simple_key] = raw[simple_key]

    # timeout — can be bare float or via TimeoutConfig
    if "timeout" in raw:
        kwargs["timeout"] = float(raw["timeout"])

    # rate_limit
    rl_raw = raw.get("rate_limit")
    if rl_raw:
        kwargs["rate_limit"] = _build_rate_limit(rl_raw)

    # budget
    budget_raw = raw.get("budget")
    if budget_raw:
        kwargs["budget"] = _build_budget(budget_raw)

    # circuit_breaker
    cb_raw = raw.get("circuit_breaker")
    if cb_raw:
        kwargs["circuit_breaker"] = _build_circuit_breaker(cb_raw)

    # retry
    retry_raw = raw.get("retry")
    if retry_raw:
        kwargs["retry"] = _build_retry(retry_raw)

    return GuardConfig(**kwargs)


def _build_rate_limit(raw: dict[str, Any]) -> RateLimitConfig:
    kwargs: dict[str, Any] = {}
    for key in ("calls_per_second", "calls_per_minute", "calls_per_hour", "burst"):
        if key in raw:
            kwargs[key] = raw[key]
    if "on_limit" in raw:
        kwargs["on_limit"] = GuardAction(raw["on_limit"])
    return RateLimitConfig(**kwargs)


def _build_budget(raw: dict[str, Any]) -> BudgetConfig:
    kwargs: dict[str, Any] = {}
    for key in (
        "max_cost_per_call",
        "max_cost_per_session",
        "max_calls_per_session",
        "alert_threshold",
        "cost_per_call",
        "use_dynamic_llm_costs",
        "model_pricing_overrides",
        "record_llm_spend",
    ):
        if key in raw:
            kwargs[key] = raw[key]
    if "on_exceed" in raw:
        kwargs["on_exceed"] = GuardAction(raw["on_exceed"])
    return BudgetConfig(**kwargs)


def _build_circuit_breaker(raw: dict[str, Any]) -> CircuitBreakerConfig:
    kwargs: dict[str, Any] = {}
    for key in ("failure_threshold", "recovery_timeout", "success_threshold"):
        if key in raw:
            kwargs[key] = raw[key]
    if "on_open" in raw:
        kwargs["on_open"] = GuardAction(raw["on_open"])
    return CircuitBreakerConfig(**kwargs)


def _build_retry(raw: dict[str, Any]) -> RetryConfig:
    kwargs: dict[str, Any] = {}
    for key in ("max_retries", "initial_delay", "max_delay", "backoff_factor", "jitter"):
        if key in raw:
            kwargs[key] = raw[key]
    return RetryConfig(**kwargs)


# ---------------------------------------------------------------------------
# Human-readable policy summary (used by CLI)
# ---------------------------------------------------------------------------


def policy_summary(
    policy: dict[str, GuardConfig],
    *,
    file_path: Optional[str | Path] = None,
) -> str:
    """Return a human-readable summary of a loaded policy.

    Args:
        policy: Dict returned by :func:`load_policy`.
        file_path: Optional path to include in the header.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    lines: list[str] = []
    header = f"Policy: {file_path}" if file_path else "Policy summary"
    lines.append(header)
    lines.append("=" * len(header))
    if not policy:
        lines.append("  (no tool configurations)")
        return "\n".join(lines)
    for tool_name, cfg in sorted(policy.items()):
        lines.append(f"\n  [{tool_name}]")
        if cfg.timeout is not None:
            lines.append(f"    timeout          = {cfg.timeout}s")
        if cfg.max_retries:
            lines.append(f"    max_retries      = {cfg.max_retries}")
        if cfg.validate_input:
            lines.append(f"    validate_input   = true")
        if cfg.validate_output:
            lines.append(f"    validate_output  = true")
        if cfg.record:
            detail = f"backend={cfg.trace_backend!r}"
            if cfg.trace_backend == "sqlite":
                detail += f", db={cfg.trace_db_path or cfg.trace_dir!r}"
            else:
                detail += f", dir={cfg.trace_dir!r}"
            lines.append(f"    record           = true  ({detail})")
        if cfg.rate_limit:
            rl = cfg.rate_limit
            parts = []
            if rl.calls_per_second:
                parts.append(f"{rl.calls_per_second}/s")
            if rl.calls_per_minute:
                parts.append(f"{rl.calls_per_minute}/min")
            if rl.calls_per_hour:
                parts.append(f"{rl.calls_per_hour}/hr")
            lines.append(f"    rate_limit       = {', '.join(parts) or 'configured'} (burst={rl.burst})")
        if cfg.budget:
            b = cfg.budget
            parts = []
            if b.max_cost_per_session is not None:
                parts.append(f"session=${b.max_cost_per_session:.2f}")
            if b.max_cost_per_call is not None:
                parts.append(f"call=${b.max_cost_per_call:.2f}")
            if b.max_calls_per_session is not None:
                parts.append(f"calls={b.max_calls_per_session}")
            lines.append(f"    budget           = {', '.join(parts) or 'configured'}")
        if cfg.circuit_breaker:
            cb = cfg.circuit_breaker
            lines.append(
                f"    circuit_breaker  = threshold={cb.failure_threshold}, "
                f"recovery={cb.recovery_timeout}s"
            )
    return "\n".join(lines)
