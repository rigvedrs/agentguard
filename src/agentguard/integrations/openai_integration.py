"""OpenAI function calling integration for agentguard.

Wraps a list of Python callables into OpenAI-compatible tool schemas,
applying agentguard protection to each one. The returned tool list can
be passed directly to ``client.chat.completions.create(tools=...)``.

Usage::

    from openai import OpenAI
    from agentguard.integrations import guard_openai_tools
    from agentguard import GuardConfig

    client = OpenAI()
    config = GuardConfig(validate_input=True, max_retries=2, record=True)

    guarded_tools = guard_openai_tools([search_web, query_db], config=config)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Find Python tutorials"}],
        tools=guarded_tools,
    )

    # Execute a tool call from the response
    from agentguard.integrations import execute_openai_tool_call
    tool_result = execute_openai_tool_call(response.choices[0].message.tool_calls[0],
                                           guarded_tools)
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Optional

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    import typing

    origin = getattr(annotation, "__origin__", None)

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}

    if origin is list:
        args = getattr(annotation, "__args__", None)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        schemas = [_python_type_to_json_schema(a) for a in args]
        none_type = type(None)
        if none_type in annotation.__args__:
            # Optional
            if len(schemas) == 1:
                return schemas[0]
            return {"anyOf": schemas + [{"type": "null"}]}
        return {"anyOf": schemas}

    # Pydantic model
    try:
        from pydantic import BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            schema = annotation.model_json_schema()
            return {"type": "object", "properties": schema.get("properties", {})}
    except ImportError:
        pass

    return {}


def function_to_openai_tool(fn: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python callable to an OpenAI function tool definition.

    Args:
        fn: The callable to introspect.

    Returns:
        Dict with ``{"type": "function", "function": {...}}`` structure.
    """
    # Unwrap guarded tools to get the original function
    original = getattr(fn, "__wrapped__", fn)
    sig = inspect.signature(original)
    hints: dict[str, Any] = {}
    try:
        import typing
        hints = typing.get_type_hints(original)
    except Exception:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        json_schema = _python_type_to_json_schema(hints.get(name, inspect.Parameter.empty))
        # Add description from docstring if available (simple heuristic)
        properties[name] = json_schema or {"type": "string"}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    docstring = (original.__doc__ or "").strip()
    description = docstring.splitlines()[0] if docstring else f"Call {original.__name__}"

    return {
        "type": "function",
        "function": {
            "name": original.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Main integration API
# ---------------------------------------------------------------------------


def guard_openai_tools(
    functions: list[Callable[..., Any]],
    config: Optional[GuardConfig] = None,
) -> list[dict[str, Any]]:
    """Wrap a list of callables with agentguard and generate OpenAI tool schemas.

    The original callables are wrapped with ``@guard`` (if not already) and
    the tool schemas are returned in OpenAI format.

    Args:
        functions: List of Python callables to guard and expose as tools.
        config: Optional :class:`~agentguard.core.types.GuardConfig` applied to
            all functions. Individual functions already wrapped with ``@guard``
            keep their own config.

    Returns:
        List of OpenAI-compatible tool schema dicts.
    """
    schemas: list[dict[str, Any]] = []
    for fn in functions:
        if not isinstance(fn, GuardedTool) and config is not None:
            fn = guard(fn, config=config)
        schemas.append(function_to_openai_tool(fn))
    return schemas


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class OpenAIToolExecutor:
    """Executes tool calls from an OpenAI API response.

    Maintains a registry of tool name → guarded callable and dispatches
    ``tool_calls`` from the model response to the appropriate function.

    Example::

        executor = OpenAIToolExecutor()
        executor.register(search_web, config=GuardConfig(max_retries=2))
        executor.register(query_db)

        # Pass executor.tools to the OpenAI API
        response = client.chat.completions.create(tools=executor.tools, ...)

        # Execute all tool calls from the response
        results = executor.execute_all(response.choices[0].message.tool_calls)
    """

    def __init__(self, config: Optional[GuardConfig] = None) -> None:
        self._config = config
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

    def register(
        self,
        fn: Callable[..., Any],
        config: Optional[GuardConfig] = None,
    ) -> "OpenAIToolExecutor":
        """Register a callable as an available tool.

        Args:
            fn: The callable to register.
            config: Per-tool guard config. Falls back to the executor's config.

        Returns:
            Self for chaining.
        """
        effective_config = config or self._config
        if not isinstance(fn, GuardedTool) and effective_config:
            fn = guard(fn, config=effective_config)
        name = getattr(fn, "__name__", fn.__class__.__name__)
        self._tools[name] = fn
        self._schemas.append(function_to_openai_tool(fn))
        return self

    @property
    def tools(self) -> list[dict[str, Any]]:
        """OpenAI-format tool schemas for the registered functions."""
        return list(self._schemas)

    def execute(self, tool_call: Any) -> str:
        """Execute a single OpenAI tool call object.

        Args:
            tool_call: An OpenAI ``ToolCall`` object with ``.function.name``
                and ``.function.arguments`` attributes.

        Returns:
            JSON string result suitable for inclusion in a follow-up message.
        """
        name = tool_call.function.name
        fn = self._tools.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name!r}"})
        try:
            kwargs = json.loads(tool_call.function.arguments or "{}")
            result = fn(**kwargs)
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc), "tool": name})

    def execute_all(self, tool_calls: list[Any]) -> list[dict[str, str]]:
        """Execute a list of tool calls and return formatted results.

        Args:
            tool_calls: List of OpenAI ``ToolCall`` objects.

        Returns:
            List of ``{"tool_call_id": ..., "role": "tool", "content": ...}`` dicts.
        """
        return [
            {
                "tool_call_id": tc.id,
                "role": "tool",
                "content": self.execute(tc),
            }
            for tc in tool_calls
        ]


def execute_openai_tool_call(
    tool_call: Any,
    tools: list[dict[str, Any]],
    registry: Optional[dict[str, Callable[..., Any]]] = None,
) -> str:
    """Execute a single OpenAI tool call using a pre-built tool list.

    Looks up the callable by name in the agentguard global registry.

    Args:
        tool_call: OpenAI ``ToolCall`` object.
        tools: The guarded tool list (used to verify the tool exists).
        registry: Optional override map of ``name → callable``.

    Returns:
        JSON string result.
    """
    name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid tool arguments JSON"})

    # Look up in provided registry first, then global
    fn: Optional[Callable[..., Any]] = None
    if registry:
        fn = registry.get(name)
    if fn is None:
        from agentguard.core.registry import global_registry
        reg = global_registry.get(name)
        if reg:
            fn = reg.guarded_func

    if fn is None:
        return json.dumps({"error": f"Tool '{name}' not found"})

    try:
        result = fn(**kwargs)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
