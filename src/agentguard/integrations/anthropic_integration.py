"""Anthropic tool use integration for agentguard.

Wraps Python callables into Anthropic-compatible tool schemas and provides
an executor for dispatching tool calls from Claude's responses.

Usage::

    import anthropic
    from agentguard.integrations import guard_anthropic_tools
    from agentguard import GuardConfig

    client = anthropic.Anthropic()
    config = GuardConfig(validate_input=True, max_retries=2)

    tools = guard_anthropic_tools([search_web, query_db], config=config)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "Find Python tutorials"}],
    )

    # Execute tool calls
    from agentguard.integrations import AnthropicToolExecutor
    executor = AnthropicToolExecutor()
    executor.register(search_web, config=config)
    results = executor.execute_all(response.content)
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Optional

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig
from agentguard.integrations.openai_integration import _python_type_to_json_schema


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def function_to_anthropic_tool(fn: Callable[..., Any]) -> dict[str, Any]:
    """Convert a callable to an Anthropic tool schema.

    Args:
        fn: The callable to introspect.

    Returns:
        Dict with ``{"name": ..., "description": ..., "input_schema": {...}}``
        structure suitable for the Anthropic ``tools`` parameter.
    """
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
        properties[name] = json_schema or {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    docstring = (original.__doc__ or "").strip()
    description = docstring.splitlines()[0] if docstring else f"Call {original.__name__}"

    return {
        "name": original.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def guard_anthropic_tools(
    functions: list[Callable[..., Any]],
    config: Optional[GuardConfig] = None,
) -> list[dict[str, Any]]:
    """Wrap callables with agentguard and generate Anthropic tool schemas.

    Args:
        functions: Callables to guard and expose as tools.
        config: Guard config applied to all functions.

    Returns:
        List of Anthropic-compatible tool schema dicts.
    """
    schemas: list[dict[str, Any]] = []
    for fn in functions:
        if not isinstance(fn, GuardedTool) and config is not None:
            fn = guard(fn, config=config)
        schemas.append(function_to_anthropic_tool(fn))
    return schemas


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class AnthropicToolExecutor:
    """Executes Anthropic tool_use blocks from a Claude API response.

    Example::

        executor = AnthropicToolExecutor()
        executor.register(search_web)
        executor.register(query_db)

        response = client.messages.create(tools=executor.tools, ...)
        results = executor.execute_all(response.content)
    """

    def __init__(self, config: Optional[GuardConfig] = None) -> None:
        self._config = config
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

    def register(
        self,
        fn: Callable[..., Any],
        config: Optional[GuardConfig] = None,
    ) -> "AnthropicToolExecutor":
        """Register a callable as a tool.

        Args:
            fn: The callable to register.
            config: Per-tool guard config.

        Returns:
            Self for chaining.
        """
        effective_config = config or self._config
        if not isinstance(fn, GuardedTool) and effective_config:
            fn = guard(fn, config=effective_config)
        name = getattr(fn, "__name__", fn.__class__.__name__)
        self._tools[name] = fn
        self._schemas.append(function_to_anthropic_tool(fn))
        return self

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Anthropic-format tool schemas."""
        return list(self._schemas)

    def execute(self, tool_use_block: Any) -> dict[str, Any]:
        """Execute a single Anthropic tool_use block.

        Args:
            tool_use_block: An Anthropic ``ToolUseBlock`` with ``.name``,
                ``.id``, and ``.input`` attributes.

        Returns:
            A ``tool_result`` content block dict.
        """
        name = tool_use_block.name
        tool_id = tool_use_block.id
        fn = self._tools.get(name)
        if fn is None:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps({"error": f"Unknown tool: {name!r}"}),
                "is_error": True,
            }
        try:
            kwargs = tool_use_block.input or {}
            result = fn(**kwargs)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result, default=str),
            }
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps({"error": str(exc)}),
                "is_error": True,
            }

    def execute_all(self, content_blocks: list[Any]) -> list[dict[str, Any]]:
        """Execute all tool_use blocks from a response.

        Args:
            content_blocks: The ``response.content`` list from an Anthropic response.

        Returns:
            List of ``tool_result`` content block dicts.
        """
        results = []
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                results.append(self.execute(block))
        return results
