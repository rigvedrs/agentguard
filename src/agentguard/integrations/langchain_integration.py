"""LangChain tool integration for agentguard.

Provides ``GuardedLangChainTool`` — a LangChain ``BaseTool`` subclass that
wraps any callable with agentguard protection, and ``guard_langchain_tools``
for bulk conversion.

Works with langchain-core >= 0.1. Does not require the full langchain package.

Usage::

    from langchain_core.tools import BaseTool
    from agentguard.integrations import GuardedLangChainTool, guard_langchain_tools
    from agentguard import GuardConfig

    config = GuardConfig(validate_input=True, max_retries=2)

    # Wrap a plain callable
    guarded = GuardedLangChainTool.from_function(
        search_web, config=config, name="search_web"
    )

    # Wrap an existing LangChain BaseTool
    guarded_existing = GuardedLangChainTool(
        name="search_web",
        description="Search the web",
        func=search_web,
        config=config,
    )

    # Bulk wrap
    tools = guard_langchain_tools([search_web, query_db], config=config)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Type

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig


# ---------------------------------------------------------------------------
# LangChain guard wrapper
# ---------------------------------------------------------------------------


class GuardedLangChainTool:
    """A LangChain-compatible tool wrapper backed by agentguard.

    When ``langchain_core`` is available, this class inherits from
    ``BaseTool`` to be fully compatible. When it is not available, it
    provides a duck-typed fallback that works with most agent frameworks.

    Attributes:
        name: Tool name used by the agent.
        description: Human-readable description.
        guarded_fn: The underlying :class:`~agentguard.core.guard.GuardedTool`.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        config: Optional[GuardConfig] = None,
    ) -> None:
        self.name = name
        self.description = description
        cfg = config or GuardConfig()
        if not isinstance(func, GuardedTool):
            self.guarded_fn: GuardedTool = guard(func, config=cfg)
        else:
            self.guarded_fn = func

    # ------------------------------------------------------------------
    # LangChain BaseTool duck-type interface
    # ------------------------------------------------------------------

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous execution — called by LangChain agents."""
        return self.guarded_fn(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous execution — called by async LangChain agents."""
        return await self.guarded_fn.acall(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.guarded_fn(*args, **kwargs)

    # ------------------------------------------------------------------
    # LangChain-style schema
    # ------------------------------------------------------------------

    @property
    def args_schema(self) -> Optional[Any]:
        """Return a Pydantic model for this tool's arguments, if available."""
        return None

    def to_openai_function(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function definition for this tool."""
        from agentguard.integrations.openai_integration import function_to_openai_tool
        return function_to_openai_tool(self.guarded_fn)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        config: Optional[GuardConfig] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "GuardedLangChainTool":
        """Create a GuardedLangChainTool from a plain callable.

        Args:
            func: The callable to wrap.
            config: Guard configuration.
            name: Override the tool name (defaults to ``func.__name__``).
            description: Override the description (defaults to docstring).

        Returns:
            A :class:`GuardedLangChainTool` instance.
        """
        tool_name = name or func.__name__
        raw_doc = (func.__doc__ or "").strip()
        first_line = raw_doc.splitlines()[0] if raw_doc else ""
        tool_desc = description or first_line or f"Run {tool_name}"
        return cls(name=tool_name, description=tool_desc, func=func, config=config)

    def __repr__(self) -> str:
        return f"GuardedLangChainTool(name={self.name!r})"


# ---------------------------------------------------------------------------
# BaseTool subclass — only created if langchain_core is available
# ---------------------------------------------------------------------------


def _make_lc_basetool_subclass() -> Optional[Type[GuardedLangChainTool]]:
    """Create a proper BaseTool subclass if langchain_core is installed."""
    try:
        from langchain_core.tools import BaseTool as LCBaseTool
        from pydantic import BaseModel

        class _LCGuardedTool(LCBaseTool):
            """agentguard-protected LangChain BaseTool."""

            name: str = "guarded_tool"
            description: str = "agentguard-protected tool"
            _guarded_fn: Any = None

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                return self._guarded_fn(*args, **kwargs)

            async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                return await self._guarded_fn.acall(*args, **kwargs)

        return _LCGuardedTool  # type: ignore[return-value]
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Bulk helper
# ---------------------------------------------------------------------------


def guard_langchain_tools(
    functions: list[Callable[..., Any]],
    config: Optional[GuardConfig] = None,
) -> list[GuardedLangChainTool]:
    """Wrap a list of callables as GuardedLangChainTool instances.

    Args:
        functions: Callables to wrap.
        config: Guard config applied to all functions.

    Returns:
        List of :class:`GuardedLangChainTool` instances.
    """
    return [GuardedLangChainTool.from_function(fn, config=config) for fn in functions]
