"""CrewAI integration for agentguard.

Provides ``GuardedCrewAITool`` — a wrapper that applies agentguard protection
to CrewAI tools, and ``guard_crewai_tools`` for bulk conversion.

CrewAI is an optional dependency. If it is not installed, the module can still
be imported; it simply won't provide CrewAI-specific base class inheritance.

Supported tool styles
---------------------
1. Functions decorated with ``@crewai.tools.tool``::

    from crewai.tools import tool

    @tool("Search Web")
    def search_web(query: str) -> str:
        '''Search the web for information.'''
        return requests.get(f"https://...?q={query}").text

2. Subclasses of ``crewai.tools.BaseTool``::

    from crewai.tools import BaseTool

    class SearchTool(BaseTool):
        name: str = "Search Web"
        description: str = "Searches the web."

        def _run(self, query: str) -> str:
            return requests.get(f"https://...?q={query}").text

Usage::

    from agentguard.integrations.crewai_integration import (
        GuardedCrewAITool,
        guard_crewai_tools,
    )
    from agentguard import GuardConfig

    config = GuardConfig(
        validate_input=True,
        detect_hallucination=True,
        max_retries=2,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    )

    # Wrap a list of tools
    guarded = guard_crewai_tools([search_web, query_db], config=config)

    # Wrap a single tool
    guarded_search = GuardedCrewAITool(search_web, config=config)
    result = guarded_search.run(query="Python tutorials")
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, Union

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig


# ---------------------------------------------------------------------------
# Optional CrewAI import
# ---------------------------------------------------------------------------

try:
    from crewai.tools import BaseTool as CrewAIBaseTool  # type: ignore[import]

    _CREWAI_AVAILABLE = True
except ImportError:
    CrewAIBaseTool = None  # type: ignore[assignment,misc]
    _CREWAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_tool_metadata(tool: Any) -> tuple[str, str]:
    """Extract (name, description) from a CrewAI tool or plain callable.

    Handles:
    - Functions decorated with @crewai.tools.tool (have .name / .description)
    - BaseTool subclass instances (.name / .description attributes)
    - Plain Python callables (use __name__ and __doc__)

    Returns:
        A (name, description) tuple.
    """
    # CrewAI @tool decorated functions and BaseTool instances expose .name
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or "unnamed_tool"
    desc = getattr(tool, "description", None)

    if not desc:
        raw_doc = (getattr(tool, "__doc__", "") or "").strip()
        desc = raw_doc.splitlines()[0] if raw_doc else f"Run {name}"

    return str(name), str(desc)


def _extract_callable(tool: Any) -> Callable[..., Any]:
    """Return the underlying callable from a CrewAI tool object.

    For BaseTool subclasses the callable is the ``_run`` method. For
    @tool-decorated functions it is usually the function itself (CrewAI stores
    it under ``func`` attribute in newer versions, or the object is itself
    callable).

    Args:
        tool: A CrewAI tool object or plain callable.

    Returns:
        A plain callable.
    """
    # BaseTool subclass instance — detected by presence of _run + name attrs
    # Works both when crewai is installed (isinstance check) and when it isn't
    # (duck-typed objects that have name/description/_run).
    if _CREWAI_AVAILABLE and CrewAIBaseTool is not None:
        try:
            if isinstance(tool, CrewAIBaseTool):
                return tool._run  # type: ignore[attr-defined]
        except TypeError:
            pass  # isinstance check failed on mock/metaclass edge cases

    # Duck-typed BaseTool: has _run but is not a plain function
    if (
        hasattr(tool, "_run")
        and hasattr(tool, "name")
        and hasattr(tool, "description")
        and not inspect.isfunction(tool)
        and not inspect.ismethod(tool)
    ):
        return tool._run  # type: ignore[attr-defined]

    # @tool-decorated function — CrewAI stores the original under .func
    if hasattr(tool, "func") and callable(tool.func):
        return tool.func  # type: ignore[attr-defined]

    # Plain callable or already a GuardedTool
    if callable(tool):
        return tool

    raise TypeError(
        f"Cannot extract a callable from {tool!r}. "
        "Expected a @crewai.tools.tool decorated function, "
        "a BaseTool subclass instance, or a plain callable."
    )


# ---------------------------------------------------------------------------
# GuardedCrewAITool
# ---------------------------------------------------------------------------


class GuardedCrewAITool:
    """An agentguard-protected wrapper for CrewAI tools.

    Accepts either a ``@crewai.tools.tool`` decorated function, a
    ``BaseTool`` subclass instance, or any plain callable. The wrapper
    preserves the original tool's name and description so it can be
    dropped into a CrewAI crew without modification.

    When ``crewai`` is installed, :meth:`to_crewai_tool` returns a proper
    ``BaseTool`` subclass that CrewAI agents can use natively.

    Attributes:
        name: Tool name forwarded from the original tool.
        description: Description forwarded from the original tool.
        guarded_fn: The underlying :class:`~agentguard.core.guard.GuardedTool`.

    Example::

        from agentguard.integrations.crewai_integration import GuardedCrewAITool
        from agentguard import GuardConfig

        config = GuardConfig(validate_input=True, max_retries=2)
        guarded = GuardedCrewAITool(search_web, config=config)

        # Use directly
        result = guarded.run(query="Python tutorials")

        # Or convert to a native CrewAI BaseTool
        crewai_tool = guarded.to_crewai_tool()
    """

    def __init__(
        self,
        tool: Any,
        config: Optional[GuardConfig] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialise the guarded wrapper.

        Args:
            tool: The CrewAI tool to protect. May be a ``@tool``-decorated
                function, a ``BaseTool`` instance, or a plain callable.
            config: Guard configuration. Defaults to a zero-config
                :class:`~agentguard.core.types.GuardConfig`.
            name: Override the tool name. Defaults to the tool's own name.
            description: Override the description. Defaults to the tool's own
                description or docstring.
        """
        cfg = config or GuardConfig()
        detected_name, detected_desc = _extract_tool_metadata(tool)
        self.name: str = name or detected_name
        self.description: str = description or detected_desc
        self._original_tool = tool

        fn = _extract_callable(tool)
        if isinstance(fn, GuardedTool):
            self.guarded_fn: GuardedTool = fn
        else:
            self.guarded_fn = guard(fn, config=cfg)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool through the agentguard protection stack.

        Accepts both positional and keyword arguments, mirroring the
        underlying tool's signature.

        Args:
            *args: Positional arguments forwarded to the tool.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            The tool's return value.

        Raises:
            ValidationError: If input validation fails.
            CircuitOpenError: If the circuit breaker is open.
            BudgetExceededError: If the budget limit is exceeded.
            RateLimitError: If the rate limit is exceeded.
        """
        return self.guarded_fn(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Async variant of :meth:`run`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The tool's return value.
        """
        return await self.guarded_fn.acall(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the guarded tool to be called directly."""
        return self.run(*args, **kwargs)

    # ------------------------------------------------------------------
    # CrewAI native interop
    # ------------------------------------------------------------------

    def to_crewai_tool(self) -> Any:
        """Return a native ``BaseTool`` subclass instance for use in a Crew.

        Requires ``crewai`` to be installed. The returned object is a proper
        ``BaseTool`` subclass so it is fully compatible with all CrewAI agents
        and crews.

        Returns:
            A ``BaseTool`` subclass instance wrapping this guarded tool.

        Raises:
            ImportError: If ``crewai`` is not installed.
        """
        if not _CREWAI_AVAILABLE:
            raise ImportError(
                "crewai is not installed. "
                "Install it with: pip install crewai"
            )

        guarded = self.guarded_fn
        tool_name = self.name
        tool_desc = self.description

        class _AgentGuardCrewAITool(CrewAIBaseTool):  # type: ignore[misc]
            name: str = tool_name
            description: str = tool_desc

            def _run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                return guarded(*args, **kwargs)

            async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                return await guarded.acall(*args, **kwargs)

        instance = _AgentGuardCrewAITool()
        instance.name = tool_name
        instance.description = tool_desc
        return instance

    # ------------------------------------------------------------------
    # Duck-type CrewAI BaseTool interface (for frameworks that use _run)
    # ------------------------------------------------------------------

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """CrewAI ``BaseTool._run`` compatibility shim."""
        return self.run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """CrewAI ``BaseTool._arun`` compatibility shim."""
        return await self.arun(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"GuardedCrewAITool(name={self.name!r}, "
            f"crewai_available={_CREWAI_AVAILABLE})"
        )


# ---------------------------------------------------------------------------
# Bulk helper
# ---------------------------------------------------------------------------


def guard_crewai_tools(
    tools: List[Any],
    config: Optional[GuardConfig] = None,
) -> List[GuardedCrewAITool]:
    """Wrap a list of CrewAI tools with agentguard protection.

    This is the recommended entry point for protecting an entire set of tools
    at once. All tools receive the same :class:`~agentguard.core.types.GuardConfig`.

    Args:
        tools: A list of CrewAI tools. Each may be a ``@tool``-decorated
            function, a ``BaseTool`` instance, or a plain callable.
        config: Guard configuration applied to all tools. Defaults to a
            zero-config :class:`~agentguard.core.types.GuardConfig`.

    Returns:
        A list of :class:`GuardedCrewAITool` instances in the same order as
        *tools*.

    Example::

        from agentguard.integrations.crewai_integration import guard_crewai_tools
        from agentguard import GuardConfig

        config = GuardConfig(
            validate_input=True,
            detect_hallucination=True,
            max_retries=2,
        )
        guarded = guard_crewai_tools([search_web, query_db, send_email], config=config)

        # Use in a CrewAI agent
        from crewai import Agent
        analyst = Agent(
            role="Research Analyst",
            goal="Research topics thoroughly",
            tools=guarded,  # drop-in replacement
        )
    """
    return [GuardedCrewAITool(t, config=config) for t in tools]
