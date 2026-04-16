"""AutoGen integration for agentguard.

Provides decorators and helpers to wrap AutoGen tool functions with agentguard
protection before they are registered with AutoGen agents.

AutoGen is an optional dependency. If it is not installed, the module can still
be imported; registration helpers that require AutoGen classes will raise
``ImportError`` when called.

Supported AutoGen versions
--------------------------
- **AutoGen 0.2.x** (``pyautogen`` / ``autogen``) — uses ``register_for_llm``
  and ``register_for_execution`` decorators on agent instances.
- **AutoGen 0.4.x** / **AG2** (``autogen-agentchat``) — same pattern with the
  new package name; the integration works identically.

Usage — single tool::

    from agentguard.integrations.autogen_integration import guard_autogen_tool
    from agentguard import GuardConfig

    config = GuardConfig(validate_input=True, max_retries=2)

    @guard_autogen_tool(config=config)
    def search_web(query: str) -> str:
        '''Search the web for information.'''
        return requests.get(f"https://...?q={query}").text

    # Register with AutoGen as normal
    @assistant.register_for_llm(description="Search the web")
    @user_proxy.register_for_execution()
    def search_web_tool(query: str) -> str:
        return search_web(query)

Usage — bulk wrapping::

    from agentguard.integrations.autogen_integration import guard_autogen_tools

    guarded = guard_autogen_tools([search_web, query_db], config=config)
    # guarded is a dict mapping function name → GuardedAutoGenTool

Usage — AssistantAgent helper (AutoGen 0.2.x)::

    from agentguard.integrations.autogen_integration import register_guarded_tools

    register_guarded_tools(
        tools=[search_web, query_db],
        llm_agent=assistant,
        executor_agent=user_proxy,
        config=config,
    )
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig


# ---------------------------------------------------------------------------
# Optional AutoGen import detection
# ---------------------------------------------------------------------------

try:
    import autogen  # type: ignore[import]

    _AUTOGEN_AVAILABLE = True
    _AUTOGEN_PACKAGE = "autogen"
except ImportError:
    try:
        import pyautogen as autogen  # type: ignore[import,no-redef]

        _AUTOGEN_AVAILABLE = True
        _AUTOGEN_PACKAGE = "pyautogen"
    except ImportError:
        autogen = None  # type: ignore[assignment]
        _AUTOGEN_AVAILABLE = False
        _AUTOGEN_PACKAGE = None


# ---------------------------------------------------------------------------
# GuardedAutoGenTool
# ---------------------------------------------------------------------------


class GuardedAutoGenTool:
    """An agentguard-protected wrapper for an AutoGen tool function.

    Wraps a plain Python function (intended for AutoGen registration) with
    the full agentguard protection stack. The wrapper preserves the original
    function's name, docstring, and type annotations so AutoGen's introspection
    works correctly.

    The wrapped function can be called directly or registered with an AutoGen
    ``AssistantAgent`` / ``UserProxyAgent`` using the standard decorators.

    Attributes:
        name: The original function name.
        description: The function's first docstring line.
        guarded_fn: The underlying :class:`~agentguard.core.guard.GuardedTool`.
        __wrapped__: The original function (for introspection).

    Example::

        from agentguard.integrations.autogen_integration import GuardedAutoGenTool
        from agentguard import GuardConfig

        def search_web(query: str) -> str:
            '''Search the web for information.'''
            return requests.get(f"https://api.example.com?q={query}").text

        guarded = GuardedAutoGenTool(search_web, config=GuardConfig(max_retries=2))

        # Register with AutoGen
        @assistant.register_for_llm(description="Search the web")
        @user_proxy.register_for_execution()
        def search_web_proxy(query: str) -> str:
            return guarded(query=query)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: Optional[GuardConfig] = None,
        *,
        description: Optional[str] = None,
    ) -> None:
        """Initialise the guarded wrapper.

        Args:
            func: The function to protect. Must have a proper signature with
                type annotations for AutoGen's schema generation.
            config: Guard configuration. Defaults to a zero-config
                :class:`~agentguard.core.types.GuardConfig`.
            description: Override the description (defaults to first docstring
                line or ``"Run <name>"``).
        """
        cfg = config or GuardConfig()
        self.name: str = func.__name__
        raw_doc = (func.__doc__ or "").strip()
        first_line = raw_doc.splitlines()[0] if raw_doc else ""
        self.description: str = description or first_line or f"Run {self.name}"

        if isinstance(func, GuardedTool):
            self.guarded_fn: GuardedTool = func
        else:
            self.guarded_fn = guard(func, config=cfg)

        self.__wrapped__ = func
        # Copy annotations so AutoGen can build the JSON schema
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__doc__ = func.__doc__

        # Create a proper proxy callable that preserves the signature
        # (AutoGen inspects __code__ / inspect.signature)
        @functools.wraps(func)
        def _proxy(*args: Any, **kwargs: Any) -> Any:
            return self.guarded_fn(*args, **kwargs)

        self._proxy = _proxy

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool through the agentguard protection stack.

        Args:
            *args: Positional arguments forwarded to the tool.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            The tool's return value.
        """
        return self.guarded_fn(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async variant of :meth:`__call__`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The tool's return value.
        """
        return await self.guarded_fn.acall(*args, **kwargs)

    # ------------------------------------------------------------------
    # AutoGen interop helpers
    # ------------------------------------------------------------------

    def as_function(self) -> Callable[..., Any]:
        """Return the wrapped function with the original signature intact.

        AutoGen inspects ``inspect.signature`` to generate JSON schemas for the
        LLM. This method returns a ``functools.wraps``-wrapped proxy that
        preserves the original function's signature while routing calls through
        the agentguard protection stack.

        Returns:
            A callable with the original function's signature.
        """
        return self._proxy

    def register(
        self,
        llm_agent: Any,
        executor_agent: Any,
        *,
        description: Optional[str] = None,
    ) -> None:
        """Register this tool with an AutoGen agent pair.

        Convenience method for the common pattern::

            @assistant.register_for_llm(description="...")
            @user_proxy.register_for_execution()
            def tool(...): ...

        Args:
            llm_agent: The ``AssistantAgent`` (or similar) that uses the tool
                definition for LLM schema generation.
            executor_agent: The ``UserProxyAgent`` (or similar) that executes
                the tool.
            description: Description passed to ``register_for_llm``. Defaults
                to ``self.description``.

        Raises:
            ImportError: If autogen is not installed.
            AttributeError: If the agents don't have registration methods.
        """
        if not _AUTOGEN_AVAILABLE:
            raise ImportError(
                f"autogen is not installed. "
                f"Install it with: pip install pyautogen"
            )

        desc = description or self.description
        proxy = self.as_function()

        # Apply decorators in the AutoGen-expected order
        proxy = executor_agent.register_for_execution()(proxy)
        proxy = llm_agent.register_for_llm(description=desc)(proxy)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GuardedAutoGenTool(name={self.name!r}, "
            f"autogen_available={_AUTOGEN_AVAILABLE})"
        )


# ---------------------------------------------------------------------------
# Decorator: @guard_autogen_tool
# ---------------------------------------------------------------------------


def guard_autogen_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[GuardConfig] = None,
    description: Optional[str] = None,
) -> Union[GuardedAutoGenTool, Callable[[Callable[..., Any]], GuardedAutoGenTool]]:
    """Decorator that wraps an AutoGen tool function with agentguard protection.

    Can be used with or without arguments::

        # Without config — uses defaults
        @guard_autogen_tool
        def search_web(query: str) -> str:
            '''Search the web.'''
            ...

        # With config
        @guard_autogen_tool(config=GuardConfig(validate_input=True, max_retries=2))
        def query_db(sql: str, limit: int = 100) -> list[dict]:
            '''Query the database.'''
            ...

    Args:
        func: The function to decorate (when used without parentheses).
        config: Guard configuration.
        description: Override the tool description.

    Returns:
        A :class:`GuardedAutoGenTool` instance (when applied to a function) or
        a decorator (when called with config arguments).
    """
    if func is not None:
        # Called as @guard_autogen_tool (no parentheses)
        return GuardedAutoGenTool(func, config=config, description=description)

    # Called as @guard_autogen_tool(...) — return a decorator
    def _decorator(fn: Callable[..., Any]) -> GuardedAutoGenTool:
        return GuardedAutoGenTool(fn, config=config, description=description)

    return _decorator


# ---------------------------------------------------------------------------
# Bulk helper: guard_autogen_tools
# ---------------------------------------------------------------------------


def guard_autogen_tools(
    tools: List[Callable[..., Any]],
    config: Optional[GuardConfig] = None,
) -> Dict[str, GuardedAutoGenTool]:
    """Wrap a list of AutoGen tool functions with agentguard protection.

    Args:
        tools: A list of plain callables to wrap.
        config: Guard configuration applied to all tools.

    Returns:
        A dict mapping each function's ``__name__`` to a
        :class:`GuardedAutoGenTool` instance.

    Example::

        from agentguard.integrations.autogen_integration import guard_autogen_tools
        from agentguard import GuardConfig

        config = GuardConfig(validate_input=True, max_retries=2)
        guarded = guard_autogen_tools([search_web, query_db], config=config)

        # Access by name
        guarded["search_web"]("Python tutorials")

        # Register all with agents
        for tool in guarded.values():
            tool.register(assistant, user_proxy)
    """
    return {fn.__name__: GuardedAutoGenTool(fn, config=config) for fn in tools}


# ---------------------------------------------------------------------------
# Convenience: register_guarded_tools
# ---------------------------------------------------------------------------


def register_guarded_tools(
    tools: List[Callable[..., Any]],
    llm_agent: Any,
    executor_agent: Any,
    config: Optional[GuardConfig] = None,
) -> Dict[str, GuardedAutoGenTool]:
    """Guard and register a list of tools with an AutoGen agent pair.

    Combines :func:`guard_autogen_tools` with per-tool :meth:`GuardedAutoGenTool.register`
    into a single call.

    Args:
        tools: A list of plain callables to protect and register.
        llm_agent: The ``AssistantAgent`` for LLM schema registration.
        executor_agent: The ``UserProxyAgent`` for execution registration.
        config: Guard configuration applied to all tools.

    Returns:
        A dict mapping function names to :class:`GuardedAutoGenTool` instances.

    Raises:
        ImportError: If autogen is not installed.

    Example::

        from agentguard.integrations.autogen_integration import register_guarded_tools
        from agentguard import GuardConfig
        import autogen

        config = GuardConfig(validate_input=True, max_retries=2, record=True)
        assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
        user_proxy = autogen.UserProxyAgent("user_proxy")

        guarded = register_guarded_tools(
            [search_web, query_db],
            llm_agent=assistant,
            executor_agent=user_proxy,
            config=config,
        )
    """
    guarded = guard_autogen_tools(tools, config=config)
    for tool in guarded.values():
        tool.register(llm_agent, executor_agent)
    return guarded
