"""Global tool registry for discovery and management.

The registry tracks all tools that have been decorated with ``@guard``,
keyed by name. It supports lookup, listing, and bulk operations.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class ToolRegistration:
    """Metadata entry for a registered, guarded tool.

    Attributes:
        name: The tool's name (usually ``func.__name__``).
        func: The original (unwrapped) callable.
        guarded_func: The guarded callable produced by ``@guard``.
        description: Optional human-readable description (from docstring).
        tags: Arbitrary labels for filtering/discovery.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        guarded_func: Callable[..., Any],
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.func = func
        self.guarded_func = guarded_func
        self.description = description or (func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else ""
        self.tags: list[str] = tags or []
        self.call_count: int = 0
        self.failure_count: int = 0

    def __repr__(self) -> str:
        return (
            f"ToolRegistration(name={self.name!r}, "
            f"calls={self.call_count}, failures={self.failure_count})"
        )


class ToolRegistry:
    """Thread-safe registry of all guarded tools.

    Most users interact with the default :data:`global_registry` singleton.
    Custom registries can be created for isolation (e.g., in tests).

    Example::

        from agentguard.core.registry import global_registry

        for reg in global_registry.list_tools():
            print(reg.name, reg.call_count)
    """

    def __init__(self, name: str = "default") -> None:
        """Initialise the registry.

        Args:
            name: Human-readable name for this registry instance.
        """
        self.name = name
        self._tools: dict[str, ToolRegistration] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        registration: ToolRegistration,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a guarded tool.

        Args:
            registration: The :class:`ToolRegistration` to store.
            overwrite: If False (default), raises ``ValueError`` when a tool
                with the same name is already registered.

        Raises:
            ValueError: If *overwrite* is False and the name is already taken.
        """
        with self._lock:
            if registration.name in self._tools and not overwrite:
                raise ValueError(
                    f"Tool '{registration.name}' is already registered. "
                    "Pass overwrite=True to replace it."
                )
            self._tools[registration.name] = registration

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: The tool name to remove.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"No tool named '{name}' is registered.")
            del self._tools[name]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolRegistration]:
        """Return the registration for *name*, or None if not found.

        Args:
            name: Tool name to look up.
        """
        with self._lock:
            return self._tools.get(name)

    def require(self, name: str) -> ToolRegistration:
        """Return the registration for *name*, raising if not found.

        Args:
            name: Tool name to look up.

        Raises:
            KeyError: If the tool is not registered.
        """
        reg = self.get(name)
        if reg is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return reg

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_tools(self, tag: Optional[str] = None) -> list[ToolRegistration]:
        """Return all registered tools, optionally filtered by tag.

        Args:
            tag: If provided, only return tools that include this tag.

        Returns:
            List of :class:`ToolRegistration` objects.
        """
        with self._lock:
            regs = list(self._tools.values())
        if tag:
            regs = [r for r in regs if tag in r.tags]
        return sorted(regs, key=lambda r: r.name)

    def names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        with self._lock:
            return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    # Stats helpers (called by guard.py)
    # ------------------------------------------------------------------

    def increment_calls(self, name: str) -> None:
        """Increment the call counter for *name*. Silently no-ops if not found."""
        with self._lock:
            if name in self._tools:
                self._tools[name].call_count += 1

    def increment_failures(self, name: str) -> None:
        """Increment the failure counter for *name*. Silently no-ops if not found."""
        with self._lock:
            if name in self._tools:
                self._tools[name].failure_count += 1

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registrations. Primarily used in tests."""
        with self._lock:
            self._tools.clear()

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for reporting.

        Returns:
            Dictionary with tool stats keyed by name.
        """
        with self._lock:
            return {
                name: {
                    "calls": r.call_count,
                    "failures": r.failure_count,
                    "tags": r.tags,
                    "description": r.description,
                }
                for name, r in self._tools.items()
            }

    def __repr__(self) -> str:
        return f"ToolRegistry(name={self.name!r}, tools={self.names()})"


#: The default singleton registry used by ``@guard`` when none is specified.
global_registry = ToolRegistry(name="global")
