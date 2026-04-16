"""Rich console output for agentguard guard events.

Provides colourful, structured terminal output for tool call results,
trace summaries, and registry overviews. Falls back gracefully when
``rich`` is not installed.

Usage::

    from agentguard.reporting.console import ConsoleReporter

    reporter = ConsoleReporter()
    reporter.print_result(tool_result)
    reporter.print_trace_summary(trace_store)
    reporter.print_registry()
"""

from __future__ import annotations

from typing import Any, Optional

from agentguard.core.types import ToolCallStatus, ToolResult, TraceEntry


# ---------------------------------------------------------------------------
# Rich availability check
# ---------------------------------------------------------------------------


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ConsoleReporter
# ---------------------------------------------------------------------------


class ConsoleReporter:
    """Formats and prints agentguard events to the terminal.

    When ``rich`` is installed, output is colour-coded and includes icons.
    Without it, plain ASCII output is used.

    Example::

        reporter = ConsoleReporter(verbose=True)
        reporter.print_result(result)
        reporter.print_session_summary(entries)
    """

    STATUS_ICONS = {
        ToolCallStatus.SUCCESS: "✓",
        ToolCallStatus.FAILURE: "✗",
        ToolCallStatus.TIMEOUT: "⏱",
        ToolCallStatus.CIRCUIT_OPEN: "⚡",
        ToolCallStatus.BUDGET_EXCEEDED: "💸",
        ToolCallStatus.RATE_LIMITED: "🚦",
        ToolCallStatus.VALIDATION_FAILED: "⚠",
        ToolCallStatus.HALLUCINATED: "👻",
        ToolCallStatus.RETRIED: "↺",
    }

    STATUS_COLORS = {
        ToolCallStatus.SUCCESS: "green",
        ToolCallStatus.FAILURE: "red",
        ToolCallStatus.TIMEOUT: "yellow",
        ToolCallStatus.CIRCUIT_OPEN: "magenta",
        ToolCallStatus.BUDGET_EXCEEDED: "red",
        ToolCallStatus.RATE_LIMITED: "yellow",
        ToolCallStatus.VALIDATION_FAILED: "red",
        ToolCallStatus.HALLUCINATED: "bright_red",
        ToolCallStatus.RETRIED: "cyan",
    }

    def __init__(self, verbose: bool = False, use_rich: Optional[bool] = None) -> None:
        """Initialise the reporter.

        Args:
            verbose: If True, print additional detail such as return values.
            use_rich: Force enable/disable rich. Defaults to auto-detect.
        """
        self.verbose = verbose
        self._use_rich = _rich_available() if use_rich is None else use_rich

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def print_result(self, result: ToolResult) -> None:
        """Print a single tool result to the console.

        Args:
            result: The tool result to display.
        """
        icon = self.STATUS_ICONS.get(result.status, "?")
        status = result.status.value.upper()
        name = result.tool_name
        ms = f"{result.execution_time_ms:.1f}ms"
        retries = f" (retries={result.retry_count})" if result.retry_count else ""
        line = f"{icon} [{status}] {name} — {ms}{retries}"

        if self._use_rich:
            self._rich_print_result(result, line)
        else:
            print(line)
            if result.exception and self.verbose:
                print(f"  Error: {result.exception}")

    def print_entry(self, entry: TraceEntry) -> None:
        """Print a trace entry to the console.

        Args:
            entry: The trace entry to display.
        """
        self.print_result(entry.result)
        if self.verbose:
            self._print_args(entry)

    def print_session_summary(self, entries: list[TraceEntry]) -> None:
        """Print a summary table for a list of trace entries.

        Args:
            entries: The entries to summarise.
        """
        if not entries:
            print("No trace entries to summarise.")
            return
        if self._use_rich:
            self._rich_session_summary(entries)
        else:
            self._plain_session_summary(entries)

    def print_registry_summary(self) -> None:
        """Print a summary of all registered tools from the global registry."""
        from agentguard.core.registry import global_registry
        summary = global_registry.summary()
        if self._use_rich:
            self._rich_registry_summary(summary)
        else:
            self._plain_registry_summary(summary)

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------

    def _rich_print_result(self, result: ToolResult, line: str) -> None:
        from rich.console import Console
        console = Console()
        color = self.STATUS_COLORS.get(result.status, "white")
        console.print(f"[{color}]{line}[/{color}]")
        if result.exception and (self.verbose or not result.succeeded):
            console.print(f"  [dim]Error: {result.exception}[/dim]")
        if self.verbose and result.return_value is not None:
            import json
            try:
                preview = json.dumps(result.return_value, default=str)[:120]
            except Exception:
                preview = repr(result.return_value)[:120]
            console.print(f"  [dim]→ {preview}[/dim]")

    def _rich_session_summary(self, entries: list[TraceEntry]) -> None:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="agentguard Session Summary", show_lines=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Retries", justify="right")
        table.add_column("Cost", justify="right")

        for e in entries:
            status = e.result.status.value
            color = self.STATUS_COLORS.get(e.result.status, "white")
            if e.result.cost_known and e.result.cost is not None:
                cost = f"${e.result.cost:.4f}"
            elif e.result.model:
                cost = "unknown"
            else:
                cost = "-"
            table.add_row(
                e.tool_name,
                f"[{color}]{status}[/{color}]",
                f"{e.result.execution_time_ms:.1f}",
                str(e.result.retry_count),
                cost,
            )

        total = len(entries)
        successes = sum(1 for e in entries if e.result.status == ToolCallStatus.SUCCESS)
        console.print(table)
        console.print(
            f"[bold]Total:[/bold] {total} calls | "
            f"[green]✓ {successes} succeeded[/green] | "
            f"[red]✗ {total - successes} failed[/red]"
        )

    def _rich_registry_summary(self, summary: dict[str, Any]) -> None:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Registered Tools", show_lines=True)
        table.add_column("Name", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Tags")

        for name, info in sorted(summary.items()):
            table.add_row(
                name,
                str(info["calls"]),
                str(info["failures"]),
                ", ".join(info["tags"]) or "-",
            )
        console.print(table)

    # ------------------------------------------------------------------
    # Plain output
    # ------------------------------------------------------------------

    def _print_args(self, entry: TraceEntry) -> None:
        if entry.call.kwargs:
            print(f"  Args: {dict(list(entry.call.kwargs.items())[:3])}")

    def _plain_session_summary(self, entries: list[TraceEntry]) -> None:
        total = len(entries)
        successes = sum(1 for e in entries if e.result.status == ToolCallStatus.SUCCESS)
        print(f"\n--- agentguard Session Summary ---")
        print(f"  Total calls  : {total}")
        print(f"  Succeeded    : {successes}")
        print(f"  Failed       : {total - successes}")
        avg_ms = sum(e.result.execution_time_ms for e in entries) / total
        print(f"  Avg latency  : {avg_ms:.1f}ms")
        print("-----------------------------------\n")

    def _plain_registry_summary(self, summary: dict[str, Any]) -> None:
        print("\n--- Registered Tools ---")
        for name, info in sorted(summary.items()):
            print(f"  {name}: calls={info['calls']}, failures={info['failures']}")
        print("------------------------\n")
