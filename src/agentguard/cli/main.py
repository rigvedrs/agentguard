"""Command-line interface for agentguard trace analysis.

Entry point: ``agentguard`` (registered in pyproject.toml).

Commands::

    agentguard traces list [DIR]          List all sessions
    agentguard traces show SESSION [DIR]  Show entries for a session
    agentguard traces report [DIR]        Generate JSON report
    agentguard traces stats [DIR]         Print summary statistics
    agentguard registry                   Show registered tools

Usage examples::

    $ agentguard traces list ./traces
    $ agentguard traces show 20240115 ./traces
    $ agentguard traces report ./traces --output report.json
    $ agentguard registry
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentguard",
        description="agentguard — runtime verification for AI agent tool calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_get_version(),
    )

    sub = parser.add_subparsers(title="commands")

    # --- traces ---
    traces_parser = sub.add_parser("traces", help="Manage and inspect trace files")
    traces_sub = traces_parser.add_subparsers(title="traces commands")

    # traces list
    p_list = traces_sub.add_parser("list", help="List recorded sessions")
    p_list.add_argument("dir", nargs="?", default="./traces", help="Traces directory")
    _add_trace_store_args(p_list)
    p_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_list.set_defaults(func=cmd_traces_list)

    # traces show
    p_show = traces_sub.add_parser("show", help="Show entries for a session")
    p_show.add_argument("session", nargs="?", default=None, help="Session ID")
    p_show.add_argument("dir", nargs="?", default="./traces", help="Traces directory")
    _add_trace_store_args(p_show)
    p_show.add_argument("--session-id", dest="session_id", default=None, help="Session ID")
    p_show.add_argument("--tool", default=None, help="Restrict to one tool")
    p_show.add_argument("--status", default=None, help="Restrict to one status")
    p_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_show.set_defaults(func=cmd_traces_show)

    # traces stats
    p_stats = traces_sub.add_parser("stats", help="Print summary statistics")
    p_stats.add_argument("dir", nargs="?", default="./traces", help="Traces directory")
    _add_trace_store_args(p_stats)
    p_stats.add_argument("--session", default=None, help="Restrict to one session")
    p_stats.add_argument("--json", action="store_true", help="Output as JSON")
    p_stats.set_defaults(func=cmd_traces_stats)

    # traces report
    p_report = traces_sub.add_parser("report", help="Generate a JSON report")
    p_report.add_argument("dir", nargs="?", default="./traces", help="Traces directory")
    _add_trace_store_args(p_report)
    p_report.add_argument("--output", "-o", default="report.json", help="Output file path")
    p_report.add_argument("--session", default=None, help="Restrict to one session")
    p_report.add_argument("--entries", action="store_true", help="Include individual entries")
    p_report.set_defaults(func=cmd_traces_report)

    # traces init
    p_init = traces_sub.add_parser("init", help="Initialise a SQLite trace database")
    p_init.add_argument("dir", nargs="?", default="./traces", help="Trace directory or db path")
    p_init.add_argument("--db-path", default=None, help="Explicit SQLite database path")
    p_init.set_defaults(func=cmd_traces_init)

    # traces import
    p_import = traces_sub.add_parser("import", help="Import legacy JSONL traces into SQLite")
    p_import.add_argument("source", help="Source JSONL traces directory")
    p_import.add_argument("dir", nargs="?", default="./traces", help="SQLite trace directory or db path")
    p_import.add_argument("--db-path", default=None, help="Explicit SQLite database path")
    p_import.set_defaults(func=cmd_traces_import)

    # traces export
    p_export = traces_sub.add_parser("export", help="Export traces to JSONL files")
    p_export.add_argument("dir", nargs="?", default="./traces", help="Source trace directory or db path")
    _add_trace_store_args(p_export)
    p_export.add_argument("--output-dir", required=True, help="Output directory for JSONL export")
    p_export.add_argument("--session", default=None, help="Restrict to one session")
    p_export.set_defaults(func=cmd_traces_export)

    # traces serve
    p_serve = traces_sub.add_parser("serve", help="Serve the local SQLite dashboard")
    p_serve.add_argument("dir", nargs="?", default="./traces", help="SQLite trace directory or db path")
    p_serve.add_argument("--db-path", default=None, help="Explicit SQLite database path")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    p_serve.set_defaults(func=cmd_traces_serve)

    # --- registry ---
    p_registry = sub.add_parser("registry", help="Show registered tools")
    p_registry.set_defaults(func=cmd_registry)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate pytest tests from traces")
    p_gen.add_argument("dir", nargs="?", default="./traces", help="Traces directory")
    _add_trace_store_args(p_gen)
    p_gen.add_argument("--output", "-o", default="tests/test_generated.py", help="Output file")
    p_gen.add_argument("--session", default=None, help="Restrict to one session")
    p_gen.set_defaults(func=cmd_generate)

    # --- benchmark ---
    p_bench = sub.add_parser(
        "benchmark",
        help="Benchmark LLM tool-calling accuracy",
        description=(
            "Run built-in tool-calling scenarios against any OpenAI-compatible model "
            "and print an accuracy report."
        ),
    )
    p_bench.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model identifier (default: gpt-4o-mini)",
    )
    p_bench.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "openrouter", "together", "groq", "fireworks", "custom"],
        help="API provider preset (sets base_url; use 'custom' with --base-url)",
    )
    p_bench.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help="Override base URL (required when --provider=custom)",
    )
    p_bench.add_argument(
        "--api-key",
        default="",
        dest="api_key",
        help="API key for the chosen provider (prefer environment variables in production)",
    )
    p_bench.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=[
            "basic", "multi_tool", "parameter_extraction",
            "hallucination_resistance", "error_handling", "tool_selection", "all",
        ],
        help="Scenario categories to run (default: all)",
    )
    p_bench.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save JSON report to this file path or directory",
    )
    p_bench.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress for each scenario",
    )
    p_bench.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        dest="max_tokens",
        help="Max tokens per model response (default: 1024)",
    )
    p_bench.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # --- policy ---
    policy_parser = sub.add_parser("policy", help="Validate and inspect policy files")
    policy_sub = policy_parser.add_subparsers(title="policy commands")

    # policy validate
    p_pol_val = policy_sub.add_parser("validate", help="Validate a policy file")
    p_pol_val.add_argument("file", help="Path to the policy file (.yaml, .yml, or .toml)")
    p_pol_val.set_defaults(func=cmd_policy_validate)

    # policy apply
    p_pol_apply = policy_sub.add_parser(
        "apply",
        help="Show what configurations would be applied from a policy file",
    )
    p_pol_apply.add_argument("file", help="Path to the policy file (.yaml, .yml, or .toml)")
    p_pol_apply.set_defaults(func=cmd_policy_apply)

    return parser


def _add_trace_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["sqlite", "jsonl"],
        default=None,
        help="Trace backend. Defaults to SQLite unless a legacy directory-only flow is inferred.",
    )
    parser.add_argument("--db-path", default=None, help="SQLite database path")


def _build_trace_store(args: argparse.Namespace, *, force_backend: str | None = None) -> Any:
    from agentguard.core.trace import TraceStore

    backend = force_backend or getattr(args, "backend", None)
    directory = getattr(args, "dir", "./traces")
    db_path = getattr(args, "db_path", None)
    return TraceStore(directory=directory, backend=backend, db_path=db_path)


def _parse_status(raw: str | None) -> Any:
    if raw is None:
        return None
    from agentguard.core.types import ToolCallStatus

    return ToolCallStatus(raw)


def _resolve_session_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "session_id", None) or getattr(args, "session", None)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_traces_list(args: argparse.Namespace) -> int:
    """List all recorded sessions in the configured trace store."""
    store = _build_trace_store(args)
    sessions = store.session_summaries()
    if not sessions:
        print(f"No trace sessions found in {args.dir!r}.")
        return 0
    if args.json:
        print(json.dumps(sessions, indent=2))
        return 0
    print(f"Sessions in {args.dir!r} ({store.backend}):")
    print(f"{'Session':<38} {'Calls':>7} {'Failures':>9} {'Started':>24} {'Duration':>12}")
    print("-" * 96)
    for session in sessions:
        duration = f"{session['duration_ms'] / 1000:.1f}s"
        print(
            f"{session['session_id']:<38} {session['calls']:>7} {session['failures']:>9} "
            f"{session['started_at']:>24} {duration:>12}"
        )
    return 0


def cmd_traces_show(args: argparse.Namespace) -> int:
    """Show trace entries for a specific session."""
    session_id = _resolve_session_arg(args)
    if not session_id:
        print("A session id is required. Pass it positionally or via --session-id.")
        return 1
    store = _build_trace_store(args)
    entries = store.filter(
        session_id=session_id,
        tool_name=args.tool,
        status=_parse_status(args.status),
    )
    if not entries:
        print(f"No entries found for session {session_id!r}.")
        return 1

    if args.json:
        data = [
            {
                "call_id": e.call_id,
                "tool_name": e.tool_name,
                "status": e.result.status.value,
                "time_ms": round(e.result.execution_time_ms, 2),
                "retries": e.result.retry_count,
                "cost_usd": e.result.cost,
                "exception": e.result.exception,
            }
            for e in entries
        ]
        print(json.dumps(data, indent=2))
        return 0

    print(f"\nSession: {session_id}  ({len(entries)} entries, backend={store.backend})")
    print(f"{'Tool':<30} {'Status':<20} {'Time (ms)':>10} {'Retries':>7} {'Cost':>10}")
    print("-" * 86)
    for e in entries:
        print(
            f"{e.tool_name:<30} {e.result.status.value:<20} "
            f"{e.result.execution_time_ms:>10.1f} {e.result.retry_count:>7} "
            f"{(e.result.cost if e.result.cost is not None else 0.0):>10.4f}"
        )
    return 0


def cmd_traces_stats(args: argparse.Namespace) -> int:
    """Print summary statistics for traces."""
    store = _build_trace_store(args)
    stats = store.stats(session_id=args.session)
    if not stats.get("total_calls"):
        print("No traces found.")
        return 0
    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return 0
    print("\n--- agentguard Trace Statistics ---")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print("-----------------------------------\n")
    return 0


def cmd_traces_report(args: argparse.Namespace) -> int:
    """Generate a JSON report from traces."""
    from agentguard.reporting.json_report import JsonReporter
    store = _build_trace_store(args)
    reporter = JsonReporter(store, session_id=args.session)
    out = reporter.save(args.output, include_entries=args.entries)
    print(f"Report written to: {out}")
    return 0


def cmd_traces_init(args: argparse.Namespace) -> int:
    """Initialise a SQLite trace database."""
    store = _build_trace_store(args, force_backend="sqlite")
    db_path = getattr(store.store, "db_path", None)
    print(f"Initialised SQLite trace store at: {db_path or args.dir}")
    return 0


def cmd_traces_import(args: argparse.Namespace) -> int:
    """Import JSONL traces into a SQLite trace database."""
    store = _build_trace_store(args, force_backend="sqlite")
    imported = store.import_jsonl(args.source)
    print(f"Imported {imported} trace entries from {args.source!r}.")
    return 0


def cmd_traces_export(args: argparse.Namespace) -> int:
    """Export traces to JSONL files."""
    store = _build_trace_store(args)
    paths = store.export_jsonl(args.output_dir, session_id=args.session)
    print(f"Exported {len(paths)} session file(s) to {args.output_dir}.")
    return 0


def cmd_traces_serve(args: argparse.Namespace) -> int:
    """Serve the local SQLite dashboard."""
    from agentguard.core.trace import SQLiteTraceStore
    from agentguard.dashboard import serve_dashboard

    store = _build_trace_store(args, force_backend="sqlite")
    concrete = store.store
    db_path = getattr(concrete, "db_path", None)
    if db_path is None:
        print("Dashboard requires a SQLite trace store.")
        return 1
    if not _port_available(args.host, args.port):
        print(f"Port {args.port} is already in use on {args.host}.")
        return 1
    SQLiteTraceStore(str(db_path))
    serve_dashboard(db_path=str(db_path), host=args.host, port=args.port)
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    """Print registered tools from the global registry."""
    from agentguard.core.registry import global_registry
    summary = global_registry.summary()
    if not summary:
        print("No tools registered yet.")
        return 0
    print(f"\n{'Tool':<30} {'Calls':>6} {'Failures':>8} Tags")
    print("-" * 60)
    for name, info in sorted(summary.items()):
        tags = ", ".join(info["tags"]) or "-"
        print(f"{name:<30} {info['calls']:>6} {info['failures']:>8}  {tags}")
    print()
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate pytest tests from recorded traces."""
    from agentguard.testing.generator import TestGenerator
    gen = TestGenerator(
        traces_dir=args.dir,
        backend=getattr(args, "backend", None),
        trace_db_path=getattr(args, "db_path", None),
    )
    code = gen.generate_tests(output=args.output, session_id=args.session)
    lines = code.count("\n")
    print(f"Generated {lines} lines of test code → {args.output}")
    return 0


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark suite against a model and print results."""
    import os

    # Resolve base URL from provider preset or explicit override
    _PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "together": "https://api.together.xyz/v1",
        "groq": "https://api.groq.com/openai/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
    }

    base_url = args.base_url or _PROVIDER_URLS.get(args.provider, "https://api.openai.com/v1")

    # API key: CLI flag > environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "together": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
        }
        api_key = os.environ.get(env_map.get(args.provider, ""), "")

    from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios

    runner = BenchmarkRunner(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    categories = args.categories or ["all"]

    category_map = {
        "basic": BuiltinScenarios.BASIC_TOOL_CALLING,
        "multi_tool": BuiltinScenarios.MULTI_TOOL_SELECTION,
        "parameter_extraction": BuiltinScenarios.PARAMETER_EXTRACTION,
        "hallucination_resistance": BuiltinScenarios.HALLUCINATION_RESISTANCE,
        "error_handling": BuiltinScenarios.ERROR_HANDLING,
        "tool_selection": BuiltinScenarios.TOOL_SELECTION,
        "all": BuiltinScenarios.ALL,
    }

    for cat in categories:
        fn = category_map.get(cat)
        if fn:
            runner.add_scenarios(fn)

    print(f"Running {len(runner.scenarios)} scenarios against {args.model} ...")
    print(f"Provider: {args.provider}  Base URL: {base_url}")
    print()

    try:
        results = runner.run(
            model=args.model,
            base_url=base_url,
            api_key=api_key,
        )
    except ImportError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return 1

    print()
    print(results.summary())

    if args.output:
        report = results.to_report()
        out = report.save(args.output)
        print(f"\nReport saved to: {out}")

    return 0


def cmd_policy_validate(args: argparse.Namespace) -> int:
    """Validate a policy file and report any errors."""
    from agentguard.policy import validate_policy
    errors = validate_policy(args.file)
    if not errors:
        print(f"✓ Policy file {args.file!r} is valid.")
        return 0
    print(f"✗ Policy file {args.file!r} has {len(errors)} error(s):")
    for err in errors:
        print(f"  - {err}")
    return 1


def cmd_policy_apply(args: argparse.Namespace) -> int:
    """Print a human-readable summary of what would be applied from a policy file."""
    from agentguard.policy import load_policy, policy_summary, PolicyValidationError
    try:
        policy = load_policy(args.file)
    except PolicyValidationError as exc:
        print(f"✗ Policy file is invalid:\n{exc}")
        return 1
    except Exception as exc:
        print(f"✗ Failed to load policy: {exc}")
        return 1
    print(policy_summary(policy, file_path=args.file))
    return 0


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("agentguard")
    except Exception:
        return "agentguard (unknown version)"


if __name__ == "__main__":
    sys.exit(main())
