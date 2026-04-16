# Changelog

All notable changes to agentguard will be documented in this file.

## [0.2.0] - 2026-04-09

### Added
- Middleware pipeline for composable guard chains
- Policy-as-Code: YAML/TOML configuration files for guard rules
- OpenTelemetry integration for structured observability
- Multi-agent shared state (SharedBudget, SharedCircuitBreaker)
- CrewAI and AutoGen framework integrations
- Benchmarking suite for tool-call reliability testing
- Async context manager support (GuardedTool.session())
- async_record_session() for async trace recording
- OpenAI-compatible provider presets (OpenRouter, Groq, Together AI, Fireworks, etc.)
- Custom validators through @guard(custom_validators=[...])
- Hallucination profile persistence on GuardedTool
- has_key() alias on AssertionBuilder
- py.typed marker for PEP 561
- Comprehensive documentation site (docs/)

### Fixed
- Hallucination detector now persists across calls (was creating fresh instance each time)
- Async retry config resolution (max_retries kwarg was ignored in async path)
- Budget pre-call check now validates cost limit, not just call count
- ToolTimeoutError properly caught alongside TimeoutError
- Jitter range corrected to 0.5x-1.5x (was 0.5x-1.0x)
- Trace store handles mixed timezone-aware/naive datetimes
- Error reconstruction preserves original error detail
- Examples no longer require sys.path hack

### Changed
- RetryPolicy jitter documented as 0.5x-1.5x range
- All subpackages have proper __init__.py re-exports

## [0.1.0] - 2026-04-08

### Added
- Initial release
- @guard decorator with zero-config and full-config modes
- Input/output validation from type hints
- Hallucination detection (multi-signal: latency, fields, patterns)
- Circuit breaker (CLOSED/OPEN/HALF_OPEN state machine)
- Token bucket rate limiter
- Session budget enforcement (call count + cost)
- Retry with exponential backoff and jitter
- Timeout enforcement (thread-based for sync, asyncio for async)
- Trace recording to JSONL files
- Auto pytest test generation from traces
- Trace replay and diffing
- Fluent assertion builder (assert_tool_call)
- OpenAI, Anthropic, LangChain, MCP integrations
- Rich console reporter
- JSON report generation
- CLI: traces list/show/stats/report, registry, generate
