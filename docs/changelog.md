# Changelog

All notable changes to agentguard are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). agentguard uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Pluggable trace persistence with `SQLiteTraceStore`, `JsonlTraceStore`, and `TraceStore` facade routing
- SQLite-backed trace database as the default production persistence backend
- `trace_backend` and `trace_db_path` configuration on `GuardConfig`
- Backend-aware trace CLI commands: `init`, `import`, `export`, and `serve`
- Read-only local SQLite dashboard for browsing sessions, tool calls, failures, latency, retries, anomalies, and cost
- Trace platform architecture documentation covering SQLite, JSONL, and OpenTelemetry roles
- CrewAI integration (`GuardedCrewAITool`, `guard_crewai_tools`) — wrap CrewAI `@tool` functions and `BaseTool` subclasses with agentguard protection
- AutoGen integration (`GuardedAutoGenTool`, `guard_autogen_tool`, `guard_autogen_tools`, `register_guarded_tools`) — guard AutoGen tool functions before registration with agent pairs
- Comprehensive documentation site with MkDocs Material theme
- Guides for all core features: guard decorator, hallucination detection, circuit breaker, rate limiting, budget enforcement, custom validators, tracing, and testing
- Integration documentation for all supported frameworks
- API reference, config reference, types reference, error reference, CLI reference
- Advanced documentation: middleware pipeline, policy as code, multi-agent safety, telemetry, benchmarking

---

## [0.1.0] — Initial Release

### Added

**Core**
- `@guard` decorator — single decorator for the full protection stack
- `GuardConfig` — unified configuration for all guard options
- `GuardedTool` — the wrapper object created by `@guard`
- `ToolRegistry` — global registry for tool discovery and statistics

**Validators**
- `HallucinationDetector` — multi-signal hallucination detection (timing, schema, patterns, confidence scoring)
- `SchemaValidator` — automatic type-hint and Pydantic input/output validation
- `SemanticValidator` — per-tool semantic validation rules
- Custom validator support via `custom_validators` in `GuardConfig`

**Guardrails**
- `CircuitBreaker` — CLOSED → OPEN → HALF_OPEN state machine
- `RateLimiter` — token bucket rate limiting (per-second, per-minute, per-hour)
- `TokenBudget` — per-call and per-session cost and call-count budget enforcement
- `RetryPolicy` — exponential backoff with jitter and configurable exception filtering
- Timeout enforcement — thread-based (sync) and asyncio (async)

**Tracing**
- `TraceRecorder` / `record_session` — context manager for production trace recording
- `TraceStore` — read, write, and manage `.jsonl` trace files

**Testing**
- `assert_tool_call()` — fluent assertion builder for `TraceEntry` objects
- `TestGenerator` — auto-generate pytest tests from production traces
- `TraceReplayer` — replay recorded traces to detect regressions

**Integrations**
- OpenAI function calling (`OpenAIToolExecutor`, `guard_openai_tools`, `function_to_openai_tool`)
- Anthropic tool use (`AnthropicToolExecutor`, `guard_anthropic_tools`)
- LangChain (`GuardedLangChainTool`, `guard_langchain_tools`)
- MCP (`GuardedMCPServer`, `GuardedMCPClient`)
- OpenAI-compatible providers (`guard_tools`, `Provider`, `Providers`, `create_client`) — OpenRouter, Groq, Together AI, Fireworks AI, DeepInfra, Mistral, Perplexity, xAI, Novita AI

**Reporting**
- `ConsoleReporter` — Rich-powered colour terminal tables
- `JsonReporter` — JSON reports with latency percentiles

**CLI**
- `agentguard traces list` — list trace sessions
- `agentguard traces show` — inspect calls in a session
- `agentguard traces stats` — latency and failure statistics
- `agentguard traces report` — generate JSON report
- `agentguard generate` — auto-generate pytest tests from traces

---

## Version Policy

- **Patch** (0.1.x): Bug fixes, documentation improvements, performance improvements
- **Minor** (0.x.0): New features, new integrations, new guardrails — backward compatible
- **Major** (x.0.0): Breaking API changes — bump only when necessary

## Deprecation Policy

Features are deprecated with a `DeprecationWarning` for at least one minor version before removal. The deprecation warning includes the recommended alternative and the version when the feature will be removed.
