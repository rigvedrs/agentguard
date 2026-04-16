# Trace Platform

agentguard now ships with a production-oriented trace platform built around a pluggable store abstraction.

## Default architecture

- **SQLite** is the default persistence backend for new trace recording.
- **JSONL** remains supported for backward compatibility, import/export, replay, and ad hoc inspection.
- **OpenTelemetry** remains the right choice when your team already has an external observability stack.

## Backend model

The trace layer exposes one public facade, `TraceStore`, plus two concrete backends:

- `SQLiteTraceStore`
- `JsonlTraceStore`

`TraceStore` preserves the historical API (`write`, `read_session`, `read_all`, `filter`, `stats`) while selecting the appropriate backend from explicit configuration or storage-path inference.

## Why SQLite is the default

SQLite gives agentguard a durable local store with:

- indexed queries by session, timestamp, tool, status, and call id
- a stable format for dashboards and CLI inspection
- easier migration than a pile of independent trace files
- no external service dependency for single-node production or developer workflows

## Configuration

Trace persistence is controlled from `GuardConfig`:

```python
from agentguard import GuardConfig

config = GuardConfig(
    record=True,
    trace_backend="sqlite",
    trace_dir="./traces",
    trace_db_path="./traces/agentguard_traces.db",
)
```

Use `trace_backend="jsonl"` to keep the legacy file-backed behavior.

## Operations flow

Recommended operator flow:

1. Record traces to SQLite in production or staging.
2. Use `agentguard traces list|show|stats|report` for scripted inspection.
3. Use `agentguard traces serve` for a local read-only dashboard.
4. Export to JSONL when you want portable files for replay, generated tests, or offline analysis.
5. Import old JSONL traces into SQLite with `agentguard traces import`.

## Retention guidance

SQLite is durable, but it is still a local file. For production use:

- back up the database regularly
- rotate or archive historical snapshots if the file grows indefinitely
- export subsets to JSONL before destructive cleanup
- use OTel alongside SQLite if you need centralised fleet-wide observability
