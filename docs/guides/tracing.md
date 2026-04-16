# Tracing

## Overview

agentguard records every guarded tool call as a structured `TraceEntry` containing inputs, outputs, timing, validation results, retry data, cost metadata, and failures.

There are now three observability paths:

- **SQLite trace store**: the default built-in production backend
- **JSONL trace files**: legacy-compatible export/import and replay format
- **OpenTelemetry**: use when you already have an external observability backend

## Default production setup

```python
from agentguard import guard

@guard(
    record=True,
    trace_backend="sqlite",
    trace_dir="./traces",
)
def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 72}
```

This writes to `./traces/agentguard_traces.db` unless you set `trace_db_path`.

## Legacy JSONL setup

```python
from agentguard import guard

@guard(
    record=True,
    trace_backend="jsonl",
    trace_dir="./traces",
)
def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 72}
```

## Session-level recording

```python
from agentguard import record_session

with record_session("./traces", backend="sqlite", session_id="agent_run_001") as recorder:
    get_weather("London")
    get_weather("NYC")

print(recorder.stats())
```

## Programmatic access

Use the facade if you want backend inference:

```python
from agentguard import TraceStore

store = TraceStore(directory="./traces")
entries = store.read_all()
session_entries = store.read_session("agent_run_001")
stats = store.stats()
```

Use concrete backends when you want explicit control:

```python
from agentguard import SQLiteTraceStore, JsonlTraceStore

sqlite_store = SQLiteTraceStore("./traces/agentguard_traces.db")
jsonl_store = JsonlTraceStore("./legacy_traces")
```

## CLI workflows

```bash
# Initialize a SQLite trace store
agentguard traces init ./traces

# Inspect sessions
agentguard traces list ./traces
agentguard traces show agent_run_001 ./traces
agentguard traces stats ./traces
agentguard traces report ./traces --output report.json

# Import legacy JSONL traces into SQLite
agentguard traces import ./legacy_traces ./traces

# Export back to JSONL for replay or generated tests
agentguard traces export ./traces --output-dir ./trace-export

# Run the built-in local dashboard
agentguard traces serve ./traces --port 8765
```

## Dashboard

`agentguard traces serve` launches a read-only local web UI backed by SQLite. The first release focuses on:

- session/run browsing
- session detail drill-down
- filtering by tool and status
- latency, retries, failures, anomalies, and cost visibility

The dashboard is intentionally local-first and single-process.

## Import and export

The intended migration path is:

1. keep existing JSONL traces if you already have them
2. import them into SQLite with `agentguard traces import`
3. use SQLite as the primary production store going forward
4. export back to JSONL when you need replay/test-generation portability

## Choosing between SQLite, JSONL, and OTel

- Choose **SQLite** for built-in production persistence and the dashboard.
- Choose **JSONL** when you need file portability or want to keep old workflows unchanged.
- Choose **OpenTelemetry** when traces must join a broader fleet-wide observability platform.
