# CLI Reference

agentguard includes a backend-aware CLI for trace inspection, migration, reporting, generated tests, and the local dashboard.

## Install

```bash
pip install awesome-agentguard
agentguard --help
```

## Trace Commands

### `agentguard traces init`

Initialise a SQLite trace store:

```bash
agentguard traces init ./traces
agentguard traces init --db-path ./traces/agentguard_traces.db
```

### `agentguard traces list`

List recorded sessions:

```bash
agentguard traces list ./traces
agentguard traces list ./traces --json
```

### `agentguard traces show`

Inspect one session:

```bash
agentguard traces show agent_run_001 ./traces
agentguard traces show ./traces --session-id agent_run_001 --tool search_web --status failure --json
```

### `agentguard traces stats`

Aggregate statistics:

```bash
agentguard traces stats ./traces
agentguard traces stats ./traces --session agent_run_001 --json
```

### `agentguard traces report`

Generate a JSON report:

```bash
agentguard traces report ./traces --output report.json
agentguard traces report ./traces --entries --session agent_run_001
```

### `agentguard traces import`

Import legacy JSONL traces into SQLite:

```bash
agentguard traces import ./legacy_traces ./traces
```

### `agentguard traces export`

Export traces to JSONL:

```bash
agentguard traces export ./traces --output-dir ./trace-export
agentguard traces export ./traces --session agent_run_001 --output-dir ./trace-export
```

### `agentguard traces serve`

Launch the local SQLite dashboard:

```bash
agentguard traces serve ./traces
agentguard traces serve --db-path ./traces/agentguard_traces.db --port 9000
```

## Backend Selection

Most trace commands accept:

- `--backend sqlite`
- `--backend jsonl`
- `--db-path PATH`

If omitted, the CLI infers the backend from the storage path. SQLite is the default for newly initialised stores.

## Test Generation

Generate pytest cases from traces:

```bash
agentguard generate ./traces --output tests/test_generated.py
agentguard generate ./traces --backend sqlite --session agent_run_001
```

## Other Commands

```bash
agentguard registry
agentguard benchmark --model gpt-4o-mini
agentguard policy validate agentguard.toml
agentguard policy apply agentguard.toml
```
