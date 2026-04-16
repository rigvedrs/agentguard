from __future__ import annotations

import json
from pathlib import Path

from agentguard.cli.main import main
from agentguard.core.trace import JsonlTraceStore, SQLiteTraceStore, TraceStore
from agentguard.core.types import ToolCall, ToolCallStatus, ToolResult, TraceEntry
from agentguard.dashboard.server import _dashboard_html, _session_payload


def _make_entry(
    *,
    tool_name: str = "search_web",
    session_id: str = "session-a",
    status: ToolCallStatus = ToolCallStatus.SUCCESS,
    cost: float | None = None,
) -> TraceEntry:
    call = ToolCall(tool_name=tool_name, kwargs={"query": "weather"}, session_id=session_id)
    result = ToolResult(
        call_id=call.call_id,
        tool_name=tool_name,
        status=status,
        execution_time_ms=123.4,
        retry_count=1 if status == ToolCallStatus.RETRIED else 0,
        cost=cost,
    )
    return TraceEntry(call=call, result=result)


def test_sqlite_store_roundtrip_and_stats(tmp_path: Path) -> None:
    store = TraceStore(directory=str(tmp_path), backend="sqlite")
    store.write(_make_entry(cost=0.12))
    store.write(_make_entry(tool_name="query_db", session_id="session-b", status=ToolCallStatus.FAILURE))

    assert set(store.list_sessions()) == {"session-a", "session-b"}
    stats = store.stats()
    assert stats["total_calls"] == 2
    assert stats["failures"] == 1
    assert stats["calls_per_tool"]["search_web"] == 1


def test_import_jsonl_into_sqlite_and_export_back(tmp_path: Path) -> None:
    source_dir = tmp_path / "jsonl"
    sqlite_dir = tmp_path / "sqlite"
    export_dir = tmp_path / "exported"

    jsonl = JsonlTraceStore(directory=str(source_dir))
    jsonl.write(_make_entry(session_id="import-me"))

    sqlite = TraceStore(directory=str(sqlite_dir), backend="sqlite")
    imported = sqlite.import_jsonl(str(source_dir))
    assert imported == 1
    assert sqlite.read_session("import-me")[0].tool_name == "search_web"

    paths = sqlite.export_jsonl(str(export_dir))
    assert len(paths) == 1
    assert paths[0].name == "import-me.jsonl"


def test_trace_store_directory_auto_detects_sqlite(tmp_path: Path) -> None:
    sqlite = TraceStore(directory=str(tmp_path), backend="sqlite")
    sqlite.write(_make_entry(session_id="auto-detect"))

    reopened = TraceStore(directory=str(tmp_path))
    assert reopened.backend == "sqlite"
    assert reopened.read_session("auto-detect")


def test_cli_trace_init_import_report_and_export(tmp_path: Path, capsys) -> None:
    jsonl_dir = tmp_path / "legacy"
    jsonl_store = JsonlTraceStore(directory=str(jsonl_dir))
    jsonl_store.write(_make_entry(session_id="cli-import"))

    db_dir = tmp_path / "ops"
    export_dir = tmp_path / "ops-export"
    report_path = tmp_path / "report.json"

    assert main(["traces", "init", str(db_dir)]) == 0
    assert main(["traces", "import", str(jsonl_dir), str(db_dir)]) == 0
    assert main(["traces", "report", str(db_dir), "--output", str(report_path)]) == 0
    assert main(["traces", "export", str(db_dir), "--output-dir", str(export_dir)]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_calls"] == 1
    assert (export_dir / "cli-import.jsonl").exists()

    captured = capsys.readouterr().out
    assert "Imported 1 trace entries" in captured


def test_dashboard_payload_and_html(tmp_path: Path) -> None:
    store = SQLiteTraceStore(db_path=str(tmp_path / "dash.db"))
    store.write(_make_entry(session_id="dash-session", cost=0.42))

    payload = _session_payload(store, "dash-session", {})
    assert payload["summary"]["total_calls"] == 1
    assert payload["entries"][0]["tool_name"] == "search_web"

    html = _dashboard_html(host="127.0.0.1", port=8765, db_path=str(tmp_path / "dash.db"))
    assert "agentguard Trace Dashboard" in html
    assert "SQLite-backed observability" in html
