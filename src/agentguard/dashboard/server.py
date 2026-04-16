"""Minimal read-only dashboard for SQLite-backed trace data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agentguard.core.trace import SQLiteTraceStore
from agentguard.core.types import ToolCallStatus
from agentguard.reporting.json_report import JsonReporter


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: str) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _session_payload(store: SQLiteTraceStore, session_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    status_raw = query.get("status", [None])[0]
    tool_name = query.get("tool", [None])[0]
    status = ToolCallStatus(status_raw) if status_raw else None
    entries = store.filter(session_id=session_id, tool_name=tool_name, status=status)
    return {
        "session_id": session_id,
        "entries": [JsonReporter._serialise_entry(entry) for entry in entries],
        "summary": JsonReporter(store, session_id=session_id).generate(include_entries=False)["summary"],
    }


def serve_dashboard(
    *,
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the local trace dashboard."""
    store = SQLiteTraceStore(db_path=db_path)

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/":
                _html_response(self, _dashboard_html(host=host, port=port, db_path=db_path))
                return
            if parsed.path == "/api/summary":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "generated_at": _utcnow(),
                        "store": {"backend": "sqlite", "db_path": db_path},
                        "stats": store.stats(),
                        "sessions": store.session_summaries(),
                    },
                )
                return
            if parsed.path == "/api/sessions":
                _json_response(self, HTTPStatus.OK, store.session_summaries())
                return
            if parsed.path.startswith("/api/session/"):
                session_id = parsed.path.removeprefix("/api/session/")
                _json_response(self, HTTPStatus.OK, _session_payload(store, session_id, query))
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"agentguard dashboard listening on http://{host}:{port} (db={db_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _dashboard_html(*, host: str, port: int, db_path: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agentguard traces</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1f2630;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #2f81f7;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #182033 0%, var(--bg) 45%);
      color: var(--text);
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      backdrop-filter: blur(12px);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }}
    .stat .label {{ color: var(--muted); font-size: 0.85rem; }}
    .stat .value {{ font-size: 1.4rem; margin-top: 6px; }}
    .session-list {{
      max-height: 72vh;
      overflow: auto;
      display: grid;
      gap: 10px;
    }}
    button.session {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 14px;
      text-align: left;
      padding: 12px;
      cursor: pointer;
    }}
    button.session.active {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(47,129,247,0.35); }}
    .row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    select {{
      background: #0f1520;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .status-success, .status-retried {{ color: var(--good); }}
    .status-failure, .status-validation_failed, .status-budget_exceeded, .status-circuit_open {{ color: var(--bad); }}
    .status-rate_limited, .status-timeout, .status-hallucinated {{ color: var(--warn); }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      font-size: 0.78rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .session-list {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>agentguard Trace Dashboard</h1>
      <p>SQLite-backed observability for runs, failures, latency, retries, anomalies, and spend. Database: <code>{db_path}</code></p>
    </div>
    <div class="stats" id="global-stats"></div>
    <div class="grid">
      <div class="panel">
        <div class="row">
          <strong>Sessions</strong>
          <span class="pill" id="session-count">0 loaded</span>
        </div>
        <div class="session-list" id="sessions"></div>
      </div>
      <div class="panel">
        <div class="row">
          <div>
            <strong id="session-title">Select a session</strong>
            <div class="muted" id="session-meta">HTTP {host}:{port}</div>
          </div>
          <div class="filters">
            <select id="tool-filter"><option value="">All tools</option></select>
            <select id="status-filter">
              <option value="">All statuses</option>
              <option value="success">success</option>
              <option value="failure">failure</option>
              <option value="timeout">timeout</option>
              <option value="circuit_open">circuit_open</option>
              <option value="budget_exceeded">budget_exceeded</option>
              <option value="rate_limited">rate_limited</option>
              <option value="validation_failed">validation_failed</option>
              <option value="hallucinated">hallucinated</option>
              <option value="retried">retried</option>
            </select>
          </div>
        </div>
        <div class="stats" id="session-stats"></div>
        <div style="overflow:auto;">
          <table>
            <thead>
              <tr>
                <th>Tool</th>
                <th>Status</th>
                <th>Time</th>
                <th>Retries</th>
                <th>Cost</th>
                <th>Exception</th>
              </tr>
            </thead>
            <tbody id="entries"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
  <script>
    const state = {{ sessions: [], selected: null, entries: [] }};

    const el = (id) => document.getElementById(id);

    function statCard(label, value) {{
      return `<div class="stat"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
    }}

    async function loadSummary() {{
      const res = await fetch('/api/summary');
      const data = await res.json();
      state.sessions = data.sessions;
      renderGlobalStats(data.stats);
      renderSessions();
      if (state.sessions.length > 0) {{
        await loadSession(state.sessions[0].session_id);
      }}
    }}

    function renderGlobalStats(stats) {{
      el('global-stats').innerHTML = [
        statCard('Total Calls', stats.total_calls ?? 0),
        statCard('Success Rate', stats.success_rate ? `${{(stats.success_rate * 100).toFixed(1)}}%` : '0%'),
        statCard('Failures', stats.failures ?? 0),
        statCard('Retries', stats.total_retries ?? 0),
        statCard('Avg Latency', stats.avg_latency_ms ? `${{stats.avg_latency_ms.toFixed(1)}} ms` : '0 ms'),
        statCard('Total Cost', stats.total_cost_usd != null ? `$${{stats.total_cost_usd.toFixed(4)}}` : 'n/a'),
      ].join('');
      el('session-count').textContent = `${{state.sessions.length}} loaded`;
    }}

    function renderSessions() {{
      el('sessions').innerHTML = state.sessions.map((session) => `
        <button class="session ${{state.selected === session.session_id ? 'active' : ''}}" data-session="${{session.session_id}}">
          <div><strong>${{session.session_id}}</strong></div>
          <div class="muted">${{session.calls}} calls · ${{session.failures}} failures</div>
          <div class="muted">${{new Date(session.started_at).toLocaleString()}}</div>
        </button>
      `).join('');
      document.querySelectorAll('button.session').forEach((button) => {{
        button.addEventListener('click', () => loadSession(button.dataset.session));
      }});
    }}

    async function loadSession(sessionId) {{
      state.selected = sessionId;
      renderSessions();
      const params = new URLSearchParams();
      const tool = el('tool-filter').value;
      const status = el('status-filter').value;
      if (tool) params.set('tool', tool);
      if (status) params.set('status', status);
      const res = await fetch(`/api/session/${{encodeURIComponent(sessionId)}}?${{params.toString()}}`);
      const data = await res.json();
      state.entries = data.entries;
      renderSession(sessionId, data.summary, data.entries);
    }}

    function renderSession(sessionId, summary, entries) {{
      el('session-title').textContent = sessionId;
      el('session-meta').textContent = `${{entries.length}} visible entries`;
      el('session-stats').innerHTML = [
        statCard('Visible Calls', summary.total_calls ?? entries.length),
        statCard('Hallucinations', summary.hallucinated_calls ?? 0),
        statCard('p95 Latency', summary.latency_ms ? `${{summary.latency_ms.p95.toFixed(1)}} ms` : '0 ms'),
        statCard('Known Cost Calls', summary.cost_known_calls ?? 0),
      ].join('');

      const tools = [...new Set(entries.map((entry) => entry.tool_name))].sort();
      el('tool-filter').innerHTML = '<option value="">All tools</option>' + tools.map((tool) => `<option value="${{tool}}">${{tool}}</option>`).join('');

      el('entries').innerHTML = entries.map((entry) => `
        <tr>
          <td>${{entry.tool_name}}</td>
          <td class="status-${{entry.status}}">${{entry.status}}</td>
          <td>${{entry.execution_time_ms.toFixed(1)}} ms</td>
          <td>${{entry.retry_count}}</td>
          <td>${{entry.cost != null ? '$' + entry.cost.toFixed(4) : 'n/a'}}</td>
          <td class="muted">${{entry.exception || ''}}</td>
        </tr>
      `).join('');
    }}

    el('tool-filter').addEventListener('change', () => state.selected && loadSession(state.selected));
    el('status-filter').addEventListener('change', () => state.selected && loadSession(state.selected));
    loadSummary();
  </script>
</body>
</html>"""
