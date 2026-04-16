"""JSON report generation for agentguard traces.

Generates structured JSON reports from recorded traces, suitable for
ingestion by monitoring systems, dashboards, or CI pipelines.

Example::

    from agentguard.reporting.json_report import JsonReporter
    from agentguard.core.trace import TraceStore

    store = TraceStore("./traces")
    reporter = JsonReporter(store)
    report = reporter.generate()

    # Save to file
    reporter.save("report.json")

    # Or get as dict
    data = reporter.generate()
    print(data["summary"]["success_rate"])
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agentguard.core.trace import TraceStore
from agentguard.core.types import ToolCallStatus, TraceEntry


class JsonReporter:
    """Generates JSON reports from agentguard trace stores.

    Reports include:
    - Summary statistics (total calls, success rate, latency percentiles)
    - Per-tool breakdowns
    - Individual call records (optional)
    - Anomaly flags (hallucinations, circuit breaker events, budget warnings)

    Example::

        reporter = JsonReporter(TraceStore("./traces"))
        reporter.save("report.json", include_entries=True)
    """

    def __init__(
        self,
        store: TraceStore,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialise the reporter.

        Args:
            store: The trace store to report on.
            session_id: Optional session to restrict the report to.
        """
        self._store = store
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        include_entries: bool = False,
        include_anomalies: bool = True,
    ) -> dict[str, Any]:
        """Generate the full report as a dictionary.

        Args:
            include_entries: Include every individual trace entry in the report.
            include_anomalies: Include a list of anomalous calls.

        Returns:
            Structured report dictionary.
        """
        if self._session_id:
            entries = self._store.read_session(self._session_id)
        else:
            entries = self._store.read_all()

        report: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "session_id": self._session_id,
            "summary": self._build_summary(entries),
            "per_tool": self._build_per_tool(entries),
        }

        if include_anomalies:
            report["anomalies"] = self._build_anomalies(entries)

        if include_entries:
            report["entries"] = [self._serialise_entry(e) for e in entries]

        return report

    def save(
        self,
        path: str,
        *,
        include_entries: bool = False,
        indent: int = 2,
    ) -> Path:
        """Generate the report and save it to *path*.

        Args:
            path: Output file path (will be created if it doesn't exist).
            include_entries: Include individual trace entries.
            indent: JSON indentation level.

        Returns:
            The resolved output path.
        """
        report = self.generate(include_entries=include_entries)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=indent, default=str), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------

    def _build_summary(self, entries: list[TraceEntry]) -> dict[str, Any]:
        if not entries:
            return {"total_calls": 0}

        total = len(entries)
        statuses: dict[str, int] = {}
        for e in entries:
            key = e.result.status.value
            statuses[key] = statuses.get(key, 0) + 1

        successes = statuses.get(ToolCallStatus.SUCCESS.value, 0)
        latencies = sorted(e.result.execution_time_ms for e in entries)
        costs = [e.result.cost for e in entries if e.result.cost is not None]
        known_costs = sum(1 for e in entries if e.result.cost_known)
        unknown_costs = sum(1 for e in entries if e.result.model and not e.result.cost_known)
        retried = sum(e.result.retry_count for e in entries)
        hallucinated = sum(
            1 for e in entries
            if e.result.hallucination and e.result.hallucination.is_hallucinated
        )

        return {
            "total_calls": total,
            "status_counts": statuses,
            "success_rate": round(successes / total, 4),
            "latency_ms": {
                "min": latencies[0],
                "max": latencies[-1],
                "avg": round(sum(latencies) / total, 2),
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
            },
            "total_retries": retried,
            "hallucinated_calls": hallucinated,
            "total_cost_usd": round(sum(costs), 6) if costs else None,
            "cost_known_calls": known_costs,
            "cost_unknown_calls": unknown_costs,
        }

    def _build_per_tool(self, entries: list[TraceEntry]) -> dict[str, Any]:
        by_tool: dict[str, list[TraceEntry]] = {}
        for e in entries:
            by_tool.setdefault(e.tool_name, []).append(e)

        result: dict[str, Any] = {}
        for tool_name, tool_entries in sorted(by_tool.items()):
            total = len(tool_entries)
            successes = sum(
                1 for e in tool_entries
                if e.result.status == ToolCallStatus.SUCCESS
            )
            latencies = sorted(e.result.execution_time_ms for e in tool_entries)
            costs = [e.result.cost for e in tool_entries if e.result.cost is not None]

            result[tool_name] = {
                "total_calls": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total, 4),
                "avg_latency_ms": round(sum(latencies) / total, 2),
                "p95_latency_ms": _percentile(latencies, 95),
                "total_cost_usd": round(sum(costs), 6) if costs else None,
                "total_retries": sum(e.result.retry_count for e in tool_entries),
                "models": sorted({e.result.model for e in tool_entries if e.result.model}),
            }
        return result

    def _build_anomalies(self, entries: list[TraceEntry]) -> list[dict[str, Any]]:
        anomalies: list[dict[str, Any]] = []
        for e in entries:
            reasons: list[str] = []
            if e.result.hallucination and e.result.hallucination.is_hallucinated:
                reasons.append(
                    f"hallucinated (confidence={e.result.hallucination.confidence:.2f})"
                )
            if e.result.status == ToolCallStatus.CIRCUIT_OPEN:
                reasons.append("circuit_breaker_open")
            if e.result.status == ToolCallStatus.BUDGET_EXCEEDED:
                reasons.append("budget_exceeded")
            if e.result.retry_count >= 3:
                reasons.append(f"high_retry_count={e.result.retry_count}")
            failed_validations = [v for v in e.result.validations if not v.valid]
            if failed_validations:
                reasons.append(f"validation_failed: {failed_validations[0].message}")

            if reasons:
                anomalies.append({
                    "call_id": e.call_id,
                    "tool_name": e.tool_name,
                    "timestamp": e.call.timestamp.isoformat(),
                    "reasons": reasons,
                    "status": e.result.status.value,
                })
        return anomalies

    @staticmethod
    def _serialise_entry(entry: TraceEntry) -> dict[str, Any]:
        return {
            "call_id": entry.call_id,
            "tool_name": entry.tool_name,
            "timestamp": entry.call.timestamp.isoformat(),
            "status": entry.result.status.value,
            "execution_time_ms": entry.result.execution_time_ms,
            "retry_count": entry.result.retry_count,
            "cost": entry.result.cost,
            "provider": entry.result.provider,
            "model": entry.result.model,
            "cost_known": entry.result.cost_known,
            "cost_estimated": entry.result.cost_estimated,
            "cost_breakdown": (
                entry.result.cost_breakdown.model_dump(mode="json")
                if entry.result.cost_breakdown
                else None
            ),
            "usage": (
                entry.result.usage.model_dump(mode="json")
                if entry.result.usage
                else None
            ),
            "exception": entry.result.exception,
            "hallucinated": (
                entry.result.hallucination.is_hallucinated
                if entry.result.hallucination
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Compute the *pct*-th percentile of *sorted_values*."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return round(sorted_values[idx], 2)
