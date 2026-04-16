"""Benchmark report generation.

Converts :class:`~agentguard.benchmark.runner.BenchmarkResults` objects into
structured JSON reports that can be saved to disk or summarised in a terminal.

Usage::

    results = runner.run(model="gpt-4o-mini", ...)
    report = results.to_report()

    # Save to JSON
    report.save("benchmark_results.json")

    # Print summary
    print(report.summary())

    # Or compare multiple models
    comparison = runner.compare([results1, results2])
    comparison_report = BenchmarkReport.from_comparison(comparison)
    comparison_report.save("comparison.json")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from agentguard.benchmark.artifacts import resolve_report_path


class BenchmarkReport:
    """Serialisable container for benchmark results.

    Accepts one or more :class:`~agentguard.benchmark.runner.BenchmarkResults`
    objects and can render them as a plain-text summary or persist them as JSON.

    Args:
        results: List of per-model benchmark results.
        title: Optional report title.
        metadata: Arbitrary metadata dict to embed in the JSON output.
    """

    def __init__(
        self,
        results: Optional[list[Any]] = None,
        title: str = "agentguard Benchmark Report",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._results: list[Any] = results or []
        self.title = title
        self.metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_comparison(cls, comparison: Any, title: str = "") -> "BenchmarkReport":
        """Build a :class:`BenchmarkReport` from a :class:`ModelComparison`.

        Args:
            comparison: A :class:`~agentguard.benchmark.runner.ModelComparison`.
            title: Optional report title.

        Returns:
            A new :class:`BenchmarkReport` instance.
        """
        return cls(
            results=comparison.results,
            title=title or "agentguard Model Comparison Report",
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary.

        Returns:
            Dict ready for JSON serialisation with top-level keys:
            ``title``, ``generated_at``, ``metadata``, ``models``, ``summary``,
            and ``per_model``.
        """
        per_model: list[dict[str, Any]] = []
        for r in self._results:
            model_data: dict[str, Any] = {
                "model": r.model,
                "base_url": r.base_url,
                "metrics": {
                    "total_scenarios": r.total_scenarios,
                    "passed": r.passed_count,
                    "failed": r.failed_count,
                    "tool_call_accuracy": round(r.tool_call_accuracy, 4),
                    "parameter_accuracy": round(r.parameter_accuracy, 4),
                    "hallucination_rate": round(r.hallucination_rate, 4),
                    "avg_latency_ms": round(r.avg_latency_ms, 1),
                    "total_tokens": r.total_tokens_used,
                    "error_count": r.error_count,
                },
                "by_category": {
                    cat: {
                        "total": metrics["total"],
                        "passed": metrics["passed"],
                        "accuracy": round(metrics["accuracy"], 4),
                    }
                    for cat, metrics in r.by_category().items()
                },
                "scenarios": [
                    {
                        "name": sr.scenario.name,
                        "category": sr.scenario.category,
                        "passed": sr.passed,
                        "latency_ms": round(sr.latency_ms, 1),
                        "prompt_tokens": sr.prompt_tokens,
                        "completion_tokens": sr.completion_tokens,
                        "actual_tool_calls": sr.actual_tool_calls,
                        "expected_tool_calls": sr.scenario.expected_tool_calls,
                        "parameter_scores": {
                            k: round(v, 4) for k, v in sr.parameter_scores.items()
                        },
                        "error": sr.error,
                    }
                    for sr in r.scenario_results
                ],
            }
            per_model.append(model_data)

        return {
            "title": self.title,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": self.metadata,
            "models": [r.model for r in self._results],
            "per_model": per_model,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save the report as a JSON file.

        Args:
            path: Output file path or directory. If omitted, the report is
                written under ``artifacts/benchmarks/<timestamp>/``.

        Returns:
            The resolved :class:`pathlib.Path` of the written file.
        """
        out = resolve_report_path(self, path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return out.resolve()

    @classmethod
    def load(cls, path: str) -> "BenchmarkReport":
        """Load a previously saved report from a JSON file.

        The loaded report can only be used for reading/displaying; it does not
        reconstruct the full :class:`~agentguard.benchmark.runner.BenchmarkResults`
        objects.

        Args:
            path: Path to the JSON file.

        Returns:
            A :class:`BenchmarkReport` with metadata and title populated.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        report = cls(title=data.get("title", ""))
        report.metadata = data.get("metadata", {})
        # Store raw data for display purposes
        report._raw_data = data  # type: ignore[attr-defined]
        return report

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable multi-line summary.

        Returns:
            Formatted string suitable for printing to a terminal.
        """
        lines = [f"=== {self.title} ==="]

        for r in self._results:
            lines.append("")
            lines.append(r.summary())

        if len(self._results) > 1:
            lines.append("")
            lines.append("--- Comparison ---")
            from agentguard.benchmark.runner import ModelComparison
            comparison = ModelComparison(results=self._results)
            lines.append(comparison.summary())

        return "\n".join(lines)

    def __repr__(self) -> str:
        models = [r.model for r in self._results]
        return f"BenchmarkReport(models={models!r}, title={self.title!r})"
