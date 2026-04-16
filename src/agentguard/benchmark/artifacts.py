"""Helpers for organizing benchmark artifacts on disk."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentguard.benchmark.report import BenchmarkReport


DEFAULT_BENCHMARK_ARTIFACTS_DIR = Path("artifacts") / "benchmarks"


def benchmark_run_dir(base_dir: str | Path | None = None, timestamp: str | None = None) -> Path:
    """Return the default directory for one benchmark run."""
    root = Path(base_dir) if base_dir is not None else DEFAULT_BENCHMARK_ARTIFACTS_DIR
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return root / stamp


def model_slug(value: str) -> str:
    """Normalise a model name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "benchmark"


def default_report_filename(report: "BenchmarkReport") -> str:
    """Choose a stable report filename for a report."""
    results = getattr(report, "_results", [])
    if len(results) == 1 and getattr(results[0], "model", ""):
        return f"{model_slug(results[0].model)}.json"
    return "comparison.json"


def resolve_report_path(report: "BenchmarkReport", path: str | Path | None = None) -> Path:
    """Resolve a report target path.

    When ``path`` is omitted, a timestamped run directory under
    ``artifacts/benchmarks`` is used. When ``path`` points to a directory
    (or has no suffix), the default report filename is appended.
    """
    if path is None:
        target = benchmark_run_dir()
    else:
        target = Path(path)

    if target.suffix.lower() == ".json":
        return target

    return target / default_report_filename(report)
