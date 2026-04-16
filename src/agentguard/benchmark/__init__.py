"""Benchmarking suite for testing LLM tool-calling accuracy.

This package provides a turn-key solution for measuring how well different
LLM models handle tool calls — including basic routing, parameter extraction,
multi-tool scenarios, hallucination resistance, and error handling.

Quick start::

    import os
    from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios

    runner = BenchmarkRunner()
    runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
    runner.add_scenarios(BuiltinScenarios.HALLUCINATION_RESISTANCE)

    results = runner.run(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",
    )

    print(results.summary())
    results.to_report().save("benchmark.json")
"""

from agentguard.benchmark.report import BenchmarkReport
from agentguard.benchmark.artifacts import (
    DEFAULT_BENCHMARK_ARTIFACTS_DIR,
    benchmark_run_dir,
    model_slug,
)
from agentguard.benchmark.runner import (
    BenchmarkResults,
    BenchmarkRunner,
    ModelComparison,
    ScenarioResult,
)
from agentguard.benchmark.scenarios import BenchmarkScenario, BuiltinScenarios

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResults",
    "BenchmarkReport",
    "BenchmarkScenario",
    "BuiltinScenarios",
    "DEFAULT_BENCHMARK_ARTIFACTS_DIR",
    "ModelComparison",
    "ScenarioResult",
    "benchmark_run_dir",
    "model_slug",
]
