#!/usr/bin/env python3
"""Run a small multi-model benchmark suite and store artifacts cleanly."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("Set OPENROUTER_API_KEY before running run_benchmarks.py")

from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios, benchmark_run_dir


MODELS = [
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("openai/gpt-4o-2024-11-20", "GPT-4o"),
    ("anthropic/claude-sonnet-4", "Claude Sonnet"),
    ("google/gemini-2.0-flash-001", "Gemini Flash"),
]


def main() -> int:
    artifact_dir = ROOT / benchmark_run_dir(base_dir=ROOT / "artifacts" / "benchmarks" / "openrouter-matrix")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for model_id, display_name in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {display_name} ({model_id})")
        print(f"{'=' * 60}")

        runner = BenchmarkRunner()
        runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
        runner.add_scenarios(BuiltinScenarios.MULTI_TOOL_SELECTION)
        runner.add_scenarios(BuiltinScenarios.HALLUCINATION_RESISTANCE)

        try:
            results = runner.run(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=model_id,
            )

            report = results.to_report()
            out = report.save(artifact_dir)
            print(report.summary())

            with open(out, encoding="utf-8") as fh:
                all_results[display_name] = json.load(fh)

            print(f"Saved to {out}")
        except Exception as exc:
            print(f"ERROR benchmarking {display_name}: {exc}")
            import traceback
            traceback.print_exc()

        time.sleep(1)

    combined_path = artifact_dir / "all-models.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    print(f"\n\n{'=' * 60}")
    print("ALL BENCHMARKS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Models tested: {list(all_results.keys())}")
    print(f"Artifacts: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
