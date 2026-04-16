# Model Benchmarking Suite

## Overview

Different LLMs have different propensities to hallucinate tool calls. agentguard's benchmarking suite lets you systematically measure and compare models on tool-call accuracy.

---

## Why Benchmark Tool-Call Accuracy?

Published benchmarks (MMLU, HumanEval, etc.) don't measure tool-call reliability. A model that scores 90% on MMLU might hallucinate 30% of tool calls when under context pressure. agentguard's benchmark measures:

- **Tool call accuracy** — does the model call the right tool with the right parameters?
- **Hallucination rate** — how often does the model fabricate responses?
- **Error recovery** — does the model fix its tool calls after a validation error?
- **Consistency** — does the model produce the same calls for the same prompts?

---

## Quick Start

```python
from agentguard.benchmarking import ToolCallBenchmark, BenchmarkSuite

# Define your tools and test cases
tools = [get_weather, search_web, query_database]

test_cases = [
    {
        "prompt": "What's the weather in London?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "London"},
    },
    {
        "prompt": "Search for recent AI papers",
        "expected_tool": "search_web",
        "expected_args": {"query": ...},  # Fuzzy match
    },
]

suite = BenchmarkSuite(tools=tools, test_cases=test_cases)

# Run across multiple models
results = suite.run(
    models=[
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "llama-3.3-70b-versatile",
    ],
    runs_per_case=5,   # Average over 5 runs per case for consistency
)

suite.print_report(results)
```

**Output:**

```
Model                        Accuracy  Hallucination%  Error Recovery  Consistency
gpt-4o                          94.2%           5.8%           78.3%       96.1%
gpt-4o-mini                     87.1%          12.9%           61.2%       89.4%
claude-3-5-sonnet-20241022      91.8%           8.2%           71.4%       93.2%
llama-3.3-70b-versatile         82.3%          17.7%           54.1%       85.7%
```

---

## Benchmark Metrics

### Accuracy

Fraction of test cases where the model called the correct tool with the correct arguments.

```python
accuracy = correct_calls / total_calls
```

"Correct" means: correct tool name + all required arguments present with correct types.

### Hallucination Rate

Fraction of calls where agentguard's `HallucinationDetector` flagged the response.

### Error Recovery Rate

When the model makes a bad call (wrong tool, wrong args, validation failure), how often does it fix the call in the next turn?

### Consistency

Given the same prompt, how often does the model make the same tool call across multiple runs?

---

## Custom Test Cases

```python
from agentguard.benchmarking import TestCase, FuzzyArg

cases = [
    TestCase(
        prompt="Get the stock price for Apple",
        expected_tool="get_stock_price",
        expected_args={
            "ticker": FuzzyArg(["AAPL", "Apple", "apple"]),  # Any of these is correct
        },
        # Optional: test that the model corrects itself after an error
        inject_error=True,
        error_message="ValidationError: ticker must be a stock symbol like 'AAPL'",
    ),
    TestCase(
        prompt="Search for Python async tutorials",
        expected_tool="search_web",
        expected_args={
            "query": FuzzyArg(contains="async"),  # Must contain "async"
        },
    ),
]
```

---

## Saving Benchmark Results

```python
suite.save_results(results, "benchmarks/results_2026_04.json")

# Compare with previous run
suite.compare("benchmarks/results_2026_03.json", "benchmarks/results_2026_04.json")
```

---

## Using Benchmark Results to Choose Models

After benchmarking, choose your deployment model based on your priorities:

| Priority | Metric to optimise | Recommendation |
|---|---|---|
| Lowest hallucination rate | `hallucination_rate` | Choose lowest |
| Best error recovery | `error_recovery_rate` | Choose highest |
| Most consistent | `consistency` | Choose highest |
| Balanced | Weighted score | See `suite.rank_models(results, weights=...)` |

```python
rankings = suite.rank_models(
    results,
    weights={
        "accuracy": 0.4,
        "hallucination": -0.3,   # Negative — lower is better
        "error_recovery": 0.2,
        "consistency": 0.1,
    }
)
```

---

## Running Benchmarks in CI

Add benchmarking to your CI pipeline to catch model regressions:

```yaml
# .github/workflows/benchmark.yml
name: Model Benchmark
on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install awesome-agentguard[benchmark]
      - run: python benchmarks/run_benchmark.py --output results.json
      - run: python benchmarks/check_regression.py --baseline benchmarks/baseline.json --current results.json --threshold 0.05
```

The regression check fails if any metric degrades by more than `threshold` (5% in this example).
