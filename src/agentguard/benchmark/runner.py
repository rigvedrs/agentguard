"""BenchmarkRunner — orchestrates LLM tool-calling accuracy benchmarks.

The runner sends each :class:`~agentguard.benchmark.scenarios.BenchmarkScenario`
to an OpenAI-compatible API endpoint, parses the model's response, validates the
tool calls against expected outputs, and collects latency / token-usage metrics.

Usage::

    import os
    from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios

    runner = BenchmarkRunner()
    runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
    runner.add_scenarios(BuiltinScenarios.MULTI_TOOL_SELECTION)

    results = runner.run(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",
    )

    print(results.summary())
    results.to_report().save("results.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agentguard.benchmark.scenarios import BenchmarkScenario


# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of running one :class:`BenchmarkScenario`.

    Attributes:
        scenario: The scenario that was run.
        model: The model identifier used.
        passed: Whether the model's tool calls matched expectations.
        actual_tool_calls: The tool calls the model actually produced.
        latency_ms: Wall-clock time for the API round-trip in milliseconds.
        prompt_tokens: Number of prompt tokens consumed (if reported).
        completion_tokens: Number of completion tokens consumed (if reported).
        error: Error message if the API call failed.
        parameter_scores: Per-parameter match scores (0.0–1.0).
    """

    scenario: BenchmarkScenario
    model: str
    passed: bool
    actual_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    parameter_scores: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this scenario."""
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Aggregate results for one model run
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results for one model across all scenarios.

    Attributes:
        model: The model identifier.
        scenario_results: Individual per-scenario outcomes.
        base_url: The API base URL used.
    """

    model: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    base_url: str = ""

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def total_scenarios(self) -> int:
        """Total number of scenarios run."""
        return len(self.scenario_results)

    @property
    def passed_count(self) -> int:
        """Number of scenarios where the model produced correct tool calls."""
        return sum(1 for r in self.scenario_results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of scenarios where the model produced incorrect tool calls."""
        return self.total_scenarios - self.passed_count

    @property
    def tool_call_accuracy(self) -> float:
        """Fraction of scenarios where tool selection was entirely correct."""
        if not self.scenario_results:
            return 0.0
        return self.passed_count / self.total_scenarios

    @property
    def parameter_accuracy(self) -> float:
        """Mean per-parameter match score across all scenarios."""
        scores = [
            score
            for r in self.scenario_results
            for score in r.parameter_scores.values()
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of hallucination-resistance scenarios where the model hallucinated."""
        hr = [
            r
            for r in self.scenario_results
            if r.scenario.category == "hallucination_resistance"
        ]
        if not hr:
            return 0.0
        # Hallucinated = scenario expected no calls but model produced some
        hallucinated = sum(
            1 for r in hr if not r.passed and len(r.actual_tool_calls) > 0
        )
        return hallucinated / len(hr)

    @property
    def avg_latency_ms(self) -> float:
        """Average API round-trip latency across all scenarios."""
        latencies = [r.latency_ms for r in self.scenario_results if r.latency_ms > 0]
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed across all scenarios."""
        return sum(r.total_tokens for r in self.scenario_results)

    @property
    def error_count(self) -> int:
        """Number of scenarios that failed due to an API or runtime error."""
        return sum(1 for r in self.scenario_results if r.error is not None)

    def by_category(self) -> dict[str, dict[str, Any]]:
        """Return accuracy metrics broken down by scenario category.

        Returns:
            Dict mapping category name → metrics dict with keys
            ``total``, ``passed``, ``accuracy``.
        """
        categories: dict[str, list[ScenarioResult]] = {}
        for r in self.scenario_results:
            categories.setdefault(r.scenario.category, []).append(r)
        return {
            cat: {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "accuracy": sum(1 for r in results if r.passed) / len(results),
            }
            for cat, results in categories.items()
        }

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns:
            Multi-line summary with key metrics.
        """
        lines = [
            f"Model: {self.model}",
            f"Scenarios: {self.total_scenarios} total, {self.passed_count} passed, "
            f"{self.failed_count} failed",
            f"Tool call accuracy: {self.tool_call_accuracy:.1%}",
            f"Parameter accuracy: {self.parameter_accuracy:.1%}",
            f"Hallucination rate: {self.hallucination_rate:.1%}",
            f"Avg latency: {self.avg_latency_ms:.0f} ms",
            f"Total tokens: {self.total_tokens_used}",
        ]
        categories = self.by_category()
        if categories:
            lines.append("By category:")
            for cat, metrics in sorted(categories.items()):
                lines.append(
                    f"  {cat}: {metrics['passed']}/{metrics['total']} "
                    f"({metrics['accuracy']:.1%})"
                )
        return "\n".join(lines)

    def to_report(self) -> "BenchmarkReport":
        """Convert this results object into a :class:`BenchmarkReport`.

        Returns:
            A :class:`BenchmarkReport` that can be saved or displayed.
        """
        from agentguard.benchmark.report import BenchmarkReport

        return BenchmarkReport(results=[self])


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class ModelComparison:
    """Side-by-side comparison of two or more :class:`BenchmarkResults` objects.

    Attributes:
        results: The per-model results being compared.
    """

    results: list[BenchmarkResults]

    def summary(self) -> str:
        """Return a formatted comparison table.

        Returns:
            Multi-line string with one row per model.
        """
        header = (
            f"{'Model':<40} {'Accuracy':>9} {'Param Acc':>10} "
            f"{'Halluc Rate':>12} {'Avg Latency':>12} {'Tokens':>8}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.results:
            rows.append(
                f"{r.model:<40} {r.tool_call_accuracy:>9.1%} "
                f"{r.parameter_accuracy:>10.1%} {r.hallucination_rate:>12.1%} "
                f"{r.avg_latency_ms:>11.0f}ms {r.total_tokens_used:>8}"
            )
        return "\n".join(rows)

    def winner(self) -> Optional[str]:
        """Return the model name with the highest overall tool call accuracy.

        Returns:
            Model name, or None if results list is empty.
        """
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.tool_call_accuracy).model

    def to_dict(self) -> dict[str, Any]:
        """Serialise comparison to a plain dict."""
        from agentguard.benchmark.report import BenchmarkReport

        return {
            "models": [r.model for r in self.results],
            "winner": self.winner(),
            "per_model": {
                r.model: {
                    "tool_call_accuracy": r.tool_call_accuracy,
                    "parameter_accuracy": r.parameter_accuracy,
                    "hallucination_rate": r.hallucination_rate,
                    "avg_latency_ms": r.avg_latency_ms,
                    "total_tokens": r.total_tokens_used,
                    "by_category": r.by_category(),
                }
                for r in self.results
            },
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Run benchmark scenarios against one or more LLM models.

    The runner maintains an ordered list of :class:`~agentguard.benchmark.scenarios.BenchmarkScenario`
    objects and can execute them against any OpenAI-compatible API endpoint.

    Usage::

        import os
        runner = BenchmarkRunner(max_tokens=512, temperature=0.0)
        runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
        runner.add_scenarios(BuiltinScenarios.PARAMETER_EXTRACTION)

        results = runner.run(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )

    Args:
        max_tokens: Maximum tokens for each model response.
        temperature: Sampling temperature (0.0 for deterministic).
        timeout: HTTP timeout in seconds for each API call.
        verbose: Print progress to stdout when True.
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        verbose: bool = False,
    ) -> None:
        self._scenarios: list[BenchmarkScenario] = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------

    def add_scenario(self, scenario: BenchmarkScenario) -> None:
        """Add a single scenario to the runner.

        Args:
            scenario: The :class:`~agentguard.benchmark.scenarios.BenchmarkScenario`
                to add.
        """
        self._scenarios.append(scenario)

    def add_scenarios(
        self,
        scenarios: "list[BenchmarkScenario] | type[Any]",
    ) -> None:
        """Add multiple scenarios to the runner.

        Accepts either a plain list or a :class:`BuiltinScenarios` static method
        (which is callable and returns a list).

        Args:
            scenarios: A list of scenarios, or a callable returning a list.
        """
        if callable(scenarios):
            scenarios = scenarios()
        self._scenarios.extend(scenarios)

    def clear_scenarios(self) -> None:
        """Remove all scenarios from the runner."""
        self._scenarios.clear()

    @property
    def scenarios(self) -> list[BenchmarkScenario]:
        """Return the current list of scenarios."""
        return list(self._scenarios)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        extra_headers: Optional[dict[str, str]] = None,
        scenario_filter: Optional[list[str]] = None,
    ) -> BenchmarkResults:
        """Run all scenarios against a model and return aggregate results.

        Args:
            model: Model identifier (e.g. ``"openai/gpt-4o-mini"``).
            base_url: Base URL of the OpenAI-compatible API endpoint.
            api_key: API key for authentication.
            extra_headers: Additional HTTP headers to send with each request.
            scenario_filter: Optional list of scenario names to run. When
                provided, only matching scenarios are executed.

        Returns:
            A :class:`BenchmarkResults` object with per-scenario outcomes and
            aggregate metrics.

        Raises:
            ImportError: If the ``openai`` package is not installed.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to run benchmarks. "
                "Install it with: pip install openai"
            ) from exc

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "placeholder",
            timeout=self.timeout,
            default_headers=extra_headers or {},
        )

        scenarios = self._scenarios
        if scenario_filter:
            scenarios = [s for s in scenarios if s.name in scenario_filter]

        aggregate = BenchmarkResults(model=model, base_url=base_url)

        for i, scenario in enumerate(scenarios):
            if self.verbose:
                print(f"[{i + 1}/{len(scenarios)}] Running: {scenario.name} ...", end=" ", flush=True)

            result = self._run_single(client, model, scenario)
            aggregate.scenario_results.append(result)

            if self.verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"{status}  ({result.latency_ms:.0f} ms)")

        return aggregate

    def compare(self, results_list: list[BenchmarkResults]) -> ModelComparison:
        """Compare multiple :class:`BenchmarkResults` objects side by side.

        Args:
            results_list: Results from two or more ``runner.run(...)`` calls.

        Returns:
            A :class:`ModelComparison` object with a summary table and winner.
        """
        return ModelComparison(results=results_list)

    # ------------------------------------------------------------------
    # Internal: single scenario execution
    # ------------------------------------------------------------------

    def _run_single(
        self,
        client: Any,
        model: str,
        scenario: BenchmarkScenario,
    ) -> ScenarioResult:
        """Execute one scenario and return its result."""
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=scenario.messages,
                tools=scenario.tools if scenario.tools else openai_tools_none(),
                tool_choice="auto",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            return ScenarioResult(
                scenario=scenario,
                model=model,
                passed=False,
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract token counts
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Parse actual tool calls from the response
        actual_calls = _parse_tool_calls(response)

        # Validate
        passed, param_scores = _validate_scenario(scenario, actual_calls)

        return ScenarioResult(
            scenario=scenario,
            model=model,
            passed=passed,
            actual_tool_calls=actual_calls,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            parameter_scores=param_scores,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def openai_tools_none() -> None:  # type: ignore[return]
    """Sentinel used when no tools should be provided to the API."""
    return None  # type: ignore[return-value]


def _parse_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI chat completion response.

    Args:
        response: An ``openai.types.chat.ChatCompletion`` object.

    Returns:
        List of dicts with ``"name"`` and ``"arguments"`` keys.
    """
    calls: list[dict[str, Any]] = []
    try:
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return calls
        for tc in tool_calls:
            fn = tc.function
            args: dict[str, Any] = {}
            try:
                args = json.loads(fn.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": fn.arguments}
            calls.append({"name": fn.name, "arguments": args})
    except (IndexError, AttributeError):
        pass
    return calls


def _validate_scenario(
    scenario: BenchmarkScenario,
    actual_calls: list[dict[str, Any]],
) -> tuple[bool, dict[str, float]]:
    """Validate actual tool calls against scenario expectations.

    Args:
        scenario: The scenario being validated.
        actual_calls: Tool calls produced by the model.

    Returns:
        Tuple of (passed: bool, param_scores: dict[str, float]).
    """
    # Custom validator takes precedence
    if scenario.validate_fn is not None:
        try:
            passed = bool(scenario.validate_fn(actual_calls))
        except Exception:  # noqa: BLE001
            passed = False
        return passed, {}

    expected = scenario.expected_tool_calls

    # No tools expected → model should produce no calls
    if not expected:
        return len(actual_calls) == 0, {}

    # At least one tool expected
    if not actual_calls:
        return False, {}

    # Check that every expected call appears in actuals (order-insensitive)
    param_scores: dict[str, float] = {}
    matched_expected = 0

    for exp in expected:
        best_score, matched = _match_expected_call(exp, actual_calls)
        if matched:
            matched_expected += 1
        # Record parameter scores
        exp_args = exp.get("arguments", {})
        for key, exp_val in exp_args.items():
            score_key = f"{exp['name']}.{key}"
            param_scores[score_key] = best_score

    passed = matched_expected == len(expected)
    return passed, param_scores


def _match_expected_call(
    expected: dict[str, Any],
    actual_calls: list[dict[str, Any]],
) -> tuple[float, bool]:
    """Find the best-matching actual call for one expected call.

    Args:
        expected: Expected tool call with ``"name"`` and ``"arguments"``.
        actual_calls: List of actual tool calls.

    Returns:
        Tuple of (best_param_score: float, matched: bool).
    """
    expected_name = expected.get("name", "")
    expected_args = expected.get("arguments", {})

    best_score = 0.0
    best_matched = False

    for actual in actual_calls:
        if actual.get("name") != expected_name:
            continue
        actual_args = actual.get("arguments", {})
        score = _score_arguments(expected_args, actual_args)
        if score > best_score:
            best_score = score
            # Consider a match if all required args are at ≥ 0.5 similarity
            best_matched = score >= 0.5 or (not expected_args and not actual_args)

    return best_score, best_matched


def _score_arguments(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> float:
    """Compute a [0, 1] similarity score between expected and actual arguments.

    Args:
        expected: Expected argument dict.
        actual: Actual argument dict.

    Returns:
        Score from 0.0 (no match) to 1.0 (perfect match).
    """
    if not expected and not actual:
        return 1.0
    if not expected:
        return 1.0  # No args expected — any args are OK
    if not actual:
        return 0.0

    total_score = 0.0
    for key, exp_val in expected.items():
        if key not in actual:
            continue
        act_val = actual[key]
        total_score += _value_similarity(exp_val, act_val)

    return total_score / len(expected)


def _value_similarity(expected: Any, actual: Any) -> float:
    """Compute similarity between two argument values.

    Handles strings (fuzzy), numbers (exact), lists, and dicts.

    Special case: math expressions like ``"2+2"`` and ``"2 + 2"`` score 1.0
    because they are semantically identical (produce the same numeric result).

    Args:
        expected: Expected value.
        actual: Actual value.

    Returns:
        Similarity score from 0.0 to 1.0.
    """
    if expected == actual:
        return 1.0

    # Numeric: allow small relative tolerance
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            return 1.0 if actual == 0 else 0.0
        diff = abs(expected - actual) / abs(expected)
        return max(0.0, 1.0 - diff)

    # String: case-insensitive contains check
    if isinstance(expected, str) and isinstance(actual, str):
        e_lower = expected.lower().strip()
        a_lower = actual.lower().strip()
        if e_lower == a_lower:
            return 1.0

        # Math expression comparison: strip spaces and evaluate numerically
        # "2+2" and "2 + 2" are semantically identical
        if e_lower.replace(" ", "") == a_lower.replace(" ", ""):
            return 1.0
        exp_numeric = _eval_math_expr(e_lower)
        act_numeric = _eval_math_expr(a_lower)
        if exp_numeric is not None and act_numeric is not None:
            if abs(exp_numeric - act_numeric) < 1e-9:
                return 1.0  # Same numeric value — semantically identical

        if e_lower in a_lower or a_lower in e_lower:
            return 0.8
        # Jaccard similarity on word sets
        e_words = set(e_lower.split())
        a_words = set(a_lower.split())
        if not e_words and not a_words:
            return 1.0
        if not e_words or not a_words:
            return 0.0
        intersection = e_words & a_words
        union = e_words | a_words
        return len(intersection) / len(union)

    # List: set-based overlap
    if isinstance(expected, list) and isinstance(actual, list):
        if not expected:
            return 1.0
        exp_set = {str(v).lower() for v in expected}
        act_set = {str(v).lower() for v in actual}
        if not exp_set:
            return 1.0
        return len(exp_set & act_set) / len(exp_set)

    return 0.0


def _eval_math_expr(expr: str) -> "Optional[float]":
    """Safely evaluate a simple arithmetic expression string.

    Only allows digits, arithmetic operators, parentheses, dots, and spaces.
    Returns None if the expression is not a safe arithmetic expression.

    Args:
        expr: Expression string to evaluate.

    Returns:
        Numeric result as float, or None.
    """
    import re as _re
    if not _re.fullmatch(r"[\d\s\.\+\-\*\/\(\)\^\%]+", expr.strip()):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return float(result)
    except Exception:
        return None
