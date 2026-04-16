"""Tests for agentguard.benchmark — BenchmarkRunner, scenarios, and report.

These tests do NOT make real API calls. All LLM interactions are mocked
via unittest.mock so the suite runs offline and deterministically.

Coverage:
- BenchmarkScenario dataclass construction
- BuiltinScenarios factory methods and category coverage
- BenchmarkRunner scenario management (add, clear, filter)
- _parse_tool_calls helper
- _validate_scenario helper (exact match, fuzzy match, no-call expected)
- _value_similarity helper (strings, numbers, lists)
- BenchmarkResults aggregate metrics
- BenchmarkReport serialisation (to_dict, save, load, summary)
- CLI cmd_benchmark
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentguard.benchmark import (
    BenchmarkReport,
    BenchmarkResults,
    BenchmarkRunner,
    BenchmarkScenario,
    BuiltinScenarios,
    DEFAULT_BENCHMARK_ARTIFACTS_DIR,
    ModelComparison,
    ScenarioResult,
)
from agentguard.benchmark.runner import (
    _parse_tool_calls,
    _validate_scenario,
    _value_similarity,
)


# ---------------------------------------------------------------------------
# BenchmarkScenario
# ---------------------------------------------------------------------------


class TestBenchmarkScenario:
    """Basic construction and attribute access."""

    def test_minimal_construction(self) -> None:
        s = BenchmarkScenario(
            name="test_basic",
            description="A simple test",
            category="basic",
            tools=[],
            messages=[{"role": "user", "content": "Hi"}],
            expected_tool_calls=[],
        )
        assert s.name == "test_basic"
        assert s.category == "basic"
        assert s.validate_fn is None
        assert s.tags == []

    def test_with_validate_fn(self) -> None:
        fn = lambda calls: True  # noqa: E731
        s = BenchmarkScenario(
            name="custom",
            description="",
            category="custom",
            tools=[],
            messages=[],
            expected_tool_calls=[],
            validate_fn=fn,
            tags=["special"],
        )
        assert s.validate_fn is fn
        assert "special" in s.tags


# ---------------------------------------------------------------------------
# BuiltinScenarios
# ---------------------------------------------------------------------------


class TestBuiltinScenarios:
    """All built-in scenario collections return non-empty lists."""

    def test_basic_tool_calling_nonempty(self) -> None:
        scenarios = BuiltinScenarios.BASIC_TOOL_CALLING()
        assert len(scenarios) >= 3
        assert all(s.category == "basic" for s in scenarios)

    def test_multi_tool_selection_nonempty(self) -> None:
        scenarios = BuiltinScenarios.MULTI_TOOL_SELECTION()
        assert len(scenarios) >= 2
        assert all(s.category == "multi_tool" for s in scenarios)

    def test_parameter_extraction_nonempty(self) -> None:
        scenarios = BuiltinScenarios.PARAMETER_EXTRACTION()
        assert len(scenarios) >= 3
        assert all(s.category == "parameter_extraction" for s in scenarios)

    def test_hallucination_resistance_nonempty(self) -> None:
        scenarios = BuiltinScenarios.HALLUCINATION_RESISTANCE()
        assert len(scenarios) >= 1
        assert all(s.category == "hallucination_resistance" for s in scenarios)

    def test_error_handling_nonempty(self) -> None:
        scenarios = BuiltinScenarios.ERROR_HANDLING()
        assert len(scenarios) >= 1

    def test_all_contains_at_least_15_scenarios(self) -> None:
        all_scenarios = BuiltinScenarios.ALL()
        assert len(all_scenarios) >= 15

    def test_all_names_are_unique(self) -> None:
        all_scenarios = BuiltinScenarios.ALL()
        names = [s.name for s in all_scenarios]
        assert len(names) == len(set(names)), "Duplicate scenario names found"

    def test_every_scenario_has_tools(self) -> None:
        """Every scenario that expects tool calls must have at least one tool."""
        for s in BuiltinScenarios.ALL():
            if s.expected_tool_calls:
                assert s.tools, f"Scenario '{s.name}' expects calls but has no tools"


# ---------------------------------------------------------------------------
# BenchmarkRunner — scenario management
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerManagement:
    """add_scenario, add_scenarios, clear_scenarios, and scenarios property."""

    def test_add_single_scenario(self) -> None:
        runner = BenchmarkRunner()
        s = BenchmarkScenario("s1", "", "basic", [], [], [])
        runner.add_scenario(s)
        assert len(runner.scenarios) == 1
        assert runner.scenarios[0].name == "s1"

    def test_add_scenarios_from_list(self) -> None:
        runner = BenchmarkRunner()
        scenarios = [
            BenchmarkScenario(f"s{i}", "", "basic", [], [], [])
            for i in range(3)
        ]
        runner.add_scenarios(scenarios)
        assert len(runner.scenarios) == 3

    def test_add_scenarios_from_callable(self) -> None:
        """add_scenarios accepts a callable (BuiltinScenarios static methods)."""
        runner = BenchmarkRunner()
        runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
        assert len(runner.scenarios) >= 3

    def test_clear_scenarios(self) -> None:
        runner = BenchmarkRunner()
        runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
        runner.clear_scenarios()
        assert runner.scenarios == []

    def test_scenarios_returns_copy(self) -> None:
        """Mutating the returned list must not affect the runner's internal state."""
        runner = BenchmarkRunner()
        runner.add_scenario(BenchmarkScenario("s1", "", "basic", [], [], []))
        copy = runner.scenarios
        copy.clear()
        assert len(runner.scenarios) == 1


# ---------------------------------------------------------------------------
# _parse_tool_calls helper
# ---------------------------------------------------------------------------


def _mock_response(tool_name: str, args: dict[str, Any]) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = json.dumps(args)
    tc = MagicMock()
    tc.function = fn
    msg = MagicMock()
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_no_tool_response() -> MagicMock:
    """Build a mock response with no tool calls."""
    msg = MagicMock()
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestParseToolCalls:
    def test_single_tool_call_parsed(self) -> None:
        resp = _mock_response("get_weather", {"city": "London"})
        calls = _parse_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "London"

    def test_no_tool_calls_returns_empty(self) -> None:
        resp = _mock_no_tool_response()
        calls = _parse_tool_calls(resp)
        assert calls == []

    def test_invalid_json_args_stored_as_raw(self) -> None:
        fn = MagicMock()
        fn.name = "my_tool"
        fn.arguments = "not valid json"
        tc = MagicMock()
        tc.function = fn
        msg = MagicMock()
        msg.tool_calls = [tc]
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        calls = _parse_tool_calls(resp)
        assert calls[0]["arguments"].get("_raw") == "not valid json"


# ---------------------------------------------------------------------------
# _validate_scenario helper
# ---------------------------------------------------------------------------


class TestValidateScenario:
    """validate_scenario returns (passed, param_scores)."""

    def _make_scenario(
        self,
        expected: list[dict[str, Any]],
        validate_fn: Any = None,
    ) -> BenchmarkScenario:
        return BenchmarkScenario(
            name="test",
            description="",
            category="test",
            tools=[],
            messages=[],
            expected_tool_calls=expected,
            validate_fn=validate_fn,
        )

    def test_exact_match_passes(self) -> None:
        s = self._make_scenario([{"name": "get_weather", "arguments": {"city": "London"}}])
        actual = [{"name": "get_weather", "arguments": {"city": "London"}}]
        passed, scores = _validate_scenario(s, actual)
        assert passed is True

    def test_wrong_tool_name_fails(self) -> None:
        s = self._make_scenario([{"name": "get_weather", "arguments": {"city": "London"}}])
        actual = [{"name": "calculate", "arguments": {"expression": "2+2"}}]
        passed, _ = _validate_scenario(s, actual)
        assert passed is False

    def test_no_expected_no_actual_passes(self) -> None:
        """Scenario expects no calls AND model produces none → pass."""
        s = self._make_scenario([])
        passed, _ = _validate_scenario(s, [])
        assert passed is True

    def test_no_expected_but_actual_exists_fails(self) -> None:
        """Scenario expects no calls but model produces one → fail."""
        s = self._make_scenario([])
        actual = [{"name": "get_weather", "arguments": {}}]
        passed, _ = _validate_scenario(s, actual)
        assert passed is False

    def test_custom_validate_fn_overrides(self) -> None:
        """Custom validate_fn takes precedence over default matching."""
        s = self._make_scenario(
            [{"name": "get_weather", "arguments": {}}],
            validate_fn=lambda _calls: False,  # Always fail
        )
        actual = [{"name": "get_weather", "arguments": {"city": "London"}}]
        passed, _ = _validate_scenario(s, actual)
        assert passed is False


# ---------------------------------------------------------------------------
# _value_similarity helper
# ---------------------------------------------------------------------------


class TestValueSimilarity:
    def test_equal_strings_score_1(self) -> None:
        assert _value_similarity("London", "London") == pytest.approx(1.0)

    def test_case_insensitive_strings_score_1(self) -> None:
        assert _value_similarity("london", "LONDON") == pytest.approx(1.0)

    def test_substring_match_high_score(self) -> None:
        score = _value_similarity("New York City", "New York")
        assert score > 0.5

    def test_completely_different_strings_low_score(self) -> None:
        score = _value_similarity("apple", "zeppelin")
        assert score < 0.5

    def test_equal_numbers_score_1(self) -> None:
        assert _value_similarity(42, 42) == pytest.approx(1.0)

    def test_close_numbers_high_score(self) -> None:
        score = _value_similarity(100.0, 99.0)
        assert score > 0.9

    def test_far_numbers_low_score(self) -> None:
        score = _value_similarity(100.0, 1.0)
        assert score < 0.5

    def test_equal_lists_score_1(self) -> None:
        assert _value_similarity(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_partial_list_overlap(self) -> None:
        score = _value_similarity(["a", "b", "c"], ["a", "b"])
        assert 0.5 < score < 1.0


# ---------------------------------------------------------------------------
# BenchmarkResults aggregate metrics
# ---------------------------------------------------------------------------


def _make_results(model: str = "test-model") -> BenchmarkResults:
    """Build a BenchmarkResults with a mix of passing and failing scenarios."""
    results = BenchmarkResults(model=model, base_url="https://example.com")

    s1 = BenchmarkScenario("s1", "", "basic", [], [], [])
    s2 = BenchmarkScenario("s2", "", "basic", [], [], [])
    s3 = BenchmarkScenario("s3", "", "hallucination_resistance", [], [], [])

    results.scenario_results = [
        ScenarioResult(
            scenario=s1,
            model=model,
            passed=True,
            latency_ms=100.0,
            prompt_tokens=50,
            completion_tokens=20,
        ),
        ScenarioResult(
            scenario=s2,
            model=model,
            passed=False,
            latency_ms=200.0,
            prompt_tokens=60,
            completion_tokens=30,
        ),
        ScenarioResult(
            scenario=s3,
            model=model,
            passed=False,
            actual_tool_calls=[{"name": "hallucinated_tool", "arguments": {}}],
            latency_ms=150.0,
        ),
    ]
    return results


class TestBenchmarkResultsMetrics:
    def test_tool_call_accuracy(self) -> None:
        r = _make_results()
        assert r.tool_call_accuracy == pytest.approx(1 / 3)

    def test_passed_and_failed_counts(self) -> None:
        r = _make_results()
        assert r.passed_count == 1
        assert r.failed_count == 2

    def test_hallucination_rate(self) -> None:
        r = _make_results()
        # 1 hallucination-resistance scenario, 1 hallucinated (failed + actual calls)
        assert r.hallucination_rate == pytest.approx(1.0)

    def test_avg_latency(self) -> None:
        r = _make_results()
        expected_avg = (100.0 + 200.0 + 150.0) / 3
        assert r.avg_latency_ms == pytest.approx(expected_avg)

    def test_total_tokens(self) -> None:
        r = _make_results()
        assert r.total_tokens_used == 50 + 20 + 60 + 30  # 160

    def test_by_category(self) -> None:
        r = _make_results()
        cats = r.by_category()
        assert "basic" in cats
        assert cats["basic"]["total"] == 2
        assert cats["basic"]["passed"] == 1

    def test_summary_string_contains_model(self) -> None:
        r = _make_results("my-model")
        summary = r.summary()
        assert "my-model" in summary

    def test_error_count(self) -> None:
        r = _make_results()
        r.scenario_results[0].error = "connection refused"
        assert r.error_count == 1


# ---------------------------------------------------------------------------
# BenchmarkReport serialisation
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_to_dict_contains_expected_keys(self) -> None:
        r = _make_results()
        report = BenchmarkReport(results=[r])
        d = report.to_dict()
        assert "title" in d
        assert "generated_at" in d
        assert "models" in d
        assert "per_model" in d

    def test_save_and_load_roundtrip(self) -> None:
        """save() writes valid JSON that load() can parse."""
        r = _make_results("gpt-test")
        report = BenchmarkReport(results=[r], title="Test Report")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            out = report.save(path)
            assert out.exists()
            with open(path) as fh:
                data = json.load(fh)
            assert data["title"] == "Test Report"
            assert "gpt-test" in data["models"]
        finally:
            os.unlink(path)

    def test_from_comparison(self) -> None:
        r1 = _make_results("model-a")
        r2 = _make_results("model-b")
        comparison = ModelComparison(results=[r1, r2])
        report = BenchmarkReport.from_comparison(comparison)
        assert isinstance(report, BenchmarkReport)

    def test_summary_includes_model_names(self) -> None:
        r = _make_results("fancy-model")
        report = BenchmarkReport(results=[r])
        summary = report.summary()
        assert "fancy-model" in summary

    def test_save_accepts_directory_path(self) -> None:
        r = _make_results("openai/gpt-4o-mini")
        report = BenchmarkReport(results=[r])
        with tempfile.TemporaryDirectory() as tmp:
            out = report.save(tmp)
            assert str(out.parent) == os.path.realpath(tmp)
            assert out.name == "openai-gpt-4o-mini.json"
            assert out.exists()

    def test_save_without_path_uses_default_artifact_layout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        r = _make_results("gpt-test")
        report = BenchmarkReport(results=[r])
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.chdir(tmp)
            out = report.save()
            assert DEFAULT_BENCHMARK_ARTIFACTS_DIR.as_posix() in str(out)
            assert out.name == "gpt-test.json"
            assert out.exists()


# ---------------------------------------------------------------------------
# ModelComparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_winner_is_higher_accuracy_model(self) -> None:
        r1 = _make_results("weak-model")
        r2 = BenchmarkResults(model="strong-model")
        s = BenchmarkScenario("x", "", "basic", [], [], [])
        r2.scenario_results = [
            ScenarioResult(scenario=s, model="strong-model", passed=True, latency_ms=100)
        ]
        comparison = ModelComparison(results=[r1, r2])
        # r2 has 1/1 = 100% accuracy vs r1's ~33%
        assert comparison.winner() == "strong-model"

    def test_summary_contains_model_names(self) -> None:
        r1 = _make_results("alpha")
        r2 = _make_results("beta")
        comparison = ModelComparison(results=[r1, r2])
        summary = comparison.summary()
        assert "alpha" in summary
        assert "beta" in summary

    def test_to_dict_structure(self) -> None:
        r1 = _make_results("m1")
        comparison = ModelComparison(results=[r1])
        d = comparison.to_dict()
        assert "models" in d
        assert "per_model" in d
        assert "winner" in d


# ---------------------------------------------------------------------------
# BenchmarkRunner.run (mocked API calls)
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerRunMocked:
    """Tests for the full run() path using mocked OpenAI clients."""

    def _make_mock_client(self, tool_name: str, args: dict[str, Any]) -> MagicMock:
        """Build a mock OpenAI client that always returns the specified tool call."""
        resp = _mock_response(tool_name, args)
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    @patch("agentguard.benchmark.runner.openai_tools_none", return_value=None)
    def test_run_single_scenario_pass(self, _: Any) -> None:
        """When the model returns the expected tool call, the scenario passes."""
        runner = BenchmarkRunner()
        scenario = BenchmarkScenario(
            name="test_weather",
            description="",
            category="basic",
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            messages=[{"role": "user", "content": "Weather in London?"}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "London"}}],
        )
        runner.add_scenario(scenario)

        mock_client = self._make_mock_client("get_weather", {"city": "London"})

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=mock_client))}):
            results = runner.run(
                model="gpt-test",
                base_url="https://example.com/v1",
                api_key="fake-key",
            )

        assert results.total_scenarios == 1
        assert results.passed_count == 1
        assert results.tool_call_accuracy == pytest.approx(1.0)

    @patch("agentguard.benchmark.runner.openai_tools_none", return_value=None)
    def test_run_single_scenario_fail(self, _: Any) -> None:
        """When the model returns the wrong tool, the scenario fails."""
        runner = BenchmarkRunner()
        scenario = BenchmarkScenario(
            name="test_wrong",
            description="",
            category="basic",
            tools=[],
            messages=[{"role": "user", "content": "Weather?"}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {}}],
        )
        runner.add_scenario(scenario)

        # Model returns calculate instead of get_weather
        mock_client = self._make_mock_client("calculate", {"expression": "2+2"})

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=mock_client))}):
            results = runner.run(
                model="gpt-test",
                base_url="https://example.com/v1",
                api_key="fake-key",
            )

        assert results.passed_count == 0
        assert results.failed_count == 1

    def test_run_raises_import_error_without_openai(self) -> None:
        """runner.run raises ImportError when openai is not installed."""
        runner = BenchmarkRunner()
        runner.add_scenario(
            BenchmarkScenario("s", "", "basic", [], [], [])
        )
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                runner.run(model="x", api_key="k")

    @patch("agentguard.benchmark.runner.openai_tools_none", return_value=None)
    def test_run_handles_api_error_gracefully(self, _: Any) -> None:
        """API errors are captured as error records (not unhandled exceptions)."""
        runner = BenchmarkRunner()
        runner.add_scenario(
            BenchmarkScenario("s", "", "basic", [], [{"role": "user", "content": "hi"}], [])
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate limit")

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=mock_client))}):
            results = runner.run(model="m", api_key="k")

        assert results.total_scenarios == 1
        assert results.error_count == 1

    def test_compare_returns_model_comparison(self) -> None:
        r1 = _make_results("m1")
        r2 = _make_results("m2")
        runner = BenchmarkRunner()
        comparison = runner.compare([r1, r2])
        assert isinstance(comparison, ModelComparison)
        assert len(comparison.results) == 2
