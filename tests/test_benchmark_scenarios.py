"""Tests verifying the fixed benchmark scenarios.

Covers:
- Fixed scenarios have unambiguous expectations
- Math expression matching works properly (2+2 == 2 + 2)
- _value_similarity handles expression equivalence
- New hallucination_resistance scenarios are well-formed
- _validate_math_expression validator works correctly
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tests for _value_similarity math expression handling
# ---------------------------------------------------------------------------


class TestValueSimilarityMath:
    """Test that _value_similarity scores math expressions correctly."""

    def _sim(self, a, b) -> float:
        from agentguard.benchmark.runner import _value_similarity
        return _value_similarity(a, b)

    def test_identical_expressions(self):
        """Identical strings score 1.0."""
        assert self._sim("2+2", "2+2") == 1.0

    def test_spaced_vs_unspaced_expression(self):
        """'2+2' and '2 + 2' are semantically identical → score 1.0."""
        assert self._sim("2+2", "2 + 2") == 1.0
        assert self._sim("2 + 2", "2+2") == 1.0

    def test_expression_with_extra_spaces(self):
        """Multiple spaces should still match."""
        assert self._sim("2+2", "2  +  2") == 1.0

    def test_division_equivalence(self):
        """'1000/8' and '1000 / 8' should match."""
        assert self._sim("1000/8", "1000 / 8") == 1.0

    def test_multiplication_equivalence(self):
        """'5*7' and '5 * 7' should match."""
        assert self._sim("5*7", "5 * 7") == 1.0

    def test_numeric_result_equivalence(self):
        """Expressions with same result but different form should match."""
        assert self._sim("2+2", "4") == pytest.approx(1.0, abs=0.01)

    def test_non_math_strings_not_affected(self):
        """Non-math strings should use normal similarity, not numeric eval."""
        score = self._sim("London", "london")
        assert score == 1.0  # case-insensitive

    def test_different_expressions(self):
        """Expressions producing different values should NOT score 1.0."""
        assert self._sim("2+2", "3+3") < 1.0

    def test_completely_unrelated(self):
        """Completely different non-math strings should score low."""
        score = self._sim("apple", "xyz123")
        assert score < 0.5


# ---------------------------------------------------------------------------
# Tests for _validate_math_expression validator
# ---------------------------------------------------------------------------


class TestValidateMathExpression:
    """Test the _validate_math_expression validator factory."""

    def _make_validator(self, expected: str):
        from agentguard.benchmark.scenarios import _validate_math_expression
        return _validate_math_expression(expected)

    def test_exact_match(self):
        """Exact string match should pass."""
        v = self._make_validator("2+2")
        calls = [{"name": "calculate", "arguments": {"expression": "2+2"}}]
        assert v(calls)

    def test_spaced_match(self):
        """'2 + 2' should match when expected is '2+2'."""
        v = self._make_validator("2+2")
        calls = [{"name": "calculate", "arguments": {"expression": "2 + 2"}}]
        assert v(calls)

    def test_numeric_equivalence(self):
        """Expressions with same numeric result should pass."""
        v = self._make_validator("1000/8")
        calls = [{"name": "calculate", "arguments": {"expression": "1000 / 8"}}]
        assert v(calls)

    def test_wrong_expression_fails(self):
        """Different math expression should fail."""
        v = self._make_validator("2+2")
        calls = [{"name": "calculate", "arguments": {"expression": "3+3"}}]
        assert not v(calls)

    def test_wrong_tool_name_fails(self):
        """Call to wrong tool should not match."""
        v = self._make_validator("2+2")
        calls = [{"name": "search_web", "arguments": {"query": "2+2"}}]
        assert not v(calls)

    def test_empty_calls_fails(self):
        """Empty call list should fail."""
        v = self._make_validator("2+2")
        assert not v([])


# ---------------------------------------------------------------------------
# Tests for hallucination resistance scenarios
# ---------------------------------------------------------------------------


class TestHallucinationResistanceScenarios:
    """Verify the new hallucination resistance scenarios are well-formed."""

    def _get_scenarios(self):
        from agentguard.benchmark.scenarios import BuiltinScenarios
        return BuiltinScenarios.HALLUCINATION_RESISTANCE()

    def test_at_least_5_scenarios(self):
        """There should be at least 5 hallucination resistance scenarios."""
        scenarios = self._get_scenarios()
        assert len(scenarios) >= 5, (
            f"Expected ≥5 scenarios, got {len(scenarios)}"
        )

    def test_no_ambiguous_scenario(self):
        """The ambiguous 'meaning of life' scenario should have been removed."""
        scenarios = self._get_scenarios()
        names = {s.name for s in scenarios}
        assert "hallucination_irrelevant_query" not in names, (
            "Ambiguous 'hallucination_irrelevant_query' scenario should be removed"
        )

    def test_all_expect_no_tool_calls(self):
        """All hallucination resistance scenarios should expect zero tool calls."""
        scenarios = self._get_scenarios()
        for s in scenarios:
            assert s.expected_tool_calls == [], (
                f"Scenario '{s.name}' should expect no tool calls"
            )

    def test_all_have_validate_fn(self):
        """All scenarios should have a validate_fn for strict checking."""
        scenarios = self._get_scenarios()
        for s in scenarios:
            assert s.validate_fn is not None, (
                f"Scenario '{s.name}' should have a validate_fn"
            )

    def test_validate_fn_rejects_tool_calls(self):
        """validate_fn should reject any tool calls."""
        scenarios = self._get_scenarios()
        fake_call = [{"name": "some_tool", "arguments": {}}]
        for s in scenarios:
            assert not s.validate_fn(fake_call), (
                f"Scenario '{s.name}' validate_fn should reject tool calls"
            )

    def test_validate_fn_accepts_no_calls(self):
        """validate_fn should accept empty call list."""
        scenarios = self._get_scenarios()
        for s in scenarios:
            assert s.validate_fn([]), (
                f"Scenario '{s.name}' validate_fn should accept empty calls"
            )

    def test_arithmetic_wrong_tool_scenario(self):
        """The '2+2 with wrong tool' scenario should exist and be unambiguous."""
        scenarios = self._get_scenarios()
        names = {s.name: s for s in scenarios}
        assert "hallucination_arithmetic_wrong_tool" in names
        s = names["hallucination_arithmetic_wrong_tool"]
        # Should only have weather tool
        tool_names = {t["function"]["name"] for t in s.tools}
        assert "get_weather" in tool_names
        assert "calculate" not in tool_names

    def test_joke_wrong_tool_scenario(self):
        """The 'joke with database tool' scenario should exist."""
        scenarios = self._get_scenarios()
        names = {s.name: s for s in scenarios}
        assert "hallucination_joke_wrong_tool" in names
        s = names["hallucination_joke_wrong_tool"]
        tool_names = {t["function"]["name"] for t in s.tools}
        assert "query_database" in tool_names

    def test_summarise_wrong_tool_scenario(self):
        """The 'summarise text with stock tool' scenario should exist."""
        scenarios = self._get_scenarios()
        names = {s.name: s for s in scenarios}
        assert "hallucination_summarise_wrong_tool" in names

    def test_greet_wrong_tool_scenario(self):
        """The 'greeting with flights tool' scenario should exist."""
        scenarios = self._get_scenarios()
        names = {s.name: s for s in scenarios}
        assert "hallucination_greet_wrong_tool" in names


# ---------------------------------------------------------------------------
# Tests for basic_calculate_2plus2 scenario fix
# ---------------------------------------------------------------------------


class TestBasicCalculateScenario:
    """Test that basic_calculate_2plus2 uses a flexible validator."""

    def _get_scenario(self):
        from agentguard.benchmark.scenarios import BuiltinScenarios
        scenarios = BuiltinScenarios.BASIC_TOOL_CALLING()
        return next(s for s in scenarios if s.name == "basic_calculate_2plus2")

    def test_scenario_exists(self):
        """basic_calculate_2plus2 should exist."""
        s = self._get_scenario()
        assert s.name == "basic_calculate_2plus2"

    def test_validate_fn_is_present(self):
        """basic_calculate_2plus2 should have a validate_fn."""
        s = self._get_scenario()
        assert s.validate_fn is not None

    def test_exact_expression_passes(self):
        """Exactly '2+2' should pass."""
        s = self._get_scenario()
        calls = [{"name": "calculate", "arguments": {"expression": "2+2"}}]
        assert s.validate_fn(calls)

    def test_spaced_expression_passes(self):
        """'2 + 2' should also pass (model-generated with spaces)."""
        s = self._get_scenario()
        calls = [{"name": "calculate", "arguments": {"expression": "2 + 2"}}]
        assert s.validate_fn(calls)

    def test_extra_spaces_passes(self):
        """Extra spaces should not cause failure."""
        s = self._get_scenario()
        calls = [{"name": "calculate", "arguments": {"expression": "2  +  2"}}]
        assert s.validate_fn(calls)


# ---------------------------------------------------------------------------
# Tests for multi_search_and_calc fix
# ---------------------------------------------------------------------------


class TestMultiSearchAndCalc:
    """Test that multi_search_and_calc uses a flexible calculator validator."""

    def _get_scenario(self):
        from agentguard.benchmark.scenarios import BuiltinScenarios
        scenarios = BuiltinScenarios.MULTI_TOOL_SELECTION()
        return next(s for s in scenarios if s.name == "multi_search_and_calc")

    def test_scenario_exists(self):
        """multi_search_and_calc should exist."""
        s = self._get_scenario()
        assert s.name == "multi_search_and_calc"

    def test_has_validate_fn(self):
        """Should have a custom validate_fn."""
        s = self._get_scenario()
        assert s.validate_fn is not None

    def test_exact_match_passes(self):
        """Exact match for both tools should pass."""
        s = self._get_scenario()
        calls = [
            {"name": "search_web", "arguments": {"query": "Tokyo population"}},
            {"name": "calculate", "arguments": {"expression": "1000/8"}},
        ]
        assert s.validate_fn(calls)

    def test_spaced_expression_passes(self):
        """'1000 / 8' should pass as equivalent to '1000/8'."""
        s = self._get_scenario()
        calls = [
            {"name": "search_web", "arguments": {"query": "Tokyo population"}},
            {"name": "calculate", "arguments": {"expression": "1000 / 8"}},
        ]
        assert s.validate_fn(calls)

    def test_missing_search_fails(self):
        """Without a search_web call, should fail."""
        s = self._get_scenario()
        calls = [
            {"name": "calculate", "arguments": {"expression": "1000/8"}},
        ]
        assert not s.validate_fn(calls)

    def test_missing_calc_fails(self):
        """Without a calculate call, should fail."""
        s = self._get_scenario()
        calls = [
            {"name": "search_web", "arguments": {"query": "Tokyo population"}},
        ]
        assert not s.validate_fn(calls)
