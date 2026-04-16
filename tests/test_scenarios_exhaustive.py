"""Exhaustive tests for the 50+ benchmark scenario suite.

Verifies:
- Total scenario count >= 50
- Every scenario has a name, category, tools, messages, and expected_tool_calls
- All hallucination scenarios have validate_fn checking for zero calls
- All multi-tool scenarios have validate_fn
- No duplicate scenario names
- Specific scenario names exist for each category
- Validators behave correctly (accept correct calls, reject wrong ones)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.ALL()


@pytest.fixture(scope="module")
def basic_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.BASIC_TOOL_CALLING()


@pytest.fixture(scope="module")
def multi_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.MULTI_TOOL_SELECTION()


@pytest.fixture(scope="module")
def hallucination_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.HALLUCINATION_RESISTANCE()


@pytest.fixture(scope="module")
def parameter_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.PARAMETER_EXTRACTION()


@pytest.fixture(scope="module")
def selection_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.TOOL_SELECTION()


@pytest.fixture(scope="module")
def error_scenarios():
    from agentguard.benchmark.scenarios import BuiltinScenarios
    return BuiltinScenarios.ERROR_HANDLING()


# ---------------------------------------------------------------------------
# Test 1: Total count >= 50
# ---------------------------------------------------------------------------


class TestScenarioCount:
    def test_total_count_at_least_50(self, all_scenarios):
        """Total scenario count must be >= 50 for credible published results."""
        count = len(all_scenarios)
        assert count >= 50, (
            f"Expected >= 50 scenarios, but got {count}. "
            "Add more scenarios across categories."
        )

    def test_basic_category_count(self, basic_scenarios):
        """Basic tool calling should have at least 10 scenarios."""
        assert len(basic_scenarios) >= 10, (
            f"Expected >= 10 basic scenarios, got {len(basic_scenarios)}"
        )

    def test_multi_tool_category_count(self, multi_scenarios):
        """Multi-tool scenarios should have at least 8."""
        assert len(multi_scenarios) >= 8, (
            f"Expected >= 8 multi-tool scenarios, got {len(multi_scenarios)}"
        )

    def test_hallucination_category_count(self, hallucination_scenarios):
        """Hallucination resistance should have at least 10 scenarios."""
        assert len(hallucination_scenarios) >= 10, (
            f"Expected >= 10 hallucination scenarios, got {len(hallucination_scenarios)}"
        )

    def test_parameter_extraction_category_count(self, parameter_scenarios):
        """Parameter extraction should have at least 10 scenarios."""
        assert len(parameter_scenarios) >= 10, (
            f"Expected >= 10 parameter extraction scenarios, got {len(parameter_scenarios)}"
        )

    def test_tool_selection_category_count(self, selection_scenarios):
        """Tool selection should have at least 8 scenarios."""
        assert len(selection_scenarios) >= 8, (
            f"Expected >= 8 tool selection scenarios, got {len(selection_scenarios)}"
        )

    def test_error_handling_category_count(self, error_scenarios):
        """Error handling should have at least 6 scenarios."""
        assert len(error_scenarios) >= 6, (
            f"Expected >= 6 error handling scenarios, got {len(error_scenarios)}"
        )


# ---------------------------------------------------------------------------
# Test 2: Every scenario has required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_all_have_name(self, all_scenarios):
        """Every scenario must have a non-empty name."""
        for s in all_scenarios:
            assert isinstance(s.name, str) and len(s.name) > 0, (
                f"Scenario is missing a name: {s}"
            )

    def test_all_have_category(self, all_scenarios):
        """Every scenario must have a non-empty category."""
        for s in all_scenarios:
            assert isinstance(s.category, str) and len(s.category) > 0, (
                f"Scenario '{s.name}' is missing a category"
            )

    def test_all_have_tools(self, all_scenarios):
        """Every scenario must have at least one tool."""
        for s in all_scenarios:
            assert isinstance(s.tools, list) and len(s.tools) > 0, (
                f"Scenario '{s.name}' has no tools"
            )

    def test_all_tools_have_function_schema(self, all_scenarios):
        """Every tool must follow the OpenAI function-calling schema."""
        for s in all_scenarios:
            for tool in s.tools:
                assert "type" in tool, (
                    f"Scenario '{s.name}': tool missing 'type' key"
                )
                assert "function" in tool, (
                    f"Scenario '{s.name}': tool missing 'function' key"
                )
                fn = tool["function"]
                assert "name" in fn, (
                    f"Scenario '{s.name}': tool function missing 'name'"
                )
                assert "parameters" in fn, (
                    f"Scenario '{s.name}': tool function missing 'parameters'"
                )

    def test_all_have_messages(self, all_scenarios):
        """Every scenario must have at least one message."""
        for s in all_scenarios:
            assert isinstance(s.messages, list) and len(s.messages) > 0, (
                f"Scenario '{s.name}' has no messages"
            )

    def test_all_messages_have_role_and_content(self, all_scenarios):
        """Every user/assistant/tool message must have role and content/tool_calls."""
        for s in all_scenarios:
            for msg in s.messages:
                assert "role" in msg, (
                    f"Scenario '{s.name}': message missing 'role'"
                )
                role = msg["role"]
                assert role in ("user", "assistant", "tool", "system"), (
                    f"Scenario '{s.name}': unknown role '{role}'"
                )
                # user and system messages must have content
                if role in ("user", "system"):
                    assert "content" in msg and msg["content"], (
                        f"Scenario '{s.name}': {role} message missing content"
                    )

    def test_all_have_expected_tool_calls(self, all_scenarios):
        """Every scenario must have an expected_tool_calls list (can be empty)."""
        for s in all_scenarios:
            assert isinstance(s.expected_tool_calls, list), (
                f"Scenario '{s.name}': expected_tool_calls must be a list"
            )


# ---------------------------------------------------------------------------
# Test 3: No duplicate names
# ---------------------------------------------------------------------------


class TestNoDuplicateNames:
    def test_no_duplicate_names(self, all_scenarios):
        """Scenario names must all be unique."""
        names = [s.name for s in all_scenarios]
        seen = set()
        duplicates = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert len(duplicates) == 0, (
            f"Duplicate scenario names found: {duplicates}"
        )


# ---------------------------------------------------------------------------
# Test 4: Hallucination scenarios — validate_fn checks for zero calls
# ---------------------------------------------------------------------------


class TestHallucinationScenarios:
    def test_all_expect_no_tool_calls(self, hallucination_scenarios):
        """All hallucination scenarios must expect exactly zero tool calls."""
        for s in hallucination_scenarios:
            assert s.expected_tool_calls == [], (
                f"Hallucination scenario '{s.name}' should expect [] but got "
                f"{s.expected_tool_calls}"
            )

    def test_all_have_validate_fn(self, hallucination_scenarios):
        """All hallucination scenarios must have a validate_fn."""
        for s in hallucination_scenarios:
            assert s.validate_fn is not None, (
                f"Hallucination scenario '{s.name}' is missing validate_fn"
            )

    def test_validate_fn_accepts_empty_calls(self, hallucination_scenarios):
        """validate_fn must return True for an empty call list."""
        for s in hallucination_scenarios:
            assert s.validate_fn([]), (
                f"Hallucination scenario '{s.name}' validate_fn rejected empty list"
            )

    def test_validate_fn_rejects_any_tool_call(self, hallucination_scenarios):
        """validate_fn must return False when any tool call is present."""
        fake_calls = [{"name": "some_tool", "arguments": {}}]
        for s in hallucination_scenarios:
            assert not s.validate_fn(fake_calls), (
                f"Hallucination scenario '{s.name}' validate_fn accepted tool calls"
            )

    def test_no_ambiguous_scenario(self, hallucination_scenarios):
        """The previously-removed ambiguous scenario must not be present."""
        names = {s.name for s in hallucination_scenarios}
        assert "hallucination_irrelevant_query" not in names, (
            "Ambiguous 'hallucination_irrelevant_query' should not be in the suite"
        )

    def test_specific_hallucination_scenarios_exist(self, hallucination_scenarios):
        """Verify specific unambiguous hallucination scenarios are present."""
        names = {s.name for s in hallucination_scenarios}
        required = [
            "hallucination_arithmetic_wrong_tool",
            "hallucination_joke_wrong_tool",
            "hallucination_ww2_wrong_tool",
            "hallucination_author_wrong_tool",
            "hallucination_sky_color_wrong_tool",
            "hallucination_french_greeting_wrong_tool",
            "hallucination_capital_japan_wrong_tool",
            "hallucination_spider_legs_wrong_tool",
            "hallucination_water_formula_wrong_tool",
            "hallucination_quantum_computing_wrong_tool",
        ]
        for req in required:
            assert req in names, (
                f"Required hallucination scenario '{req}' not found. Present: {sorted(names)}"
            )

    def test_irrelevant_tools_in_hallucination_scenarios(self, hallucination_scenarios):
        """Each hallucination scenario should NOT have a tool that matches the question."""
        for s in hallucination_scenarios:
            tool_names = {t["function"]["name"] for t in s.tools}
            if "arithmetic" in s.name:
                assert "calculate" not in tool_names, (
                    f"'{s.name}' should not have calculate tool"
                )
            if "weather" in s.name and "hallucination_no_weather_tool" == s.name:
                assert "get_weather" not in tool_names, (
                    f"'{s.name}' should not have get_weather tool"
                )
            if "joke" in s.name:
                assert "get_weather" not in tool_names or "query_database" in tool_names


# ---------------------------------------------------------------------------
# Test 5: Multi-tool scenarios — must have validate_fn
# ---------------------------------------------------------------------------


class TestMultiToolScenarios:
    def test_all_have_validate_fn(self, multi_scenarios):
        """All multi-tool scenarios must have a validate_fn (order-insensitive check)."""
        for s in multi_scenarios:
            assert s.validate_fn is not None, (
                f"Multi-tool scenario '{s.name}' is missing validate_fn"
            )

    def test_all_have_multiple_expected_calls(self, multi_scenarios):
        """All multi-tool scenarios should expect at least 2 tool calls."""
        for s in multi_scenarios:
            assert len(s.expected_tool_calls) >= 2, (
                f"Multi-tool scenario '{s.name}' only expects "
                f"{len(s.expected_tool_calls)} call(s)"
            )

    def test_weather_and_calc_validator_correct(self, multi_scenarios):
        """multi_weather_and_calc validator accepts correct calls."""
        s = next(s for s in multi_scenarios if s.name == "multi_weather_and_calc")
        calls = [
            {"name": "get_weather", "arguments": {"city": "New York City"}},
            {"name": "calculate", "arguments": {"expression": "5*7"}},
        ]
        assert s.validate_fn(calls)

    def test_weather_and_calc_validator_rejects_partial(self, multi_scenarios):
        """multi_weather_and_calc validator rejects single-tool calls."""
        s = next(s for s in multi_scenarios if s.name == "multi_weather_and_calc")
        assert not s.validate_fn([{"name": "get_weather", "arguments": {"city": "NYC"}}])
        assert not s.validate_fn([{"name": "calculate", "arguments": {"expression": "5*7"}}])

    def test_two_cities_validator_correct(self, multi_scenarios):
        """multi_weather_two_cities validator accepts London + Paris."""
        s = next(s for s in multi_scenarios if s.name == "multi_weather_two_cities")
        calls = [
            {"name": "get_weather", "arguments": {"city": "London"}},
            {"name": "get_weather", "arguments": {"city": "Paris"}},
        ]
        assert s.validate_fn(calls)

    def test_two_cities_validator_rejects_one_city(self, multi_scenarios):
        """multi_weather_two_cities validator rejects only one city."""
        s = next(s for s in multi_scenarios if s.name == "multi_weather_two_cities")
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        assert not s.validate_fn(calls)

    def test_two_translations_validator_correct(self, multi_scenarios):
        """multi_translate_two_languages validator accepts es + fr."""
        s = next(s for s in multi_scenarios if s.name == "multi_translate_two_languages")
        calls = [
            {"name": "translate_text", "arguments": {"text": "hello", "target_language": "es"}},
            {"name": "translate_text", "arguments": {"text": "hello", "target_language": "fr"}},
        ]
        assert s.validate_fn(calls)

    def test_two_translations_validator_rejects_one_lang(self, multi_scenarios):
        """multi_translate_two_languages validator rejects only one language."""
        s = next(s for s in multi_scenarios if s.name == "multi_translate_two_languages")
        calls = [{"name": "translate_text", "arguments": {"text": "hello", "target_language": "es"}}]
        assert not s.validate_fn(calls)

    def test_two_stocks_validator_correct(self, multi_scenarios):
        """multi_stock_prices_two_tickers validator accepts AAPL + GOOGL."""
        s = next(s for s in multi_scenarios if s.name == "multi_stock_prices_two_tickers")
        calls = [
            {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
            {"name": "get_stock_price", "arguments": {"ticker": "GOOGL"}},
        ]
        assert s.validate_fn(calls)

    def test_search_and_calc_with_spaced_expression(self, multi_scenarios):
        """multi_search_and_calc validator accepts spaced expression '1000 / 8'."""
        s = next(s for s in multi_scenarios if s.name == "multi_search_and_calc")
        calls = [
            {"name": "search_web", "arguments": {"query": "Tokyo population"}},
            {"name": "calculate", "arguments": {"expression": "1000 / 8"}},
        ]
        assert s.validate_fn(calls)

    def test_specific_multi_scenarios_exist(self, multi_scenarios):
        """Verify all required multi-tool scenarios are present."""
        names = {s.name for s in multi_scenarios}
        required = [
            "multi_weather_and_calc",
            "multi_weather_two_cities",
            "multi_search_and_calc",
            "multi_translate_two_languages",
            "multi_stock_prices_two_tickers",
            "multi_weather_and_currency",
            "multi_read_file_and_search",
            "multi_calc_power_and_time",
        ]
        for req in required:
            assert req in names, (
                f"Required multi-tool scenario '{req}' not found"
            )


# ---------------------------------------------------------------------------
# Test 6: Basic scenarios
# ---------------------------------------------------------------------------


class TestBasicScenarios:
    def test_specific_basic_scenarios_exist(self, basic_scenarios):
        """Verify all required basic scenarios exist."""
        names = {s.name for s in basic_scenarios}
        required = [
            "basic_weather_london",
            "basic_calculate_divide",
            "basic_web_search_python",
            "basic_stock_price_aapl",
            "basic_translate_hello_spanish",
            "basic_read_file_etc_hosts",
            "basic_get_time_tokyo",
            "basic_convert_currency_usd_eur",
            "basic_weather_tokyo_fahrenheit",
        ]
        for req in required:
            assert req in names, f"Required basic scenario '{req}' not found"

    def test_math_scenarios_have_validate_fn(self, basic_scenarios):
        """All basic math scenarios must have validate_fn for flexible expression matching."""
        math_scenarios = [s for s in basic_scenarios if "math" in s.tags]
        for s in math_scenarios:
            assert s.validate_fn is not None, (
                f"Math scenario '{s.name}' must have validate_fn"
            )

    def test_divide_validator_accepts_spaced(self, basic_scenarios):
        """basic_calculate_divide validator accepts '144 / 12'."""
        s = next(s for s in basic_scenarios if s.name == "basic_calculate_divide")
        assert s.validate_fn is not None
        calls = [{"name": "calculate", "arguments": {"expression": "144 / 12"}}]
        assert s.validate_fn(calls)

    def test_divide_validator_accepts_exact(self, basic_scenarios):
        """basic_calculate_divide validator accepts exact '144/12'."""
        s = next(s for s in basic_scenarios if s.name == "basic_calculate_divide")
        calls = [{"name": "calculate", "arguments": {"expression": "144/12"}}]
        assert s.validate_fn(calls)

    def test_divide_validator_rejects_wrong_expression(self, basic_scenarios):
        """basic_calculate_divide validator rejects wrong expression."""
        s = next(s for s in basic_scenarios if s.name == "basic_calculate_divide")
        calls = [{"name": "calculate", "arguments": {"expression": "100/5"}}]
        assert not s.validate_fn(calls)

    def test_weather_london_correct_tool(self, basic_scenarios):
        """basic_weather_london expects get_weather with city=London."""
        s = next(s for s in basic_scenarios if s.name == "basic_weather_london")
        assert s.expected_tool_calls[0]["name"] == "get_weather"
        assert s.expected_tool_calls[0]["arguments"]["city"] == "London"

    def test_stock_aapl_correct_ticker(self, basic_scenarios):
        """basic_stock_price_aapl expects get_stock_price with ticker=AAPL."""
        s = next(s for s in basic_scenarios if s.name == "basic_stock_price_aapl")
        assert s.expected_tool_calls[0]["name"] == "get_stock_price"
        assert s.expected_tool_calls[0]["arguments"]["ticker"] == "AAPL"

    def test_translate_hello_spanish_params(self, basic_scenarios):
        """basic_translate_hello_spanish expects correct text and language code."""
        s = next(s for s in basic_scenarios if s.name == "basic_translate_hello_spanish")
        args = s.expected_tool_calls[0]["arguments"]
        assert args["text"].lower() == "hello"
        assert args["target_language"] == "es"

    def test_read_file_path(self, basic_scenarios):
        """basic_read_file_etc_hosts expects correct file path."""
        s = next(s for s in basic_scenarios if s.name == "basic_read_file_etc_hosts")
        assert s.expected_tool_calls[0]["arguments"]["path"] == "/etc/hosts"

    def test_get_time_tokyo_timezone(self, basic_scenarios):
        """basic_get_time_tokyo expects Asia/Tokyo timezone."""
        s = next(s for s in basic_scenarios if s.name == "basic_get_time_tokyo")
        assert "Asia/Tokyo" in s.expected_tool_calls[0]["arguments"]["timezone"]

    def test_currency_usd_eur_params(self, basic_scenarios):
        """basic_convert_currency_usd_eur expects correct amount and currency codes."""
        s = next(s for s in basic_scenarios if s.name == "basic_convert_currency_usd_eur")
        args = s.expected_tool_calls[0]["arguments"]
        assert args["amount"] == 100
        assert args["from_currency"].upper() == "USD"
        assert args["to_currency"].upper() == "EUR"

    def test_weather_tokyo_fahrenheit(self, basic_scenarios):
        """basic_weather_tokyo_fahrenheit expects units=fahrenheit."""
        s = next(s for s in basic_scenarios if s.name == "basic_weather_tokyo_fahrenheit")
        args = s.expected_tool_calls[0]["arguments"]
        assert "tokyo" in args["city"].lower()
        assert args["units"].lower() == "fahrenheit"


# ---------------------------------------------------------------------------
# Test 7: Parameter extraction scenarios
# ---------------------------------------------------------------------------


class TestParameterExtractionScenarios:
    def test_specific_parameter_scenarios_exist(self, parameter_scenarios):
        """Verify all required parameter extraction scenarios exist."""
        names = {s.name for s in parameter_scenarios}
        required = [
            "param_flight_search_full",
            "param_email_extraction",
            "param_calendar_sprint_review",
            "param_translate_japanese",
            "param_flight_lax_ord",
            "param_weather_sao_paulo_celsius",
            "param_currency_jpy_gbp",
            "param_db_query_users_age",
            "param_email_two_recipients",
            "param_calendar_doctor_appointment",
        ]
        for req in required:
            assert req in names, f"Required parameter scenario '{req}' not found"

    def test_all_parameter_scenarios_have_validate_fn(self, parameter_scenarios):
        """All parameter extraction scenarios should have a validate_fn."""
        for s in parameter_scenarios:
            assert s.validate_fn is not None, (
                f"Parameter scenario '{s.name}' should have validate_fn"
            )

    def test_flight_sfo_jfk_validator(self, parameter_scenarios):
        """param_flight_search_full validator checks all four params."""
        s = next(s for s in parameter_scenarios if s.name == "param_flight_search_full")
        correct = [
            {
                "name": "search_flights",
                "arguments": {
                    "from_airport": "SFO",
                    "to_airport": "JFK",
                    "date": "2025-08-15",
                    "max_price_usd": 300,
                },
            }
        ]
        assert s.validate_fn(correct)
        # Missing max_price should still fail
        wrong = [
            {
                "name": "search_flights",
                "arguments": {
                    "from_airport": "SFO",
                    "to_airport": "JFK",
                    "date": "2025-08-15",
                    "max_price_usd": 500,  # wrong price
                },
            }
        ]
        assert not s.validate_fn(wrong)

    def test_email_alice_validator(self, parameter_scenarios):
        """param_email_extraction validator checks recipient, subject."""
        s = next(s for s in parameter_scenarios if s.name == "param_email_extraction")
        correct = [
            {
                "name": "send_email",
                "arguments": {
                    "to": ["alice@example.com"],
                    "subject": "Meeting tomorrow",
                    "body": "Hi Alice, just a reminder about our 10am meeting tomorrow.",
                },
            }
        ]
        assert s.validate_fn(correct)

    def test_calendar_sprint_review_validator(self, parameter_scenarios):
        """param_calendar_sprint_review checks all four calendar fields."""
        s = next(s for s in parameter_scenarios if s.name == "param_calendar_sprint_review")
        correct = [
            {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Sprint Review",
                    "date": "2025-03-15",
                    "time": "14:00",
                    "duration_minutes": 90,
                },
            }
        ]
        assert s.validate_fn(correct)
        # Wrong duration should fail
        wrong = [
            {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Sprint Review",
                    "date": "2025-03-15",
                    "time": "14:00",
                    "duration_minutes": 60,
                },
            }
        ]
        assert not s.validate_fn(wrong)

    def test_translate_japanese_validator(self, parameter_scenarios):
        """param_translate_japanese validates text content and 'ja' language code."""
        s = next(s for s in parameter_scenarios if s.name == "param_translate_japanese")
        correct = [
            {
                "name": "translate_text",
                "arguments": {
                    "text": "Good morning, how are you?",
                    "target_language": "ja",
                },
            }
        ]
        assert s.validate_fn(correct)
        # Wrong language should fail
        wrong = [
            {
                "name": "translate_text",
                "arguments": {
                    "text": "Good morning, how are you?",
                    "target_language": "zh",
                },
            }
        ]
        assert not s.validate_fn(wrong)

    def test_currency_jpy_gbp_validator(self, parameter_scenarios):
        """param_currency_jpy_gbp validates amount=500, JPY→GBP."""
        s = next(s for s in parameter_scenarios if s.name == "param_currency_jpy_gbp")
        correct = [
            {
                "name": "convert_currency",
                "arguments": {
                    "amount": 500,
                    "from_currency": "JPY",
                    "to_currency": "GBP",
                },
            }
        ]
        assert s.validate_fn(correct)

    def test_email_two_recipients_validator(self, parameter_scenarios):
        """param_email_two_recipients validates both bob and carol are in to list."""
        s = next(s for s in parameter_scenarios if s.name == "param_email_two_recipients")
        correct = [
            {
                "name": "send_email",
                "arguments": {
                    "to": ["bob@test.com", "carol@test.com"],
                    "subject": "Q4 Results",
                    "body": "Please review attached.",
                },
            }
        ]
        assert s.validate_fn(correct)
        # Only one recipient should fail
        wrong = [
            {
                "name": "send_email",
                "arguments": {
                    "to": ["bob@test.com"],
                    "subject": "Q4 Results",
                    "body": "Please review attached.",
                },
            }
        ]
        assert not s.validate_fn(wrong)


# ---------------------------------------------------------------------------
# Test 8: Tool selection scenarios
# ---------------------------------------------------------------------------


class TestToolSelectionScenarios:
    def test_specific_selection_scenarios_exist(self, selection_scenarios):
        """Verify all required tool selection scenarios exist."""
        names = {s.name for s in selection_scenarios}
        required = [
            "selection_pick_weather_from_10",
            "selection_pick_calculator_from_10",
            "selection_pick_translate_from_10",
            "selection_pick_stock_from_10",
            "selection_pick_read_file_from_10",
            "selection_pick_flights_from_10",
            "selection_pick_get_time_from_10",
            "selection_pick_currency_from_10",
        ]
        for req in required:
            assert req in names, f"Required selection scenario '{req}' not found"

    def test_all_selection_scenarios_have_many_tools(self, selection_scenarios):
        """Each tool selection scenario should have >= 10 tools available."""
        for s in selection_scenarios:
            assert len(s.tools) >= 10, (
                f"Selection scenario '{s.name}' only has {len(s.tools)} tools; "
                "expected >= 10 for meaningful selection challenge"
            )

    def test_tesla_ticker_is_tsla(self, selection_scenarios):
        """selection_pick_stock_from_10 must use TSLA ticker."""
        s = next(s for s in selection_scenarios if s.name == "selection_pick_stock_from_10")
        assert s.expected_tool_calls[0]["arguments"]["ticker"].upper() == "TSLA"

    def test_calculator_scenario_has_validate_fn(self, selection_scenarios):
        """Calculator selection scenario must use validate_fn for expression flexibility."""
        s = next(s for s in selection_scenarios if s.name == "selection_pick_calculator_from_10")
        assert s.validate_fn is not None

    def test_calc_selector_accepts_spaced_expr(self, selection_scenarios):
        """Calculator selection validator accepts spaced expression."""
        s = next(s for s in selection_scenarios if s.name == "selection_pick_calculator_from_10")
        calls = [{"name": "calculate", "arguments": {"expression": "144 / 12"}}]
        assert s.validate_fn(calls)


# ---------------------------------------------------------------------------
# Test 9: Error handling scenarios
# ---------------------------------------------------------------------------


class TestErrorHandlingScenarios:
    def test_specific_error_scenarios_exist(self, error_scenarios):
        """Verify required error handling scenarios exist."""
        names = {s.name for s in error_scenarios}
        required = [
            "error_tool_returns_error",
            "error_missing_required_param",
            "error_tool_returns_empty",
            "error_tool_partial_data",
            "error_missing_email_body",
        ]
        for req in required:
            assert req in names, f"Required error scenario '{req}' not found"

    def test_error_scenarios_have_validate_fn(self, error_scenarios):
        """All error handling scenarios should have a validate_fn."""
        # The flight scenario may allow a tool call with just required fields
        for s in error_scenarios:
            if s.name == "error_ambiguous_no_flight_date":
                # This one allows the call — just verify it has validate_fn
                assert s.validate_fn is not None
            else:
                assert s.validate_fn is not None, (
                    f"Error scenario '{s.name}' missing validate_fn"
                )

    def test_error_tool_returns_error_expects_no_calls(self, error_scenarios):
        """error_tool_returns_error must expect no subsequent tool calls."""
        s = next(s for s in error_scenarios if s.name == "error_tool_returns_error")
        assert s.expected_tool_calls == []
        assert s.validate_fn([])
        assert not s.validate_fn([{"name": "get_weather", "arguments": {"city": "London"}}])

    def test_missing_param_expects_no_calls(self, error_scenarios):
        """error_missing_required_param must expect no tool calls."""
        s = next(s for s in error_scenarios if s.name == "error_missing_required_param")
        assert s.expected_tool_calls == []
        assert s.validate_fn([])
        assert not s.validate_fn([{"name": "get_weather", "arguments": {"city": "London"}}])

    def test_empty_results_expects_no_calls(self, error_scenarios):
        """error_tool_returns_empty must expect no further tool calls."""
        s = next(s for s in error_scenarios if s.name == "error_tool_returns_empty")
        assert s.validate_fn([])
        assert not s.validate_fn([{"name": "search_web", "arguments": {"query": "retry"}}])

    def test_partial_data_expects_no_calls(self, error_scenarios):
        """error_tool_partial_data must expect no further tool calls."""
        s = next(s for s in error_scenarios if s.name == "error_tool_partial_data")
        assert s.validate_fn([])
        assert not s.validate_fn([{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}])


# ---------------------------------------------------------------------------
# Test 10: Category membership
# ---------------------------------------------------------------------------


class TestCategoryMembership:
    EXPECTED_CATEGORIES = {
        "basic",
        "multi_tool",
        "hallucination_resistance",
        "parameter_extraction",
        "tool_selection",
        "error_handling",
    }

    def test_all_categories_present(self, all_scenarios):
        """All six required categories must be represented."""
        found = {s.category for s in all_scenarios}
        for cat in self.EXPECTED_CATEGORIES:
            assert cat in found, f"Category '{cat}' not found in any scenario"

    def test_no_unknown_categories(self, all_scenarios):
        """No scenario should have an unrecognised category."""
        found = {s.category for s in all_scenarios}
        unknown = found - self.EXPECTED_CATEGORIES
        assert len(unknown) == 0, f"Unknown categories: {unknown}"


# ---------------------------------------------------------------------------
# Test 11: New tool schemas present in the module
# ---------------------------------------------------------------------------


class TestNewToolSchemas:
    def test_get_time_tool_exists(self):
        """_GET_TIME_TOOL must be defined in the module."""
        from agentguard.benchmark.scenarios import _GET_TIME_TOOL
        assert _GET_TIME_TOOL["function"]["name"] == "get_time"
        assert "timezone" in _GET_TIME_TOOL["function"]["parameters"]["properties"]

    def test_convert_currency_tool_exists(self):
        """_CONVERT_CURRENCY_TOOL must be defined in the module."""
        from agentguard.benchmark.scenarios import _CONVERT_CURRENCY_TOOL
        assert _CONVERT_CURRENCY_TOOL["function"]["name"] == "convert_currency"
        props = _CONVERT_CURRENCY_TOOL["function"]["parameters"]["properties"]
        assert "amount" in props
        assert "from_currency" in props
        assert "to_currency" in props

    def test_get_directions_tool_exists(self):
        """_GET_DIRECTIONS_TOOL must be defined in the module."""
        from agentguard.benchmark.scenarios import _GET_DIRECTIONS_TOOL
        assert _GET_DIRECTIONS_TOOL["function"]["name"] == "get_directions"
        props = _GET_DIRECTIONS_TOOL["function"]["parameters"]["properties"]
        assert "origin" in props
        assert "destination" in props


# ---------------------------------------------------------------------------
# Test 12: Validate math expression helper
# ---------------------------------------------------------------------------


class TestValidateMathExpression:
    def _make_validator(self, expected: str):
        from agentguard.benchmark.scenarios import _validate_math_expression
        return _validate_math_expression(expected)

    def test_exact_match(self):
        v = self._make_validator("144/12")
        calls = [{"name": "calculate", "arguments": {"expression": "144/12"}}]
        assert v(calls)

    def test_spaced_match(self):
        v = self._make_validator("144/12")
        calls = [{"name": "calculate", "arguments": {"expression": "144 / 12"}}]
        assert v(calls)

    def test_numeric_equivalence(self):
        v = self._make_validator("2**10")
        calls = [{"name": "calculate", "arguments": {"expression": "2 ** 10"}}]
        assert v(calls)

    def test_wrong_expression_fails(self):
        v = self._make_validator("144/12")
        calls = [{"name": "calculate", "arguments": {"expression": "100/5"}}]
        assert not v(calls)

    def test_wrong_tool_name_fails(self):
        v = self._make_validator("2+2")
        calls = [{"name": "search_web", "arguments": {"query": "2+2"}}]
        assert not v(calls)

    def test_empty_calls_fails(self):
        v = self._make_validator("2+2")
        assert not v([])


# ---------------------------------------------------------------------------
# Test 13: _validate_no_tool_call helper
# ---------------------------------------------------------------------------


class TestValidateNoToolCall:
    def test_accepts_empty_list(self):
        from agentguard.benchmark.scenarios import _validate_no_tool_call
        assert _validate_no_tool_call([])

    def test_rejects_single_call(self):
        from agentguard.benchmark.scenarios import _validate_no_tool_call
        assert not _validate_no_tool_call([{"name": "anything", "arguments": {}}])

    def test_rejects_multiple_calls(self):
        from agentguard.benchmark.scenarios import _validate_no_tool_call
        calls = [
            {"name": "tool_a", "arguments": {}},
            {"name": "tool_b", "arguments": {}},
        ]
        assert not _validate_no_tool_call(calls)


# ---------------------------------------------------------------------------
# Test 14: _make_tool_names_validator helper
# ---------------------------------------------------------------------------


class TestMakeToolNamesValidator:
    def test_single_required_tool(self):
        from agentguard.benchmark.scenarios import _make_tool_names_validator
        v = _make_tool_names_validator("get_weather")
        assert v([{"name": "get_weather", "arguments": {}}])
        assert not v([{"name": "calculate", "arguments": {}}])

    def test_two_required_tools_both_present(self):
        from agentguard.benchmark.scenarios import _make_tool_names_validator
        v = _make_tool_names_validator("get_weather", "calculate")
        calls = [
            {"name": "get_weather", "arguments": {}},
            {"name": "calculate", "arguments": {}},
        ]
        assert v(calls)

    def test_two_required_tools_one_missing(self):
        from agentguard.benchmark.scenarios import _make_tool_names_validator
        v = _make_tool_names_validator("get_weather", "calculate")
        assert not v([{"name": "get_weather", "arguments": {}}])

    def test_order_insensitive(self):
        from agentguard.benchmark.scenarios import _make_tool_names_validator
        v = _make_tool_names_validator("tool_a", "tool_b")
        # Either order should pass
        assert v([
            {"name": "tool_b", "arguments": {}},
            {"name": "tool_a", "arguments": {}},
        ])
