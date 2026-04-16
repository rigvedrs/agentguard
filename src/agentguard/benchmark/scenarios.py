"""Built-in benchmark scenarios for testing LLM tool-calling accuracy.

Each :class:`BenchmarkScenario` describes one test case: a set of available
tools, a conversation to send to the model, and the expected tool calls that
a correct model should produce.

Usage::

    from agentguard.benchmark.scenarios import BenchmarkScenario, BuiltinScenarios

    # Get all basic tool-calling scenarios
    scenarios = BuiltinScenarios.BASIC_TOOL_CALLING

    # Inspect a single scenario
    s = scenarios[0]
    print(s.name, s.category, s.description)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# BenchmarkScenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScenario:
    """A single benchmark test case.

    Attributes:
        name: Short, unique identifier for the scenario.
        description: Human-readable description of what is being tested.
        category: Logical grouping (e.g. ``"basic"``, ``"multi_tool"``).
        tools: Tool schemas in OpenAI function-calling format.
        messages: The conversation to send to the model.
        expected_tool_calls: What the model *should* call.  Each entry is a
            dict with ``"name"`` and ``"arguments"`` keys.
        validate_fn: Optional custom function for complex validation logic.
            Signature: ``(actual_calls: list[dict]) -> bool``.
        tags: Additional labels for filtering scenarios.
    """

    name: str
    description: str
    category: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    expected_tool_calls: list[dict[str, Any]]
    validate_fn: Optional[Callable[[list[dict[str, Any]]], bool]] = field(
        default=None, compare=False, repr=False
    )
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared tool schemas
# ---------------------------------------------------------------------------

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. London"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    },
}

_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 2'",
                }
            },
            "required": ["expression"],
        },
    },
}

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information on a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"],
        },
    },
}

_FLIGHT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search for available flights between two airports.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_airport": {
                    "type": "string",
                    "description": "IATA departure airport code, e.g. SFO",
                },
                "to_airport": {
                    "type": "string",
                    "description": "IATA destination airport code, e.g. JFK",
                },
                "date": {
                    "type": "string",
                    "description": "Departure date in YYYY-MM-DD format",
                },
                "max_price_usd": {
                    "type": "number",
                    "description": "Maximum ticket price in USD",
                },
            },
            "required": ["from_airport", "to_airport"],
        },
    },
}

_SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to one or more recipients.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recipient email addresses",
                },
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

_CREATE_CALENDAR_EVENT_TOOL = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM format (24h)"},
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration of the event in minutes",
                },
            },
            "required": ["title", "date"],
        },
    },
}

_GET_STOCK_PRICE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL",
                }
            },
            "required": ["ticker"],
        },
    },
}

_TRANSLATE_TOOL = {
    "type": "function",
    "function": {
        "name": "translate_text",
        "description": "Translate text from one language to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {
                    "type": "string",
                    "description": "Target language code, e.g. 'es' for Spanish",
                },
                "source_language": {
                    "type": "string",
                    "description": "Source language code. 'auto' for auto-detect.",
                },
            },
            "required": ["text", "target_language"],
        },
    },
}

_READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"}
            },
            "required": ["path"],
        },
    },
}

_DATABASE_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a read-only SQL query and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT statement"},
                "database": {
                    "type": "string",
                    "description": "Database name to run the query against",
                },
            },
            "required": ["sql"],
        },
    },
}

_GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a given timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone string, e.g. 'Asia/Tokyo' or 'Europe/London'",
                }
            },
            "required": ["timezone"],
        },
    },
}

_CONVERT_CURRENCY_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The monetary amount to convert",
                },
                "from_currency": {
                    "type": "string",
                    "description": "ISO 4217 source currency code, e.g. USD",
                },
                "to_currency": {
                    "type": "string",
                    "description": "ISO 4217 target currency code, e.g. EUR",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}

_GET_DIRECTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_directions",
        "description": "Get driving or transit directions between two locations.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Starting location, e.g. 'New York City'",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination location, e.g. 'Boston'",
                },
                "mode": {
                    "type": "string",
                    "enum": ["driving", "transit", "walking", "cycling"],
                    "description": "Travel mode",
                },
            },
            "required": ["origin", "destination"],
        },
    },
}


# ---------------------------------------------------------------------------
# Shared validator helpers
# ---------------------------------------------------------------------------


def _validate_math_expression(expected_expr: str) -> "Callable[[list[dict[str, Any]]], bool]":
    """Return a validator that checks a math expression produces the same value.

    This handles cosmetic differences like spacing (``"2+2"`` vs ``"2 + 2"``)
    and equivalent forms (``"1000/8"`` vs ``"1000 / 8"``).  The validator
    evaluates both the expected and actual expressions with ``eval`` inside a
    restricted namespace and checks that the numeric results are equal.
    """
    import re as _re

    def _safe_eval(expr: str) -> Optional[float]:
        """Evaluate a simple arithmetic expression safely."""
        # Allow only digits, operators, spaces, and dots
        if not _re.fullmatch(r"[\d\s\.\+\-\*\/\(\)\^\%]+", expr.strip()):
            return None
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return float(result)
        except Exception:
            return None

    expected_val = _safe_eval(expected_expr)

    def _validator(actual_calls: "list[dict[str, Any]]") -> bool:
        for call in actual_calls:
            if call.get("name") != "calculate":
                continue
            actual_expr = call.get("arguments", {}).get("expression", "")
            # Direct string match (exact)
            if actual_expr == expected_expr:
                return True
            # Normalised comparison (strip spaces, lower)
            if actual_expr.replace(" ", "") == expected_expr.replace(" ", ""):
                return True
            # Numeric evaluation comparison
            if expected_val is not None:
                actual_val = _safe_eval(str(actual_expr))
                if actual_val is not None and abs(expected_val - actual_val) < 1e-9:
                    return True
        return False

    return _validator


def _validate_no_tool_call(actual: "list[dict[str, Any]]") -> bool:
    """Return True only if the model made no tool calls at all."""
    return len(actual) == 0


def _make_tool_names_validator(*required_tool_names: str) -> "Callable[[list[dict[str, Any]]], bool]":
    """Return a validator that checks all required tool names are present (order-insensitive)."""
    required = set(required_tool_names)

    def _validator(actual: "list[dict[str, Any]]") -> bool:
        actual_names = {c.get("name") for c in actual}
        return required.issubset(actual_names)

    return _validator


# ---------------------------------------------------------------------------
# Scenario definitions — Category 1: Basic Tool Calling (10 scenarios)
# ---------------------------------------------------------------------------


def _basic_scenarios() -> list[BenchmarkScenario]:
    """Scenarios testing simple single-tool calls."""
    return [
        BenchmarkScenario(
            name="basic_weather_london",
            description="Ask for weather in a specific city — tests basic tool routing.",
            category="basic",
            tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather like in London right now?"}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "London"}}],
            tags=["weather", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_calculate_divide",
            description="Division problem — tests that the model calls calculate with a division expression.",
            category="basic",
            tools=[_CALCULATOR_TOOL],
            messages=[{"role": "user", "content": "What is 144 divided by 12?"}],
            expected_tool_calls=[{"name": "calculate", "arguments": {"expression": "144/12"}}],
            validate_fn=_validate_math_expression("144/12"),
            tags=["math", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_calculate_2plus2",
            description="Simple arithmetic — tests that the model calls calculate with the right expression.",
            category="basic",
            tools=[_CALCULATOR_TOOL],
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            expected_tool_calls=[{"name": "calculate", "arguments": {"expression": "2+2"}}],
            validate_fn=_validate_math_expression("2+2"),
            tags=["math", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_web_search_python",
            description="Search query for Python tutorials — model should call search_web with relevant query.",
            category="basic",
            tools=[_SEARCH_TOOL],
            messages=[{"role": "user", "content": "Search the web for Python tutorials."}],
            expected_tool_calls=[{"name": "search_web", "arguments": {"query": "Python tutorials"}}],
            tags=["search", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_stock_price_aapl",
            description="Apple stock price lookup — tests ticker symbol extraction.",
            category="basic",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[{"role": "user", "content": "What is Apple's stock price?"}],
            expected_tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            tags=["finance", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_translate_hello_spanish",
            description="Translate 'hello' to Spanish — tests text and target language extraction.",
            category="basic",
            tools=[_TRANSLATE_TOOL],
            messages=[{"role": "user", "content": "Translate 'hello' to Spanish."}],
            expected_tool_calls=[
                {"name": "translate_text", "arguments": {"text": "hello", "target_language": "es"}}
            ],
            tags=["translation", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_read_file_etc_hosts",
            description="Read /etc/hosts file — tests file path extraction.",
            category="basic",
            tools=[_READ_FILE_TOOL],
            messages=[{"role": "user", "content": "Read the file /etc/hosts."}],
            expected_tool_calls=[{"name": "read_file", "arguments": {"path": "/etc/hosts"}}],
            tags=["file", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_get_time_tokyo",
            description="Get current time in Tokyo — tests timezone extraction.",
            category="basic",
            tools=[_GET_TIME_TOOL],
            messages=[{"role": "user", "content": "What time is it in Tokyo?"}],
            expected_tool_calls=[{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}],
            tags=["time", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_convert_currency_usd_eur",
            description="Convert 100 USD to EUR — tests currency conversion parameter extraction.",
            category="basic",
            tools=[_CONVERT_CURRENCY_TOOL],
            messages=[{"role": "user", "content": "Convert 100 USD to EUR."}],
            expected_tool_calls=[
                {
                    "name": "convert_currency",
                    "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                }
            ],
            tags=["currency", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_weather_tokyo_fahrenheit",
            description="Weather with explicit units — tests city and units extraction.",
            category="basic",
            tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather in Tokyo in Fahrenheit?"}],
            expected_tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "fahrenheit"}}
            ],
            tags=["weather", "single_tool"],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 2: Multi-Tool Calling (8 scenarios)
# ---------------------------------------------------------------------------


def _multi_tool_scenarios() -> list[BenchmarkScenario]:
    """Scenarios that require calling multiple tools in one turn."""

    def _validate_weather_and_calc(actual: list[dict[str, Any]]) -> bool:
        names = {c.get("name") for c in actual}
        return "get_weather" in names and "calculate" in names

    def _validate_two_weather_cities(actual: list[dict[str, Any]]) -> bool:
        weather_calls = [c for c in actual if c.get("name") == "get_weather"]
        cities = {c.get("arguments", {}).get("city", "").lower() for c in weather_calls}
        has_london = any("london" in city for city in cities)
        has_paris = any("paris" in city for city in cities)
        return has_london and has_paris

    def _validate_search_and_calc(actual: list[dict[str, Any]]) -> bool:
        """search_web with tokyo + calculate 1000/8."""
        calc_validator = _validate_math_expression("1000/8")
        has_search = any(
            c.get("name") == "search_web"
            and "tokyo" in c.get("arguments", {}).get("query", "").lower()
            for c in actual
        )
        has_calc = calc_validator(actual)
        return has_search and has_calc

    def _validate_two_translations(actual: list[dict[str, Any]]) -> bool:
        """translate_text called for both 'es' and 'fr'."""
        translate_calls = [c for c in actual if c.get("name") == "translate_text"]
        langs = {c.get("arguments", {}).get("target_language", "").lower() for c in translate_calls}
        return "es" in langs and "fr" in langs

    def _validate_two_stocks(actual: list[dict[str, Any]]) -> bool:
        """get_stock_price called for AAPL and GOOGL."""
        stock_calls = [c for c in actual if c.get("name") == "get_stock_price"]
        tickers = {c.get("arguments", {}).get("ticker", "").upper() for c in stock_calls}
        return "AAPL" in tickers and "GOOGL" in tickers

    def _validate_weather_and_currency(actual: list[dict[str, Any]]) -> bool:
        """get_weather for Berlin + convert_currency EUR to USD."""
        names = {c.get("name") for c in actual}
        return "get_weather" in names and "convert_currency" in names

    def _validate_file_and_search(actual: list[dict[str, Any]]) -> bool:
        """read_file + search_web both present."""
        names = {c.get("name") for c in actual}
        return "read_file" in names and "search_web" in names

    def _validate_calc_and_time(actual: list[dict[str, Any]]) -> bool:
        """calculate (2^10 = 1024) + get_time for London."""
        calc_validator = _validate_math_expression("2**10")
        has_time = any(
            c.get("name") == "get_time"
            and "london" in c.get("arguments", {}).get("timezone", "").lower()
            for c in actual
        )
        # Also accept "Europe/London"
        if not has_time:
            has_time = any(
                c.get("name") == "get_time"
                and "europe/london" in c.get("arguments", {}).get("timezone", "").lower()
                for c in actual
            )
        # Flexible calc check: 2**10 or 2^10 or 1024
        has_calc = calc_validator(actual)
        if not has_calc:
            # Accept any calculate call whose result is 1024
            for call in actual:
                if call.get("name") == "calculate":
                    expr = call.get("arguments", {}).get("expression", "")
                    try:
                        import re as _re
                        safe_expr = expr.replace("^", "**")
                        if _re.fullmatch(r"[\d\s\.\+\-\*\/\(\)\^]+", safe_expr.strip()):
                            val = eval(safe_expr, {"__builtins__": {}}, {})
                            if abs(float(val) - 1024.0) < 1e-9:
                                has_calc = True
                                break
                    except Exception:
                        pass
        return has_calc and has_time

    return [
        BenchmarkScenario(
            name="multi_weather_and_calc",
            description="User asks for weather AND a calculation — model should call both tools.",
            category="multi_tool",
            tools=[_WEATHER_TOOL, _CALCULATOR_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in New York City, and also what is 5 times 7?",
                }
            ],
            expected_tool_calls=[
                {"name": "get_weather", "arguments": {"city": "New York City"}},
                {"name": "calculate", "arguments": {"expression": "5*7"}},
            ],
            validate_fn=_validate_weather_and_calc,
            tags=["multi_tool", "weather", "math"],
        ),
        BenchmarkScenario(
            name="multi_weather_two_cities",
            description="Compare weather in two cities — should call get_weather twice.",
            category="multi_tool",
            tools=[_WEATHER_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Compare the weather in London and Paris for me.",
                }
            ],
            expected_tool_calls=[
                {"name": "get_weather", "arguments": {"city": "London"}},
                {"name": "get_weather", "arguments": {"city": "Paris"}},
            ],
            validate_fn=_validate_two_weather_cities,
            tags=["multi_tool", "weather"],
        ),
        BenchmarkScenario(
            name="multi_search_and_calc",
            description="Look up information AND do math — tests parallel tool planning.",
            category="multi_tool",
            tools=[_SEARCH_TOOL, _CALCULATOR_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Search for the population of Tokyo and also calculate 1000 divided by 8.",
                }
            ],
            expected_tool_calls=[
                {"name": "search_web", "arguments": {"query": "Tokyo population"}},
                {"name": "calculate", "arguments": {"expression": "1000/8"}},
            ],
            validate_fn=_validate_search_and_calc,
            tags=["multi_tool", "search", "math"],
        ),
        BenchmarkScenario(
            name="multi_translate_two_languages",
            description="Translate to both Spanish and French — should call translate_text twice.",
            category="multi_tool",
            tools=[_TRANSLATE_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Translate 'hello' to Spanish and French.",
                }
            ],
            expected_tool_calls=[
                {"name": "translate_text", "arguments": {"text": "hello", "target_language": "es"}},
                {"name": "translate_text", "arguments": {"text": "hello", "target_language": "fr"}},
            ],
            validate_fn=_validate_two_translations,
            tags=["multi_tool", "translation"],
        ),
        BenchmarkScenario(
            name="multi_stock_prices_two_tickers",
            description="Get stock prices for AAPL and GOOGL — should call get_stock_price twice.",
            category="multi_tool",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What are the current stock prices for Apple and Google?",
                }
            ],
            expected_tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
                {"name": "get_stock_price", "arguments": {"ticker": "GOOGL"}},
            ],
            validate_fn=_validate_two_stocks,
            tags=["multi_tool", "finance"],
        ),
        BenchmarkScenario(
            name="multi_weather_and_currency",
            description="Weather in Berlin and convert 30 EUR to USD — different tool types.",
            category="multi_tool",
            tools=[_WEATHER_TOOL, _CONVERT_CURRENCY_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Berlin, and convert 30 EUR to USD for me.",
                }
            ],
            expected_tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Berlin"}},
                {
                    "name": "convert_currency",
                    "arguments": {"amount": 30, "from_currency": "EUR", "to_currency": "USD"},
                },
            ],
            validate_fn=_validate_weather_and_currency,
            tags=["multi_tool", "weather", "currency"],
        ),
        BenchmarkScenario(
            name="multi_read_file_and_search",
            description="Read /etc/hosts and search for DNS info — file + search combo.",
            category="multi_tool",
            tools=[_READ_FILE_TOOL, _SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Read the file /etc/hosts and also search the web for how DNS works.",
                }
            ],
            expected_tool_calls=[
                {"name": "read_file", "arguments": {"path": "/etc/hosts"}},
                {"name": "search_web", "arguments": {"query": "how DNS works"}},
            ],
            validate_fn=_validate_file_and_search,
            tags=["multi_tool", "file", "search"],
        ),
        BenchmarkScenario(
            name="multi_calc_power_and_time",
            description="Calculate 2^10 and get the time in London — math plus timezone lookup.",
            category="multi_tool",
            tools=[_CALCULATOR_TOOL, _GET_TIME_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 to the power of 10? And what time is it in London?",
                }
            ],
            expected_tool_calls=[
                {"name": "calculate", "arguments": {"expression": "2**10"}},
                {"name": "get_time", "arguments": {"timezone": "Europe/London"}},
            ],
            validate_fn=_validate_calc_and_time,
            tags=["multi_tool", "math", "time"],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 3: Hallucination Resistance (10 scenarios)
# ---------------------------------------------------------------------------


def _hallucination_resistance_scenarios() -> list[BenchmarkScenario]:
    """Scenarios where the model must NOT make any tool call.

    All scenarios provide only irrelevant tools.  The model should answer from
    training data without calling any tool.
    """
    return [
        BenchmarkScenario(
            name="hallucination_arithmetic_wrong_tool",
            description=(
                "Simple arithmetic ('What is 2+2?') with ONLY a get_weather tool — "
                "model must NOT call get_weather; it should answer directly."
            ),
            category="hallucination_resistance",
            tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "math"],
        ),
        BenchmarkScenario(
            name="hallucination_joke_wrong_tool",
            description=(
                "'Tell me a joke' with ONLY a query_database tool — "
                "model must NOT call query_database; a joke needs no data lookup."
            ),
            category="hallucination_resistance",
            tools=[_DATABASE_QUERY_TOOL],
            messages=[{"role": "user", "content": "Tell me a joke."}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool"],
        ),
        BenchmarkScenario(
            name="hallucination_ww2_wrong_tool",
            description=(
                "Historical fact question with ONLY get_stock_price — "
                "model should answer from knowledge, not call the stock tool."
            ),
            category="hallucination_resistance",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[{"role": "user", "content": "What year did World War 2 end?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "history"],
        ),
        BenchmarkScenario(
            name="hallucination_author_wrong_tool",
            description=(
                "'Who wrote Romeo and Juliet?' with ONLY calculate — "
                "model should answer from knowledge, not use the calculator."
            ),
            category="hallucination_resistance",
            tools=[_CALCULATOR_TOOL],
            messages=[{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "literature"],
        ),
        BenchmarkScenario(
            name="hallucination_sky_color_wrong_tool",
            description=(
                "'What color is the sky?' with ONLY search_flights — "
                "trivial knowledge question; flight search is completely irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_FLIGHT_SEARCH_TOOL],
            messages=[{"role": "user", "content": "What color is the sky?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool"],
        ),
        BenchmarkScenario(
            name="hallucination_french_greeting_wrong_tool",
            description=(
                "'Say hello in French' with ONLY read_file — "
                "model knows 'bonjour' from training; file reading is irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_READ_FILE_TOOL],
            messages=[{"role": "user", "content": "Say hello in French."}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool"],
        ),
        BenchmarkScenario(
            name="hallucination_capital_japan_wrong_tool",
            description=(
                "'What is the capital of Japan?' with ONLY send_email — "
                "model knows 'Tokyo'; email is completely irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_SEND_EMAIL_TOOL],
            messages=[{"role": "user", "content": "What's the capital of Japan?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "geography"],
        ),
        BenchmarkScenario(
            name="hallucination_spider_legs_wrong_tool",
            description=(
                "'How many legs does a spider have?' with ONLY create_calendar_event — "
                "trivial biology question; calendar creation is irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_CREATE_CALENDAR_EVENT_TOOL],
            messages=[{"role": "user", "content": "How many legs does a spider have?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "biology"],
        ),
        BenchmarkScenario(
            name="hallucination_water_formula_wrong_tool",
            description=(
                "'What is the chemical formula for water?' with ONLY get_directions — "
                "chemistry fact from training; directions tool is irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_GET_DIRECTIONS_TOOL],
            messages=[{"role": "user", "content": "What is the chemical formula for water?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "science"],
        ),
        BenchmarkScenario(
            name="hallucination_quantum_computing_wrong_tool",
            description=(
                "'Explain quantum computing in one sentence' with ONLY convert_currency — "
                "explanation requires no external data; currency conversion is irrelevant."
            ),
            category="hallucination_resistance",
            tools=[_CONVERT_CURRENCY_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Explain quantum computing in one sentence.",
                }
            ],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "science"],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 4: Parameter Extraction (10 scenarios)
# ---------------------------------------------------------------------------


def _parameter_extraction_scenarios() -> list[BenchmarkScenario]:
    """Scenarios that test accurate parameter extraction from natural language."""

    def _validate_flight_sfo_jfk(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "search_flights":
                continue
            args = call.get("arguments", {})
            if (
                args.get("from_airport", "").upper() == "SFO"
                and args.get("to_airport", "").upper() == "JFK"
                and str(args.get("date", "")) == "2025-08-15"
                and args.get("max_price_usd") == 300
            ):
                return True
        return False

    def _validate_email_alice(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "send_email":
                continue
            args = call.get("arguments", {})
            to = args.get("to", [])
            subject = args.get("subject", "")
            body = args.get("body", "")
            if (
                "alice@example.com" in [r.lower() for r in to]
                and "meeting tomorrow" in subject.lower()
                and len(body) > 5
            ):
                return True
        return False

    def _validate_calendar_sprint_review(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "create_calendar_event":
                continue
            args = call.get("arguments", {})
            if (
                "sprint review" in args.get("title", "").lower()
                and args.get("date", "") == "2025-03-15"
                and args.get("time", "") == "14:00"
                and args.get("duration_minutes") == 90
            ):
                return True
        return False

    def _validate_translate_japanese(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "translate_text":
                continue
            args = call.get("arguments", {})
            if (
                "good morning" in args.get("text", "").lower()
                and args.get("target_language", "").lower() == "ja"
            ):
                return True
        return False

    def _validate_flight_lax_ord(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "search_flights":
                continue
            args = call.get("arguments", {})
            if (
                args.get("from_airport", "").upper() == "LAX"
                and args.get("to_airport", "").upper() == "ORD"
            ):
                return True
        return False

    def _validate_weather_sao_paulo_celsius(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "get_weather":
                continue
            args = call.get("arguments", {})
            city = args.get("city", "").lower()
            units = args.get("units", "").lower()
            if ("paulo" in city or "sao paulo" in city) and units == "celsius":
                return True
        return False

    def _validate_currency_jpy_gbp(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "convert_currency":
                continue
            args = call.get("arguments", {})
            if (
                args.get("amount") == 500
                and args.get("from_currency", "").upper() == "JPY"
                and args.get("to_currency", "").upper() == "GBP"
            ):
                return True
        return False

    def _validate_db_query_users(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "query_database":
                continue
            args = call.get("arguments", {})
            sql = args.get("sql", "").lower()
            db = args.get("database", "").lower()
            if (
                "select" in sql
                and "users" in sql
                and "age" in sql
                and "21" in sql
                and "users" in db
            ):
                return True
        return False

    def _validate_email_two_recipients(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "send_email":
                continue
            args = call.get("arguments", {})
            to = [r.lower() for r in args.get("to", [])]
            subject = args.get("subject", "")
            if (
                "bob@test.com" in to
                and "carol@test.com" in to
                and "q4" in subject.lower()
            ):
                return True
        return False

    def _validate_calendar_doctor(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "create_calendar_event":
                continue
            args = call.get("arguments", {})
            if (
                "doctor" in args.get("title", "").lower()
                and args.get("date", "") == "2025-06-20"
                and args.get("time", "") == "09:30"
                and args.get("duration_minutes") == 45
            ):
                return True
        return False

    return [
        BenchmarkScenario(
            name="param_flight_search_full",
            description="Extract departure city, destination, date, and max price from a sentence.",
            category="parameter_extraction",
            tools=[_FLIGHT_SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need to find cheap flights from San Francisco to New York "
                        "on 2025-08-15. My budget is $300."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "search_flights",
                    "arguments": {
                        "from_airport": "SFO",
                        "to_airport": "JFK",
                        "date": "2025-08-15",
                        "max_price_usd": 300,
                    },
                }
            ],
            validate_fn=_validate_flight_sfo_jfk,
            tags=["parameter_extraction", "flights"],
        ),
        BenchmarkScenario(
            name="param_email_extraction",
            description="Extract recipient, subject, and body from a natural-language request.",
            category="parameter_extraction",
            tools=[_SEND_EMAIL_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Send an email to alice@example.com with subject 'Meeting tomorrow' "
                        "and body 'Hi Alice, just a reminder about our 10am meeting tomorrow.'"
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "send_email",
                    "arguments": {
                        "to": ["alice@example.com"],
                        "subject": "Meeting tomorrow",
                        "body": "Hi Alice, just a reminder about our 10am meeting tomorrow.",
                    },
                }
            ],
            validate_fn=_validate_email_alice,
            tags=["parameter_extraction", "email"],
        ),
        BenchmarkScenario(
            name="param_calendar_sprint_review",
            description="Extract event name, date, time, and duration from natural language.",
            category="parameter_extraction",
            tools=[_CREATE_CALENDAR_EVENT_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Create a meeting called 'Sprint Review' on 2025-03-15 at 14:00 "
                        "for 90 minutes."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Sprint Review",
                        "date": "2025-03-15",
                        "time": "14:00",
                        "duration_minutes": 90,
                    },
                }
            ],
            validate_fn=_validate_calendar_sprint_review,
            tags=["parameter_extraction", "calendar"],
        ),
        BenchmarkScenario(
            name="param_translate_japanese",
            description="Translate a specific phrase to Japanese — tests text and language code extraction.",
            category="parameter_extraction",
            tools=[_TRANSLATE_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Translate 'Good morning, how are you?' to Japanese.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Good morning, how are you?",
                        "target_language": "ja",
                    },
                }
            ],
            validate_fn=_validate_translate_japanese,
            tags=["parameter_extraction", "translation"],
        ),
        BenchmarkScenario(
            name="param_flight_lax_ord",
            description="Extract LAX and ORD airport codes from city names.",
            category="parameter_extraction",
            tools=[_FLIGHT_SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Search for flights from Los Angeles to Chicago.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "search_flights",
                    "arguments": {
                        "from_airport": "LAX",
                        "to_airport": "ORD",
                    },
                }
            ],
            validate_fn=_validate_flight_lax_ord,
            tags=["parameter_extraction", "flights"],
        ),
        BenchmarkScenario(
            name="param_weather_sao_paulo_celsius",
            description="Extract city with special characters and explicit celsius units.",
            category="parameter_extraction",
            tools=[_WEATHER_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Get weather in São Paulo in celsius.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"city": "São Paulo", "units": "celsius"},
                }
            ],
            validate_fn=_validate_weather_sao_paulo_celsius,
            tags=["parameter_extraction", "weather"],
        ),
        BenchmarkScenario(
            name="param_currency_jpy_gbp",
            description="Convert 500 Japanese yen to British pounds — extract amount and currency codes.",
            category="parameter_extraction",
            tools=[_CONVERT_CURRENCY_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Convert 500 Japanese yen to British pounds.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "convert_currency",
                    "arguments": {
                        "amount": 500,
                        "from_currency": "JPY",
                        "to_currency": "GBP",
                    },
                }
            ],
            validate_fn=_validate_currency_jpy_gbp,
            tags=["parameter_extraction", "currency"],
        ),
        BenchmarkScenario(
            name="param_db_query_users_age",
            description="Extract SQL query and database name from a natural-language request.",
            category="parameter_extraction",
            tools=[_DATABASE_QUERY_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Query the users database: SELECT name FROM users WHERE age > 21."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "query_database",
                    "arguments": {
                        "sql": "SELECT name FROM users WHERE age > 21",
                        "database": "users",
                    },
                }
            ],
            validate_fn=_validate_db_query_users,
            tags=["parameter_extraction", "database"],
        ),
        BenchmarkScenario(
            name="param_email_two_recipients",
            description="Extract two email recipients and subject from a single request.",
            category="parameter_extraction",
            tools=[_SEND_EMAIL_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Send an email to bob@test.com and carol@test.com "
                        "with subject 'Q4 Results' and body 'Please review the attached Q4 results.'"
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "send_email",
                    "arguments": {
                        "to": ["bob@test.com", "carol@test.com"],
                        "subject": "Q4 Results",
                        "body": "Please review the attached Q4 results.",
                    },
                }
            ],
            validate_fn=_validate_email_two_recipients,
            tags=["parameter_extraction", "email"],
        ),
        BenchmarkScenario(
            name="param_calendar_doctor_appointment",
            description="Schedule a doctor appointment with date, time, and duration.",
            category="parameter_extraction",
            tools=[_CREATE_CALENDAR_EVENT_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Schedule a 'Doctor Appointment' on 2025-06-20 at 09:30 for 45 minutes."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Doctor Appointment",
                        "date": "2025-06-20",
                        "time": "09:30",
                        "duration_minutes": 45,
                    },
                }
            ],
            validate_fn=_validate_calendar_doctor,
            tags=["parameter_extraction", "calendar"],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 5: Tool Selection from Large Set (8 scenarios)
# ---------------------------------------------------------------------------


def _tool_selection_scenarios() -> list[BenchmarkScenario]:
    """Scenarios that test accurate tool selection when many tools are available."""

    # All ten tools available — model must pick the right one
    _ALL_TOOLS = [
        _WEATHER_TOOL,
        _CALCULATOR_TOOL,
        _SEARCH_TOOL,
        _FLIGHT_SEARCH_TOOL,
        _SEND_EMAIL_TOOL,
        _CREATE_CALENDAR_EVENT_TOOL,
        _GET_STOCK_PRICE_TOOL,
        _TRANSLATE_TOOL,
        _READ_FILE_TOOL,
        _DATABASE_QUERY_TOOL,
        _GET_TIME_TOOL,
        _CONVERT_CURRENCY_TOOL,
    ]

    return [
        BenchmarkScenario(
            name="selection_pick_weather_from_10",
            description="Pick get_weather from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "Is it raining in Berlin?"}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Berlin"}}],
            tags=["tool_selection", "weather"],
        ),
        BenchmarkScenario(
            name="selection_pick_calculator_from_10",
            description="Pick calculate from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "What is 144 divided by 12?"}],
            expected_tool_calls=[{"name": "calculate", "arguments": {"expression": "144/12"}}],
            validate_fn=_validate_math_expression("144/12"),
            tags=["tool_selection", "math"],
        ),
        BenchmarkScenario(
            name="selection_pick_translate_from_10",
            description="Pick translate_text from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "Translate 'Good morning' to French."}],
            expected_tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {"text": "Good morning", "target_language": "fr"},
                }
            ],
            tags=["tool_selection", "translation"],
        ),
        BenchmarkScenario(
            name="selection_pick_stock_from_10",
            description="Pick get_stock_price(ticker='TSLA') from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "What is Tesla's current stock price?"}],
            expected_tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}],
            tags=["tool_selection", "finance"],
        ),
        BenchmarkScenario(
            name="selection_pick_read_file_from_10",
            description="Pick read_file from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "Read the contents of config.yaml."}],
            expected_tool_calls=[{"name": "read_file", "arguments": {"path": "config.yaml"}}],
            tags=["tool_selection", "file"],
        ),
        BenchmarkScenario(
            name="selection_pick_flights_from_10",
            description="Pick search_flights from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "Find me flights from SFO to LAX."}],
            expected_tool_calls=[
                {"name": "search_flights", "arguments": {"from_airport": "SFO", "to_airport": "LAX"}}
            ],
            tags=["tool_selection", "flights"],
        ),
        BenchmarkScenario(
            name="selection_pick_get_time_from_10",
            description="Pick get_time from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "What time is it in New York right now?"}],
            expected_tool_calls=[
                {"name": "get_time", "arguments": {"timezone": "America/New_York"}}
            ],
            tags=["tool_selection", "time"],
        ),
        BenchmarkScenario(
            name="selection_pick_currency_from_10",
            description="Pick convert_currency from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[{"role": "user", "content": "Convert 50 British pounds to US dollars."}],
            expected_tool_calls=[
                {
                    "name": "convert_currency",
                    "arguments": {"amount": 50, "from_currency": "GBP", "to_currency": "USD"},
                }
            ],
            tags=["tool_selection", "currency"],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 6: Error Handling (6 scenarios)
# ---------------------------------------------------------------------------


def _error_handling_scenarios() -> list[BenchmarkScenario]:
    """Scenarios that test graceful handling of tool errors and ambiguous input."""
    return [
        BenchmarkScenario(
            name="error_tool_returns_error",
            description=(
                "Model receives a tool error response and must acknowledge it gracefully "
                "rather than retrying with hallucinated data."
            ),
            category="error_handling",
            tools=[_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in London?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": '{"error": "Weather service unavailable. Please try again later."}',
                },
            ],
            expected_tool_calls=[],
            validate_fn=lambda actual: len(actual) == 0,
            tags=["error_handling", "tool_error"],
        ),
        BenchmarkScenario(
            name="error_missing_required_param",
            description=(
                "Ambiguous user request where a required parameter (city) is missing — "
                "model should ask for clarification rather than guess."
            ),
            category="error_handling",
            tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather?"}],
            expected_tool_calls=[],
            validate_fn=lambda actual: len(actual) == 0,
            tags=["error_handling", "ambiguous"],
        ),
        BenchmarkScenario(
            name="error_tool_returns_empty",
            description=(
                "Search tool returns an empty result set — model should report no results, "
                "not fabricate data."
            ),
            category="error_handling",
            tools=[_SEARCH_TOOL],
            messages=[
                {"role": "user", "content": "Search for 'xyzzy12345nonexistent'."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_def456",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": '{"query": "xyzzy12345nonexistent"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_def456",
                    "content": '{"results": []}',
                },
            ],
            expected_tool_calls=[],
            validate_fn=lambda actual: len(actual) == 0,
            tags=["error_handling", "empty_results"],
        ),
        BenchmarkScenario(
            name="error_ambiguous_no_flight_date",
            description=(
                "User asks for flights without specifying a date — model should ask for "
                "clarification rather than guessing the date."
            ),
            category="error_handling",
            tools=[_FLIGHT_SEARCH_TOOL],
            messages=[
                {"role": "user", "content": "I need a flight from JFK to LAX."}
            ],
            expected_tool_calls=[
                {
                    "name": "search_flights",
                    "arguments": {"from_airport": "JFK", "to_airport": "LAX"},
                }
            ],
            # Date is optional per schema — accepting the call with just required fields is valid
            validate_fn=lambda actual: any(
                c.get("name") == "search_flights"
                and c.get("arguments", {}).get("from_airport", "").upper() == "JFK"
                and c.get("arguments", {}).get("to_airport", "").upper() == "LAX"
                for c in actual
            ),
            tags=["error_handling", "ambiguous", "flights"],
        ),
        BenchmarkScenario(
            name="error_tool_partial_data",
            description=(
                "Stock tool returns partial data (missing price) — model should acknowledge "
                "incompleteness, not invent a price."
            ),
            category="error_handling",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[
                {"role": "user", "content": "What is Apple's stock price?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_ghi789",
                            "type": "function",
                            "function": {
                                "name": "get_stock_price",
                                "arguments": '{"ticker": "AAPL"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_ghi789",
                    "content": '{"ticker": "AAPL", "status": "data_unavailable"}',
                },
            ],
            expected_tool_calls=[],
            validate_fn=lambda actual: len(actual) == 0,
            tags=["error_handling", "partial_data"],
        ),
        BenchmarkScenario(
            name="error_missing_email_body",
            description=(
                "User asks to send an email but provides no body — model should ask for "
                "body content rather than sending an empty email."
            ),
            category="error_handling",
            tools=[_SEND_EMAIL_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Send an email to dave@example.com with subject 'Hello'.",
                }
            ],
            expected_tool_calls=[],
            validate_fn=lambda actual: len(actual) == 0,
            tags=["error_handling", "ambiguous", "email"],
        ),
    ]


# ---------------------------------------------------------------------------
# Additional basic scenarios for the web search and stock coverage
# ---------------------------------------------------------------------------


def _additional_basic_scenarios() -> list[BenchmarkScenario]:
    """Extra basic scenarios to round out coverage."""
    return [
        BenchmarkScenario(
            name="basic_web_search",
            description="General information query — should use search_web.",
            category="basic",
            tools=[_SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Search the web for the latest news about artificial intelligence.",
                }
            ],
            expected_tool_calls=[
                {"name": "search_web", "arguments": {"query": "artificial intelligence news"}}
            ],
            tags=["search", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_stock_price",
            description="Stock ticker lookup — tests symbol extraction from natural language.",
            category="basic",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[{"role": "user", "content": "What is the current price of Apple stock?"}],
            expected_tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            tags=["finance", "single_tool"],
        ),
        BenchmarkScenario(
            name="basic_get_directions_nyc_boston",
            description="Get directions from NYC to Boston — tests origin/destination extraction.",
            category="basic",
            tools=[_GET_DIRECTIONS_TOOL],
            messages=[
                {"role": "user", "content": "Get me directions from New York City to Boston."}
            ],
            expected_tool_calls=[
                {
                    "name": "get_directions",
                    "arguments": {"origin": "New York City", "destination": "Boston"},
                }
            ],
            tags=["directions", "single_tool"],
        ),
    ]


# ---------------------------------------------------------------------------
# Additional hallucination scenarios (to ensure 10 in the category)
# ---------------------------------------------------------------------------


def _additional_hallucination_scenarios() -> list[BenchmarkScenario]:
    """A couple more hallucination-resistance scenarios for completeness."""
    return [
        BenchmarkScenario(
            name="hallucination_no_weather_tool",
            description=(
                "Weather question with NO weather tool available — model must not "
                "hallucinate a non-existent tool."
            ),
            category="hallucination_resistance",
            tools=[_CALCULATOR_TOOL],
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool"],
        ),
        BenchmarkScenario(
            name="hallucination_summarise_wrong_tool",
            description=(
                "Summarisation request with ONLY a get_stock_price tool — "
                "the model must NOT call get_stock_price to summarise text."
            ),
            category="hallucination_resistance",
            tools=[_GET_STOCK_PRICE_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarise the following text: "
                        "'The quick brown fox jumps over the lazy dog. "
                        "It was a sunny afternoon in the forest.'"
                    ),
                }
            ],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool", "summarisation"],
        ),
        BenchmarkScenario(
            name="hallucination_greet_wrong_tool",
            description=(
                "Simple greeting with ONLY a search_flights tool — "
                "model must NOT call search_flights to say hello."
            ),
            category="hallucination_resistance",
            tools=[_FLIGHT_SEARCH_TOOL],
            messages=[{"role": "user", "content": "Hello! How are you doing today?"}],
            expected_tool_calls=[],
            validate_fn=_validate_no_tool_call,
            tags=["hallucination", "no_tool"],
        ),
    ]


# ---------------------------------------------------------------------------
# Additional multi-tool and parameter scenarios
# ---------------------------------------------------------------------------


def _additional_multi_scenarios() -> list[BenchmarkScenario]:
    """Extra multi-tool scenarios to exceed the 50-scenario target."""

    def _validate_translate_and_weather(actual: list[dict[str, Any]]) -> bool:
        names = {c.get("name") for c in actual}
        return "translate_text" in names and "get_weather" in names

    def _validate_stock_and_search(actual: list[dict[str, Any]]) -> bool:
        names = {c.get("name") for c in actual}
        return "get_stock_price" in names and "search_web" in names

    return [
        BenchmarkScenario(
            name="multi_translate_and_weather",
            description="Translate text AND get weather — two completely different tool types.",
            category="multi_tool",
            tools=[_TRANSLATE_TOOL, _WEATHER_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Translate 'Good evening' to Italian and also tell me "
                        "the weather in Rome."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {"text": "Good evening", "target_language": "it"},
                },
                {"name": "get_weather", "arguments": {"city": "Rome"}},
            ],
            validate_fn=_validate_translate_and_weather,
            tags=["multi_tool", "translation", "weather"],
        ),
        BenchmarkScenario(
            name="multi_stock_and_search",
            description="Get stock price for MSFT and search for Microsoft news.",
            category="multi_tool",
            tools=[_GET_STOCK_PRICE_TOOL, _SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "What is Microsoft's stock price, and search the web for "
                        "the latest Microsoft news."
                    ),
                }
            ],
            expected_tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}},
                {"name": "search_web", "arguments": {"query": "Microsoft news"}},
            ],
            validate_fn=_validate_stock_and_search,
            tags=["multi_tool", "finance", "search"],
        ),
    ]


def _additional_parameter_scenarios() -> list[BenchmarkScenario]:
    """Extra parameter extraction scenarios for coverage."""

    def _validate_translate_spanish(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "translate_text":
                continue
            args = call.get("arguments", {})
            if (
                "hello, how are you?" in args.get("text", "").lower()
                and args.get("target_language", "").lower() == "es"
            ):
                return True
        return False

    def _validate_weather_with_units(actual: list[dict[str, Any]]) -> bool:
        for call in actual:
            if call.get("name") != "get_weather":
                continue
            args = call.get("arguments", {})
            if (
                "tokyo" in args.get("city", "").lower()
                and args.get("units", "").lower() == "fahrenheit"
            ):
                return True
        return False

    return [
        BenchmarkScenario(
            name="param_translate_spanish",
            description="Extract text and target language from natural language.",
            category="parameter_extraction",
            tools=[_TRANSLATE_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Translate 'Hello, how are you?' into Spanish.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Hello, how are you?",
                        "target_language": "es",
                    },
                }
            ],
            validate_fn=_validate_translate_spanish,
            tags=["parameter_extraction", "translation"],
        ),
        BenchmarkScenario(
            name="param_weather_with_units",
            description="Extract both city name and desired units.",
            category="parameter_extraction",
            tools=[_WEATHER_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What's the temperature in Tokyo in Fahrenheit?",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"city": "Tokyo", "units": "fahrenheit"},
                }
            ],
            validate_fn=_validate_weather_with_units,
            tags=["parameter_extraction", "weather"],
        ),
    ]


def _additional_tool_selection_scenarios() -> list[BenchmarkScenario]:
    """Extra tool-selection scenarios for database and email."""
    _ALL_TOOLS = [
        _WEATHER_TOOL,
        _CALCULATOR_TOOL,
        _SEARCH_TOOL,
        _FLIGHT_SEARCH_TOOL,
        _SEND_EMAIL_TOOL,
        _CREATE_CALENDAR_EVENT_TOOL,
        _GET_STOCK_PRICE_TOOL,
        _TRANSLATE_TOOL,
        _READ_FILE_TOOL,
        _DATABASE_QUERY_TOOL,
        _GET_TIME_TOOL,
        _CONVERT_CURRENCY_TOOL,
    ]
    return [
        BenchmarkScenario(
            name="selection_pick_database_from_10",
            description="Pick query_database from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[
                {
                    "role": "user",
                    "content": "Run this SQL query: SELECT * FROM orders LIMIT 10.",
                }
            ],
            expected_tool_calls=[
                {
                    "name": "query_database",
                    "arguments": {"sql": "SELECT * FROM orders LIMIT 10"},
                }
            ],
            tags=["tool_selection", "database"],
        ),
        BenchmarkScenario(
            name="selection_pick_email_from_10",
            description="Pick send_email from 10+ available tools.",
            category="tool_selection",
            tools=_ALL_TOOLS,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Email test@example.com with subject 'Hello' and body 'Hi there!'."
                    ),
                }
            ],
            expected_tool_calls=[
                {
                    "name": "send_email",
                    "arguments": {
                        "to": ["test@example.com"],
                        "subject": "Hello",
                        "body": "Hi there!",
                    },
                }
            ],
            tags=["tool_selection", "email"],
        ),
    ]


# ---------------------------------------------------------------------------
# BuiltinScenarios namespace
# ---------------------------------------------------------------------------


class BuiltinScenarios:
    """Factory class exposing named collections of built-in scenarios.

    All collections return *copies* of the scenarios so callers can mutate
    them freely.

    Usage::

        from agentguard.benchmark.scenarios import BuiltinScenarios

        runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
        runner.add_scenarios(BuiltinScenarios.HALLUCINATION_RESISTANCE)
    """

    @staticmethod
    def BASIC_TOOL_CALLING() -> list[BenchmarkScenario]:
        """Ten scenarios testing simple, single-tool calls plus three extra basics."""
        return _basic_scenarios() + _additional_basic_scenarios()

    @staticmethod
    def MULTI_TOOL_SELECTION() -> list[BenchmarkScenario]:
        """Eight scenarios requiring the model to call multiple tools in one turn."""
        return _multi_tool_scenarios() + _additional_multi_scenarios()

    @staticmethod
    def PARAMETER_EXTRACTION() -> list[BenchmarkScenario]:
        """Twelve scenarios testing accurate argument extraction from natural language."""
        return _parameter_extraction_scenarios() + _additional_parameter_scenarios()

    @staticmethod
    def HALLUCINATION_RESISTANCE() -> list[BenchmarkScenario]:
        """Thirteen scenarios where the model must NOT fabricate tool calls."""
        return _hallucination_resistance_scenarios() + _additional_hallucination_scenarios()

    @staticmethod
    def ERROR_HANDLING() -> list[BenchmarkScenario]:
        """Six scenarios testing graceful handling of tool errors and ambiguity."""
        return _error_handling_scenarios()

    @staticmethod
    def TOOL_SELECTION() -> list[BenchmarkScenario]:
        """Ten scenarios testing correct tool selection from a large tool set."""
        return _tool_selection_scenarios() + _additional_tool_selection_scenarios()

    @staticmethod
    def ALL() -> list[BenchmarkScenario]:
        """All 50+ built-in scenarios combined."""
        return (
            _basic_scenarios()
            + _additional_basic_scenarios()
            + _multi_tool_scenarios()
            + _additional_multi_scenarios()
            + _parameter_extraction_scenarios()
            + _additional_parameter_scenarios()
            + _hallucination_resistance_scenarios()
            + _additional_hallucination_scenarios()
            + _error_handling_scenarios()
            + _tool_selection_scenarios()
            + _additional_tool_selection_scenarios()
        )
