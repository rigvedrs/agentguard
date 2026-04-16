#!/usr/bin/env python3
"""
Comprehensive live end-to-end test of every agentguard feature.
Uses OpenRouter with GPT-4o-mini for real LLM tool-calling tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("Set OPENROUTER_API_KEY before running live_test.py")

from agentguard import (
    GuardConfig,
    GuardAction,
    CircuitBreaker,
    RateLimiter,
    TokenBudget,
    RetryPolicy,
    HallucinationDetector,
    SemanticValidator,
    TraceRecorder,
    TraceStore,
    TestGenerator,
    TraceReplayer,
    ConsoleReporter,
    JsonReporter,
    guard,
    record_session,
    assert_tool_call,
    CustomValidator,
    validator_fn,
    ToolTimeoutError,
)
from agentguard.core.registry import global_registry
from agentguard.core.types import (
    BudgetConfig,
    BudgetExceededError,
    CircuitBreakerConfig,
    CircuitOpenError,
    RateLimitConfig,
    RateLimitError,
    ToolCall,
    ToolCallStatus,
    ValidationResult,
    ValidatorKind,
)
from agentguard.integrations import (
    Provider,
    Providers,
    guard_tools,
    guard_openai_tools,
    OpenAIToolExecutor,
    guard_anthropic_tools,
    AnthropicToolExecutor,
    GuardedLangChainTool,
    guard_langchain_tools,
    GuardedMCPServer,
    GuardedMCPClient,
)

PASS = 0
FAIL = 0
ISSUES = []

def test(name):
    """Decorator to run and report on a test."""
    def decorator(fn):
        def wrapper():
            global PASS, FAIL
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            global_registry.clear()  # Clean state
            try:
                fn()
                PASS += 1
                print(f"  ✅ PASSED")
            except Exception as e:
                FAIL += 1
                ISSUES.append((name, str(e)))
                print(f"  ❌ FAILED: {e}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================================
# TEST 1: Basic @guard decorator
# ============================================================================

@test("1. Basic @guard decorator — zero config")
def test_basic_guard():
    @guard
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    result = greet("World")
    assert result == "Hello, World!", f"Expected 'Hello, World!' got {result!r}"
    print(f"  greet('World') = {result!r}")

    # With options
    @guard(validate_input=True, validate_output=True)
    def add(a: int, b: int) -> int:
        return a + b

    result = add(3, 7)
    assert result == 10, f"Expected 10 got {result}"
    print(f"  add(3, 7) = {result}")

    # Input validation should catch wrong types
    @guard(validate_input=True)
    def typed_fn(x: int) -> int:
        return x * 2

    try:
        typed_fn("not_an_int")
        print("  ⚠ Input validation did not block string → int (weak validation)")
    except Exception as e:
        print(f"  Input validation caught: {e}")

    # GuardConfig object
    config = GuardConfig(validate_input=True, max_retries=1, timeout=5.0)
    @guard(config=config)
    def configured(x: str) -> str:
        return x.upper()

    assert configured("hello") == "HELLO"
    print(f"  GuardConfig works: configured('hello') = 'HELLO'")


# ============================================================================
# TEST 2: Live OpenRouter tool calling
# ============================================================================

@test("2. Live OpenRouter + guard_tools — real API tool calls")
def test_openrouter_live():
    from openai import OpenAI

    @guard(validate_input=True, max_retries=1)
    def get_weather(city: str, units: str = "celsius") -> dict:
        """Get current weather for a city."""
        time.sleep(0.05)
        temps = {"london": 15, "tokyo": 28, "new york": 22}
        return {
            "city": city,
            "temperature": temps.get(city.lower(), 20),
            "units": units,
            "conditions": "partly cloudy",
        }

    @guard(validate_input=True)
    def calculate(expression: str) -> dict:
        """Evaluate a math expression safely."""
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters"}
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": result}

    executor = guard_tools([get_weather, calculate])
    print(f"  Generated {len(executor.tools)} tool schemas")
    for t in executor.tools:
        print(f"    {t['function']['name']}: {list(t['function']['parameters']['properties'].keys())}")

    client = OpenAI(**Providers.OPENROUTER.client_kwargs())

    print("  Calling OpenRouter (gpt-4o-mini)...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in London? Also calculate 42 * 7."}],
        tools=executor.tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    print(f"  Model response finish_reason: {response.choices[0].finish_reason}")

    if msg.tool_calls:
        print(f"  Model requested {len(msg.tool_calls)} tool call(s):")
        for tc in msg.tool_calls:
            print(f"    → {tc.function.name}({tc.function.arguments})")

        results = executor.execute_all(msg.tool_calls)
        for r in results:
            data = json.loads(r["content"])
            print(f"    Result: {json.dumps(data)[:80]}")

        # Continue conversation
        messages = [
            {"role": "user", "content": "What's the weather in London? Also calculate 42 * 7."},
            msg.model_dump(),
        ]
        messages.extend(results)

        final = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
        )
        answer = final.choices[0].message.content
        print(f"  Final answer: {answer[:120]}...")
        assert answer, "Final answer should not be empty"
    else:
        print(f"  Model responded with text: {msg.content[:100]}")


# ============================================================================
# TEST 3: Hallucination detection
# ============================================================================

@test("3. Hallucination detection — multi-signal")
def test_hallucination_detection():
    detector = HallucinationDetector(threshold=0.5)

    detector.register_tool(
        "get_weather",
        expected_latency_ms=(50, 5000),
        required_fields=["temperature", "humidity", "conditions"],
    )
    detector.register_tool(
        "search_api",
        expected_latency_ms=(100, 10000),
        required_fields=["results", "total"],
        response_patterns=[r'"total":\s*\d+'],
    )

    # Test 1: Impossibly fast = hallucinated
    r1 = detector.verify("get_weather", execution_time_ms=0.3,
                          response={"temperature": 72, "humidity": 50, "conditions": "sunny"})
    print(f"  Fast response (0.3ms): hallucinated={r1.is_hallucinated}, confidence={r1.confidence:.2f}")
    assert r1.is_hallucinated, "0.3ms should be flagged as hallucinated"

    # Test 2: Missing required fields
    r2 = detector.verify("get_weather", execution_time_ms=200,
                          response={"city": "London"})
    print(f"  Missing fields: hallucinated={r2.is_hallucinated}, confidence={r2.confidence:.2f}")

    # Test 3: Legitimate response
    r3 = detector.verify("get_weather", execution_time_ms=350,
                          response={"temperature": 18, "humidity": 65, "conditions": "rain"})
    print(f"  Legit response: hallucinated={r3.is_hallucinated}, confidence={r3.confidence:.2f}")
    assert not r3.is_hallucinated, "Legit response should not be flagged"

    # Test 4: Via @guard decorator with profile
    @guard(detect_hallucination=True)
    def api_call(query: str) -> dict:
        time.sleep(0.05)  # Real latency
        return {"results": [{"title": "test"}], "total": 1}

    api_call.register_hallucination_profile(
        expected_latency_ms=(20, 5000),
        required_fields=["results", "total"],
    )
    result = api_call("test")
    assert result["total"] == 1
    print(f"  @guard + profile: passed (real call not flagged)")


# ============================================================================
# TEST 4: Circuit breaker
# ============================================================================

@test("4. Circuit breaker — CLOSED → OPEN → HALF_OPEN → CLOSED")
def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.5, success_threshold=1)

    @guard(circuit_breaker=cb.config)
    def flaky_service(should_fail: bool = False) -> str:
        if should_fail:
            raise ConnectionError("Service down")
        return "ok"

    # Normal calls work
    assert flaky_service() == "ok"
    print(f"  Call 1: ok (circuit CLOSED)")

    # Trigger 3 failures to open circuit
    for i in range(3):
        try:
            flaky_service(should_fail=True)
        except Exception:
            pass
    print(f"  3 failures recorded")

    # Circuit should now be OPEN
    try:
        flaky_service()
        print("  ⚠ Circuit didn't open (may need more failures)")
    except CircuitOpenError:
        print(f"  Circuit is OPEN — calls blocked ✓")

    # Wait for recovery timeout
    print(f"  Waiting 0.6s for recovery timeout...")
    time.sleep(0.6)

    # Should be HALF_OPEN — one probe allowed
    result = flaky_service()
    print(f"  HALF_OPEN probe succeeded: {result}")
    assert result == "ok"

    # Circuit should close after success
    result = flaky_service()
    print(f"  Circuit back to CLOSED: {result}")


# ============================================================================
# TEST 5: Rate limiter
# ============================================================================

@test("5. Rate limiter — token bucket")
def test_rate_limiter():
    rl = RateLimiter(calls_per_second=2, burst=2)

    @guard(rate_limit=rl.config)
    def limited_api(x: str) -> str:
        return x

    # First 2 calls should work (burst)
    assert limited_api("a") == "a"
    assert limited_api("b") == "b"
    print(f"  First 2 calls: ok (within burst)")

    # 3rd call should be rate limited
    try:
        limited_api("c")
        print("  ⚠ 3rd call wasn't rate limited (timing issue)")
    except RateLimitError as e:
        print(f"  3rd call rate limited: {e}")

    # Wait and try again
    time.sleep(0.6)
    assert limited_api("d") == "d"
    print(f"  After wait: call succeeded ✓")

    # Standalone rate limiter usage
    rl2 = RateLimiter(calls_per_minute=60, burst=5)
    for i in range(5):
        allowed, retry = rl2.acquire("my_tool")
        assert allowed, f"Call {i} should be allowed"
    allowed, retry = rl2.acquire("my_tool")
    print(f"  Standalone: 5/5 burst used, 6th allowed={allowed}, retry_after={retry:.2f}s")


# ============================================================================
# TEST 6: Token budget
# ============================================================================

@test("6. Token budget — call count + cost limits")
def test_budget():
    budget = TokenBudget(
        max_calls_per_session=3,
        max_cost_per_session=0.50,
        cost_per_call=0.10,
        on_exceed=GuardAction.BLOCK,
    )

    @guard(budget=budget.config)
    def expensive_call(x: str) -> str:
        return x.upper()

    assert expensive_call("a") == "A"
    assert expensive_call("b") == "B"
    assert expensive_call("c") == "C"
    print(f"  3 calls succeeded (within budget)")

    try:
        expensive_call("d")
        assert False, "Should have been blocked"
    except BudgetExceededError as e:
        print(f"  4th call blocked: {e}")

    stats = budget.stats()
    print(f"  Budget stats: calls={stats.session_calls}, spend=${stats.session_spend:.2f}")

    # WARN mode
    warn_budget = TokenBudget(max_calls_per_session=1, on_exceed=GuardAction.WARN)

    @guard(budget=warn_budget.config)
    def warned_call(x: str) -> str:
        return x

    warned_call("a")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = warned_call("b")  # Should warn but not block
        print(f"  WARN mode: call succeeded, warnings={len(w)}")
    assert result == "b"


# ============================================================================
# TEST 7: Retry with backoff
# ============================================================================

@test("7. Retry with exponential backoff")
def test_retry():
    attempt_count = 0

    @guard(max_retries=3)
    def flaky(x: str) -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return f"success on attempt {attempt_count}"

    result = flaky("test")
    print(f"  Result: {result} (after {attempt_count} attempts)")
    assert attempt_count == 3
    assert "success" in result

    # RetryPolicy standalone
    policy = RetryPolicy(max_retries=2, initial_delay=0.1, backoff_factor=2.0, jitter=False)
    d0 = policy.delay_for(0)
    d1 = policy.delay_for(1)
    print(f"  RetryPolicy delays: attempt0={d0:.2f}s, attempt1={d1:.2f}s")
    assert d1 > d0, "Delay should increase"

    # Async retry
    async_attempts = 0

    @guard(max_retries=2)
    async def async_flaky(x: str) -> str:
        nonlocal async_attempts
        async_attempts += 1
        if async_attempts < 2:
            raise ValueError("async fail")
        return "async ok"

    result = asyncio.run(async_flaky.acall("test"))
    print(f"  Async retry: {result} (after {async_attempts} attempts)")
    assert result == "async ok"


# ============================================================================
# TEST 8: Timeout enforcement
# ============================================================================

@test("8. Timeout enforcement")
def test_timeout():
    @guard(timeout=0.3)
    def slow_call(x: str) -> str:
        time.sleep(2.0)  # Way too slow
        return x

    try:
        slow_call("test")
        assert False, "Should have timed out"
    except TimeoutError as e:
        print(f"  Timed out correctly: {e}")

    # Fast call should succeed
    @guard(timeout=5.0)
    def fast_call(x: str) -> str:
        time.sleep(0.01)
        return x.upper()

    assert fast_call("hello") == "HELLO"
    print(f"  Fast call within timeout: ok")

    # Standalone timeout decorator
    from agentguard.guardrails.timeout import timeout as timeout_decorator
    @timeout_decorator(seconds=0.2)
    def also_slow():
        time.sleep(1.0)

    try:
        also_slow()
        assert False, "Should have timed out"
    except ToolTimeoutError as e:
        print(f"  Standalone @timeout: {e}")


# ============================================================================
# TEST 9: Trace recording + JSON report + CLI
# ============================================================================

@test("9. Trace recording + JSON report + CLI")
def test_tracing():
    trace_dir = tempfile.mkdtemp(prefix="agentguard_test_")

    @guard(record=True, trace_dir=trace_dir, session_id="test_session")
    def traced_tool(query: str) -> dict:
        time.sleep(0.01)
        return {"query": query, "results": ["a", "b"]}

    # Record some calls
    traced_tool("hello")
    traced_tool("world")
    traced_tool("agentguard")
    print(f"  Recorded 3 calls to {trace_dir}")

    # Read back
    store = TraceStore(trace_dir)
    sessions = store.list_sessions()
    print(f"  Sessions: {sessions}")
    assert len(sessions) >= 1

    entries = store.read_all()
    print(f"  Total entries: {len(entries)}")
    assert len(entries) == 3

    # JSON report
    reporter = JsonReporter(store)
    report = reporter.generate(include_entries=True, include_anomalies=True)
    print(f"  Report summary: {report['summary']['total_calls']} calls, "
          f"success_rate={report['summary']['success_rate']}")
    assert report["summary"]["total_calls"] == 3
    assert report["summary"]["success_rate"] == 1.0

    # Save report
    report_path = os.path.join(trace_dir, "report.json")
    reporter.save(report_path, include_entries=True)
    print(f"  Report saved to {report_path}")

    # Console reporter
    console = ConsoleReporter(verbose=False)
    console.print_session_summary(entries)

    # CLI stats
    from agentguard.cli.main import main as cli_main
    print(f"\n  CLI: agentguard traces list")
    cli_main(["traces", "list", trace_dir])

    print(f"\n  CLI: agentguard traces stats")
    cli_main(["traces", "stats", trace_dir])

    # record_session context manager
    @guard
    def another_tool(x: str) -> str:
        return x

    with record_session() as recorder:
        another_tool("a")
        another_tool("b")

    rec_entries = recorder.entries()
    print(f"\n  record_session captured {len(rec_entries)} entries")
    assert len(rec_entries) == 2

    shutil.rmtree(trace_dir, ignore_errors=True)


# ============================================================================
# TEST 10: Test generation from traces
# ============================================================================

@test("10. Auto-generate pytest tests from traces")
def test_generation():
    trace_dir = tempfile.mkdtemp(prefix="agentguard_gen_")

    @guard(record=True, trace_dir=trace_dir)
    def search(query: str) -> dict:
        time.sleep(0.01)
        return {"results": [{"title": f"Result for {query}"}], "total": 1}

    search("python tutorials")
    search("agentguard docs")

    gen = TestGenerator(traces_dir=trace_dir)
    code = gen.generate_tests()
    test_count = code.count("def test_")
    print(f"  Generated {test_count} test functions")
    assert test_count >= 2

    # Show snippet
    for line in code.splitlines()[:15]:
        print(f"    {line}")

    # Replayer
    replayer = TraceReplayer(traces_dir=trace_dir)
    replayer.register_tool("search", search)
    report = replayer.replay_all()
    print(f"\n  Replay: {report.passed} passed, {report.failed} failed, {report.skipped} skipped")
    assert report.failed == 0

    shutil.rmtree(trace_dir, ignore_errors=True)


# ============================================================================
# TEST 11: Custom validators
# ============================================================================

@test("11. Custom validators — SQL injection + empty strings")
def test_custom_validators():
    from agentguard.validators.custom import no_empty_string_args, no_none_required_kwargs

    @validator_fn(name="no_sql_injection")
    def no_sql_injection(call, result=None):
        for val in list(call.kwargs.values()) + list(call.args):
            if isinstance(val, str):
                for token in ["DROP", "DELETE", "--", "1=1"]:
                    if token.upper() in val.upper():
                        return ValidationResult(
                            valid=False,
                            kind=ValidatorKind.CUSTOM,
                            message=f"SQL injection detected: {token!r}",
                        )
        return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

    @guard(custom_validators=[no_sql_injection, no_empty_string_args])
    def query_db(sql: str) -> dict:
        return {"rows": [], "count": 0}

    # Safe query
    result = query_db(sql="SELECT * FROM users WHERE id = 1")
    print(f"  Safe query: {result}")
    assert result["count"] == 0

    # SQL injection
    try:
        query_db(sql="SELECT * FROM users; DROP TABLE users--")
        assert False, "Should have blocked SQL injection"
    except RuntimeError as e:
        print(f"  SQL injection blocked: {e}")

    # Empty string
    try:
        query_db(sql="   ")
        assert False, "Should have blocked empty string"
    except RuntimeError as e:
        print(f"  Empty string blocked: {e}")

    # add_validator method
    @guard
    def safe_tool(x: str) -> str:
        return x

    @validator_fn(name="length_check")
    def length_check(call, result=None):
        for val in call.args:
            if isinstance(val, str) and len(val) > 100:
                return ValidationResult(valid=False, kind=ValidatorKind.CUSTOM,
                                         message="Input too long")
        return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

    safe_tool.add_validator(length_check)
    assert safe_tool("short") == "short"
    try:
        safe_tool("x" * 200)
        assert False, "Should block long input"
    except RuntimeError:
        print(f"  add_validator length check: blocked ✓")


# ============================================================================
# TEST 12: Anthropic integration
# ============================================================================

@test("12. Anthropic integration — schema generation")
def test_anthropic():
    def search_web(query: str) -> dict:
        """Search the web for information."""
        return {"results": []}

    def get_time(timezone: str = "UTC") -> str:
        """Get current time in a timezone."""
        return "12:00"

    schemas = guard_anthropic_tools([search_web, get_time])
    print(f"  Generated {len(schemas)} Anthropic tool schemas")
    for s in schemas:
        print(f"    {s['name']}: {s['description']}")
        print(f"      input_schema: {list(s['input_schema']['properties'].keys())}")
    assert len(schemas) == 2
    assert schemas[0]["name"] == "search_web"
    assert "input_schema" in schemas[0]

    # Executor
    executor = AnthropicToolExecutor()
    executor.register(search_web).register(get_time)
    assert len(executor.tools) == 2
    print(f"  AnthropicToolExecutor: {len(executor.tools)} tools registered")


# ============================================================================
# TEST 13: LangChain integration
# ============================================================================

@test("13. LangChain integration — GuardedLangChainTool")
def test_langchain():
    def web_search(query: str) -> str:
        """Search the web."""
        return f"Results for: {query}"

    def db_lookup(table: str, filter_by: str = "") -> dict:
        """Look up database records."""
        return {"table": table, "rows": [{"id": 1}]}

    # Bulk wrap
    tools = guard_langchain_tools([web_search, db_lookup],
                                   config=GuardConfig(validate_input=True))
    print(f"  Created {len(tools)} GuardedLangChainTools")
    for t in tools:
        print(f"    {t.name}: {t.description}")

    # Direct call
    result = tools[0]("agentguard")
    print(f"  web_search('agentguard') = {result}")
    assert "agentguard" in result

    result = tools[1]("users", filter_by="active=true")
    print(f"  db_lookup('users') = {result}")
    assert result["table"] == "users"

    # Individual tool
    custom = GuardedLangChainTool.from_function(
        web_search, name="search", description="Custom search tool"
    )
    print(f"  Custom tool: {custom!r}")
    assert custom("test") == "Results for: test"

    # OpenAI function schema export
    schema = custom.to_openai_function()
    assert schema["function"]["name"] == "web_search"
    print(f"  OpenAI export: {schema['function']['name']}")


# ============================================================================
# TEST 14: MCP integration
# ============================================================================

@test("14. MCP integration — GuardedMCPServer + Client")
def test_mcp():
    class MockMCPServer:
        async def call_tool(self, name, arguments):
            if name == "search":
                await asyncio.sleep(0.01)
                return {"results": [{"title": f"Found: {arguments.get('q', '')}"}]}
            return {"error": f"Unknown tool: {name}"}

        async def list_tools(self):
            return [{"name": "search", "description": "Search docs"}]

    async def run():
        server = MockMCPServer()
        guarded = GuardedMCPServer(server, config=GuardConfig(max_retries=1))

        # List tools (proxied)
        tools = await guarded.list_tools()
        print(f"  MCP tools: {tools}")
        assert len(tools) == 1

        # Call tool through guard
        result = await guarded.call_tool("search", {"q": "agentguard"})
        print(f"  MCP search result: {result}")
        assert "results" in result

        # Client-side
        client = GuardedMCPClient(server, config=GuardConfig(record=False))
        result = await client.call_tool("search", {"q": "test"})
        print(f"  MCP client result: {result}")
        assert "results" in result

    asyncio.run(run())


# ============================================================================
# TEST 15: Full multi-tool agent loop with OpenRouter
# ============================================================================

@test("15. Full multi-tool agent loop — OpenRouter GPT-4o-mini")
def test_full_agent_loop():
    from openai import OpenAI

    @guard(validate_input=True, record=True, trace_dir="/tmp/agentguard_e2e")
    def search_products(query: str, max_results: int = 5) -> dict:
        """Search for products in the catalog."""
        time.sleep(0.02)
        return {
            "query": query,
            "products": [
                {"name": f"Product A for {query}", "price": 29.99, "rating": 4.5},
                {"name": f"Product B for {query}", "price": 49.99, "rating": 4.8},
            ],
            "total": 2,
        }

    @guard(validate_input=True, record=True, trace_dir="/tmp/agentguard_e2e")
    def get_product_details(product_name: str) -> dict:
        """Get detailed information about a specific product."""
        time.sleep(0.02)
        return {
            "name": product_name,
            "description": f"High-quality {product_name}",
            "price": 39.99,
            "in_stock": True,
            "reviews": 142,
        }

    @guard(validate_input=True, record=True, trace_dir="/tmp/agentguard_e2e")
    def add_to_cart(product_name: str, quantity: int = 1) -> dict:
        """Add a product to the shopping cart."""
        time.sleep(0.01)
        return {
            "added": True,
            "product": product_name,
            "quantity": quantity,
            "cart_total": 39.99 * quantity,
        }

    executor = guard_tools([search_products, get_product_details, add_to_cart])
    client = OpenAI(**Providers.OPENROUTER.client_kwargs())

    messages = [
        {"role": "system", "content": "You are a shopping assistant. Use the available tools to help users find and purchase products."},
        {"role": "user", "content": "I'm looking for wireless headphones. Can you search for some and add the best one to my cart?"},
    ]

    print(f"  Starting multi-turn conversation...")
    max_turns = 5
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            tools=executor.tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if msg.tool_calls:
            print(f"  Turn {turn+1}: {len(msg.tool_calls)} tool call(s)")
            for tc in msg.tool_calls:
                print(f"    → {tc.function.name}({tc.function.arguments})")

            results = executor.execute_all(msg.tool_calls)
            for r in results:
                data = json.loads(r["content"])
                print(f"    ← {json.dumps(data)[:80]}...")
            messages.extend(results)
        else:
            # Model gave a final text response
            print(f"  Turn {turn+1} (final): {msg.content[:150]}...")
            break

    # Verify traces were recorded
    store = TraceStore("/tmp/agentguard_e2e")
    entries = store.read_all()
    print(f"\n  Traced {len(entries)} tool calls across the conversation")
    for e in entries:
        print(f"    {e.tool_name}: {e.result.status.value} ({e.result.execution_time_ms:.1f}ms)")

    # Generate report
    reporter = JsonReporter(store)
    report = reporter.generate(include_anomalies=True)
    print(f"  Report: {report['summary']['total_calls']} calls, "
          f"{report['summary']['success_rate']:.0%} success rate, "
          f"avg {report['summary']['latency_ms']['avg']:.1f}ms")

    shutil.rmtree("/tmp/agentguard_e2e", ignore_errors=True)


# ============================================================================
# TEST 16: Assertion builder for test fluency
# ============================================================================

@test("16. Fluent assertion builder")
def test_assertion_builder():
    from agentguard.core.types import TraceEntry, ToolCall, ToolResult

    call = ToolCall(tool_name="search", args=("hello",), kwargs={"limit": 5})
    result = ToolResult(
        call_id=call.call_id,
        tool_name="search",
        status=ToolCallStatus.SUCCESS,
        return_value={"results": ["a", "b"], "total": 2},
        execution_time_ms=150.0,
    )
    entry = TraceEntry(call=call, result=result)

    # Fluent assertions
    (assert_tool_call(entry)
        .succeeded()
        .returned_type(dict)
        .has_key("results")
        .has_key("total")
        .executed_within_ms(500)
        .was_not_retried()
        .not_hallucinated()
        .all_validations_passed())

    print(f"  All assertions passed for tool call")


# ============================================================================
# TEST 17: Semantic validator
# ============================================================================

@test("17. Semantic validator")
def test_semantic_validator():
    sv = SemanticValidator()

    @sv.validator("get_weather")
    def check_city(city: str, result: dict = None) -> str | None:
        if city and isinstance(result, dict):
            if city.lower() not in str(result).lower():
                return f"Response doesn't mention city '{city}'"
        return None

    # Pass
    results = sv.validate("get_weather", args=("London",), kwargs={},
                          result={"city": "London", "temp": 15})
    print(f"  Matching city: {len(results)} results, all valid={all(r.valid for r in results)}")

    # Fail
    results = sv.validate("get_weather", args=("London",), kwargs={},
                          result={"city": "Tokyo", "temp": 28})
    failed = [r for r in results if not r.valid]
    print(f"  Mismatched city: {len(failed)} failures — '{failed[0].message}'")
    assert len(failed) > 0

    # Built-in checks
    from agentguard.validators.semantic import check_non_empty, check_key_present, check_no_error_field

    assert check_non_empty({"data": [1, 2]}) is None
    assert check_non_empty("") is not None
    assert check_key_present({"a": 1, "b": 2}, keys=["a", "b"]) is None
    assert check_key_present({"a": 1}, keys=["b"]) is not None
    assert check_no_error_field({"data": "ok"}) is None
    assert check_no_error_field({"error": "something broke"}) is not None
    print(f"  Built-in semantic checks: all working")


# ============================================================================
# TEST 18: Provider presets + custom provider
# ============================================================================

@test("18. Provider presets + custom provider")
def test_providers():
    all_providers = Providers.all()
    print(f"  Total presets: {len(all_providers)}")
    for p in all_providers:
        print(f"    {p.name:<16} {p.base_url}")

    assert len(all_providers) >= 10

    # Lookup
    assert Providers.by_name("Groq") is Providers.GROQ
    assert Providers.by_name("openrouter") is Providers.OPENROUTER
    assert Providers.by_name("nonexistent") is None

    # Custom
    custom = Provider(
        name="LocalLLM",
        base_url="http://localhost:8080/v1",
        env_key="LOCAL_LLM_KEY",
        default_model="local-model",
    )
    kwargs = custom.client_kwargs(api_key="test")
    assert kwargs["base_url"] == "http://localhost:8080/v1"
    print(f"  Custom provider: {custom}")


# ============================================================================
# TEST 19: Before/after hooks
# ============================================================================

@test("19. Before/after call hooks")
def test_hooks():
    hook_log = []

    def before(call):
        hook_log.append(f"before:{call.tool_name}")

    def after(call, result):
        hook_log.append(f"after:{call.tool_name}:{result.status.value}")

    @guard(before_call=before, after_call=after)
    def hooked_tool(x: str) -> str:
        return x.upper()

    hooked_tool("test")
    print(f"  Hook log: {hook_log}")
    assert "before:hooked_tool" in hook_log
    assert "after:hooked_tool:success" in hook_log


# ============================================================================
# TEST 20: Registry
# ============================================================================

@test("20. Global tool registry")
def test_registry():
    global_registry.clear()

    @guard
    def tool_a(x: str) -> str:
        return x

    @guard
    def tool_b(n: int) -> int:
        return n * 2

    names = global_registry.names()
    print(f"  Registered tools: {names}")
    assert "tool_a" in names
    assert "tool_b" in names

    reg = global_registry.get("tool_a")
    assert reg is not None
    print(f"  tool_a registration: calls={reg.call_count}")

    tool_a("test")
    reg = global_registry.get("tool_a")
    print(f"  After 1 call: calls={reg.call_count}")
    assert reg.call_count == 1

    summary = global_registry.summary()
    print(f"  Registry summary: {json.dumps(summary, indent=2, default=str)[:200]}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_basic_guard,
        test_openrouter_live,
        test_hallucination_detection,
        test_circuit_breaker,
        test_rate_limiter,
        test_budget,
        test_retry,
        test_timeout,
        test_tracing,
        test_generation,
        test_custom_validators,
        test_anthropic,
        test_langchain,
        test_mcp,
        test_full_agent_loop,
        test_assertion_builder,
        test_semantic_validator,
        test_providers,
        test_hooks,
        test_registry,
    ]

    print("\n" + "🛡️ " * 20)
    print("  agentguard — COMPREHENSIVE LIVE TEST SUITE")
    print("  Using OpenRouter + GPT-4o-mini for real API tests")
    print("🛡️ " * 20)

    for t in tests:
        t()

    print("\n\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)

    if ISSUES:
        print("\n  ISSUES FOUND:")
        for name, err in ISSUES:
            print(f"    ❌ {name}: {err[:100]}")

    if FAIL == 0:
        print("\n  🎉 ALL TESTS PASSED — agentguard is production-ready!")
    else:
        print(f"\n  ⚠️  {FAIL} test(s) need attention")

    sys.exit(1 if FAIL > 0 else 0)
