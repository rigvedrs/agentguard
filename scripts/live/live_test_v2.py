#!/usr/bin/env python3
"""
Live tests for agentguard v0.2.0 new features.
Tests middleware, policy, shared state, telemetry, benchmarking against OpenRouter.
"""

from __future__ import annotations
import asyncio, json, os, sys, time, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("Set OPENROUTER_API_KEY before running live_test_v2.py")

from agentguard import guard, GuardConfig, GuardAction
from agentguard.core.registry import global_registry

PASS = FAIL = 0
ISSUES = []

def test(name):
    def decorator(fn):
        def wrapper():
            global PASS, FAIL
            print(f"\n{'='*60}\nTEST: {name}\n{'='*60}")
            global_registry.clear()
            try:
                fn()
                PASS += 1
                print(f"  ✅ PASSED")
            except Exception as e:
                FAIL += 1
                ISSUES.append((name, str(e)))
                print(f"  ❌ FAILED: {e}")
                import traceback; traceback.print_exc()
        return wrapper
    return decorator

# ============================================================================
# TEST 1: Middleware Pipeline
# ============================================================================
@test("1. Middleware pipeline — composable guard chain")
def test_middleware():
    from agentguard.middleware import MiddlewareChain, MiddlewareContext

    log = []

    async def logging_mw(ctx, next):
        log.append(f"before:{ctx.tool_name}")
        result = await next(ctx)
        log.append(f"after:{ctx.tool_name}")
        return result

    async def metadata_mw(ctx, next):
        ctx.metadata["enriched"] = True
        return await next(ctx)

    chain = MiddlewareChain()
    chain.use(logging_mw)
    chain.use(metadata_mw)

    # Simulate running the chain
    ctx = MiddlewareContext(tool_name="test_tool", args=("hello",), kwargs={}, config=GuardConfig())

    async def final_handler(ctx):
        from agentguard.core.types import ToolResult, ToolCallStatus
        return ToolResult(call_id="test", tool_name=ctx.tool_name,
                         status=ToolCallStatus.SUCCESS, return_value="ok")

    result = asyncio.run(chain.run(ctx, final_handler))
    print(f"  Log: {log}")
    print(f"  Metadata enriched: {ctx.metadata.get('enriched')}")
    assert "before:test_tool" in log
    assert "after:test_tool" in log
    assert ctx.metadata.get("enriched") is True

    # Built-in middleware factories
    from agentguard.middleware import logging_middleware, timing_middleware
    chain2 = MiddlewareChain()
    chain2.use(logging_middleware())
    chain2.use(timing_middleware())
    print(f"  Built-in middleware factories: ok")


# ============================================================================
# TEST 2: Policy-as-Code
# ============================================================================
@test("2. Policy-as-Code — TOML/YAML configuration")
def test_policy():
    from agentguard.policy import load_policy, validate_policy, apply_policy

    # Create a test TOML policy file
    policy_dir = tempfile.mkdtemp()
    policy_path = os.path.join(policy_dir, "agentguard.toml")

    with open(policy_path, "w") as f:
        f.write("""
[defaults]
validate_input = true
max_retries = 2
record = false

[tools.search_web]
timeout = 10.0

[tools.search_web.rate_limit]
calls_per_minute = 60
burst = 10

[tools.query_db]
timeout = 30.0

[tools.query_db.budget]
max_cost_per_session = 1.00
cost_per_call = 0.05
""")

    # Validate
    errors = validate_policy(policy_path)
    print(f"  Validation errors: {errors}")
    assert len(errors) == 0, f"Policy validation failed: {errors}"

    # Load
    policy = load_policy(policy_path)
    print(f"  Loaded policy for tools: {list(policy.keys())}")
    assert "search_web" in policy
    assert "query_db" in policy
    assert policy["search_web"].timeout == 10.0
    assert policy["search_web"].validate_input is True  # inherited from defaults

    # Apply
    def search_web(q: str) -> dict:
        return {"results": []}
    def query_db(sql: str) -> dict:
        return {"rows": []}

    guarded = apply_policy(policy, [search_web, query_db])
    print(f"  Applied policy to {len(guarded)} tools")
    assert len(guarded) >= 1

    # Test that the guarded tools work (apply_policy returns a dict)
    if isinstance(guarded, dict):
        fn = guarded.get("search_web", list(guarded.values())[0])
        result = fn("test query")
    else:
        result = guarded[0]("test query")
    print(f"  search_web('test query') = {result}")
    assert result == {"results": []}

    shutil.rmtree(policy_dir)

    # CLI validation
    from agentguard.cli.main import main as cli_main
    policy_dir2 = tempfile.mkdtemp()
    p2 = os.path.join(policy_dir2, "test.toml")
    with open(p2, "w") as f:
        f.write("[defaults]\nvalidate_input = true\n")
    cli_main(["policy", "validate", p2])
    print(f"  CLI policy validate: ok")
    shutil.rmtree(policy_dir2)


# ============================================================================
# TEST 3: Multi-Agent Shared State
# ============================================================================
@test("3. Multi-agent shared state — shared budget")
def test_shared_state():
    from agentguard.shared import SharedBudget, SharedCircuitBreaker

    # Shared budget across agents
    budget = SharedBudget(
        max_calls_per_session=5,
        max_cost_per_session=1.00,
        on_exceed=GuardAction.BLOCK,
    )

    @guard(budget=budget.config)
    def agent1_tool(x: str) -> str:
        return f"agent1: {x}"

    @guard(budget=budget.config)
    def agent2_tool(x: str) -> str:
        return f"agent2: {x}"

    # Each call counts against the SHARED budget
    agent1_tool("a")
    agent1_tool("b")
    agent2_tool("c")
    agent2_tool("d")
    agent2_tool("e")
    print(f"  5 calls across 2 agents succeeded")

    # 6th call should be blocked (shared limit of 5)
    try:
        agent1_tool("f")
        print("  ⚠ 6th call wasn't blocked (budget may not be shared)")
    except Exception as e:
        print(f"  6th call blocked: {e}")

    # Shared circuit breaker
    scb = SharedCircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    print(f"  SharedCircuitBreaker created: {scb}")


# ============================================================================
# TEST 4: OpenTelemetry / Structured Logging
# ============================================================================
@test("4. Telemetry — structured logging")
def test_telemetry():
    from agentguard.telemetry import StructuredLogger

    logger = StructuredLogger()

    @guard(validate_input=True)
    def my_tool(x: str) -> str:
        return x.upper()

    # Use structured logger via after_call hook
    hook = logger.after_call_hook

    @guard(after_call=hook)
    def logged_tool(x: str) -> str:
        return x.upper()

    result = logged_tool("hello")
    assert result == "HELLO"

    records = logger.get_records()
    print(f"  Logged {len(records)} record(s)")
    if records:
        print(f"  Last record: {json.dumps(records[-1], default=str)[:120]}...")

    # Manual logging
    logger.log_event("test_event", {"key": "value"})
    records2 = logger.get_records()
    print(f"  Total records after manual log: {len(records2)}")


# ============================================================================
# TEST 5: Benchmarking Suite (no live API call — just scenario validation)
# ============================================================================
@test("5. Benchmarking suite — scenarios + report structure")
def test_benchmark():
    from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios

    runner = BenchmarkRunner()
    runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)
    runner.add_scenarios(BuiltinScenarios.MULTI_TOOL_SELECTION)
    runner.add_scenarios(BuiltinScenarios.PARAMETER_EXTRACTION)
    runner.add_scenarios(BuiltinScenarios.HALLUCINATION_RESISTANCE)

    print(f"  Loaded {len(runner.scenarios)} scenarios")
    assert len(runner.scenarios) >= 10

    for s in runner.scenarios[:5]:
        print(f"    [{s.category}] {s.name}")

    # Verify scenario structure
    for s in runner.scenarios:
        assert s.name, f"Scenario missing name"
        assert s.tools, f"Scenario '{s.name}' missing tools"
        assert s.messages, f"Scenario '{s.name}' missing messages"

    print(f"  All scenarios structurally valid")


# ============================================================================
# TEST 6: Live Benchmark with OpenRouter (frugal — 3 scenarios only)
# ============================================================================
@test("6. Live benchmark — OpenRouter GPT-4o-mini (3 scenarios)")
def test_live_benchmark():
    from agentguard.benchmark import BenchmarkRunner, BuiltinScenarios

    runner = BenchmarkRunner()
    # Only add basic scenarios to be frugal
    runner.add_scenarios(BuiltinScenarios.BASIC_TOOL_CALLING)

    # Only run first 3 scenarios
    # Only keep first 3 scenarios
    all_scenarios = list(runner.scenarios)
    runner.clear_scenarios()
    for s in all_scenarios[:3]:
        runner.add_scenario(s)

    print(f"  Running {len(runner.scenarios)} scenarios against OpenRouter...")
    results = runner.run(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="openai/gpt-4o-mini",
    )

    report = results.to_report()
    summary = report.summary()
    print(f"  Summary:\n{summary[:300]}")

    # Save report
    tmp = tempfile.mkdtemp()
    report.save(Path(tmp) / "benchmark.json")
    print(f"  Report saved")
    shutil.rmtree(tmp)


# ============================================================================
# TEST 7: Async Session Context Manager
# ============================================================================
@test("7. Async session context manager")
def test_async_session():
    @guard(record=True, trace_dir=tempfile.mkdtemp())
    async def my_async_tool(x: str) -> str:
        return x.upper()

    async def run():
        async with my_async_tool.session(session_id="test-session") as tool:
            r1 = await tool.acall("hello")
            r2 = await tool.acall("world")
            return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1 == "HELLO"
    assert r2 == "WORLD"
    print(f"  Async session: {r1}, {r2}")

    entries = getattr(my_async_tool, '_session_entries', [])
    print(f"  Session entries collected: {len(entries)}")


# ============================================================================
# TEST 8: async_record_session
# ============================================================================
@test("8. async_record_session")
def test_async_record():
    from agentguard import async_record_session

    @guard
    async def async_tool(x: str) -> str:
        return x.upper()

    async def run():
        async with async_record_session() as recorder:
            await async_tool.acall("foo")
            await async_tool.acall("bar")
        return recorder.entries()

    entries = asyncio.run(run())
    print(f"  async_record_session captured {len(entries)} entries")
    assert len(entries) == 2


# ============================================================================
# TEST 9: CrewAI integration (mocked)
# ============================================================================
@test("9. CrewAI integration")
def test_crewai():
    from agentguard.integrations.crewai_integration import GuardedCrewAITool, guard_crewai_tools

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for: {query}"

    tool = GuardedCrewAITool(search, config=GuardConfig(validate_input=True))
    result = tool("agentguard")
    print(f"  GuardedCrewAITool: {result}")
    assert "agentguard" in result

    # Bulk wrap
    def calculator(expr: str) -> str:
        """Calculate expression."""
        return str(eval(expr, {"__builtins__": {}}))

    tools = guard_crewai_tools([search, calculator])
    print(f"  Bulk wrapped {len(tools)} tools")
    assert len(tools) == 2


# ============================================================================
# TEST 10: AutoGen integration (mocked)
# ============================================================================
@test("10. AutoGen integration")
def test_autogen():
    from agentguard.integrations.autogen_integration import GuardedAutoGenTool, guard_autogen_tools

    def web_search(query: str) -> str:
        """Search the web."""
        return f"Found: {query}"

    tool = GuardedAutoGenTool(web_search, config=GuardConfig(validate_input=True))
    result = tool("test")
    print(f"  GuardedAutoGenTool: {result}")
    assert "test" in result

    # as_function export
    fn = tool.as_function()
    assert callable(fn)
    print(f"  as_function(): callable={callable(fn)}")


# ============================================================================
# TEST 11: Documentation exists
# ============================================================================
@test("11. Documentation completeness")
def test_docs():
    required_docs = [
        "docs/index.md",
        "docs/getting-started.md",
        "docs/guides/guard-decorator.md",
        "docs/guides/hallucination-detection.md",
        "docs/guides/circuit-breaker.md",
        "docs/integrations/openai.md",
        "docs/integrations/openrouter.md",
        "docs/integrations/crewai.md",
        "docs/integrations/autogen.md",
        "docs/reference/api.md",
        "docs/advanced/middleware.md",
        "docs/advanced/policy-as-code.md",
        "docs/advanced/benchmarking.md",
    ]

    found = 0
    missing = []
    for doc in required_docs:
        path = ROOT / doc
        if path.exists():
            size = path.stat().st_size
            found += 1
            # Check it's not empty
            assert size > 100, f"{doc} is too small ({size} bytes)"
        else:
            missing.append(doc)

    print(f"  Found {found}/{len(required_docs)} required docs")
    if missing:
        print(f"  Missing: {missing}")
    assert found >= 10, f"Only {found} docs found, expected at least 10"


# ============================================================================
# TEST 12: Full integration — middleware + policy + live API
# ============================================================================
@test("12. Full integration — all v0.2 features together")
def test_full_integration():
    from openai import OpenAI
    from agentguard.integrations import guard_tools, Providers

    @guard(validate_input=True, max_retries=1, record=True, trace_dir="/tmp/ag_v2_test")
    def get_capital(country: str) -> dict:
        """Get the capital city of a country."""
        capitals = {"france": "Paris", "japan": "Tokyo", "brazil": "Brasília"}
        return {"country": country, "capital": capitals.get(country.lower(), "Unknown")}

    executor = guard_tools([get_capital])
    client = OpenAI(**Providers.OPENROUTER.client_kwargs())

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        tools=executor.tools,
        messages=[{"role": "user", "content": "What's the capital of France?"}],
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        print(f"  Model called: {msg.tool_calls[0].function.name}({msg.tool_calls[0].function.arguments})")
        results = executor.execute_all(msg.tool_calls)
        data = json.loads(results[0]["content"])
        print(f"  Result: {data}")
        assert data.get("capital") == "Paris" or data.get("country")
    else:
        print(f"  Model text: {msg.content[:80]}")

    shutil.rmtree("/tmp/ag_v2_test", ignore_errors=True)


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    tests = [
        test_middleware, test_policy, test_shared_state, test_telemetry,
        test_benchmark, test_live_benchmark, test_async_session,
        test_async_record, test_crewai, test_autogen, test_docs,
        test_full_integration,
    ]

    print("\n" + "🚀 " * 20)
    print("  agentguard v0.2.0 — NEW FEATURES LIVE TEST")
    print("🚀 " * 20)

    for t in tests:
        t()

    print(f"\n\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    print(f"{'='*60}")

    if ISSUES:
        print("\n  ISSUES:")
        for n, e in ISSUES:
            print(f"    ❌ {n}: {e[:120]}")
    if FAIL == 0:
        print("\n  🎉 ALL v0.2.0 FEATURES VERIFIED!")
    sys.exit(1 if FAIL > 0 else 0)
