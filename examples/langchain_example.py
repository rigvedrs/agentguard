"""LangChain integration example for agentguard.

Demonstrates GuardedLangChainTool usage with langchain agents.
Run with: python examples/langchain_example.py
(Works with or without langchain installed — mock mode available.)
"""

from __future__ import annotations

import time

from agentguard import GuardConfig, guard
from agentguard.integrations import GuardedLangChainTool, guard_langchain_tools


# ---------------------------------------------------------------------------
# Define the tools (plain Python functions)
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Search the web for information."""
    time.sleep(0.01)
    return f"Search results for '{query}': [Result 1, Result 2, Result 3]"


def database_lookup(table: str, filter_by: str = "") -> dict:
    """Look up records in the database."""
    time.sleep(0.005)
    return {
        "table": table,
        "filter": filter_by,
        "rows": [
            {"id": 1, "value": "record A"},
            {"id": 2, "value": "record B"},
        ],
        "count": 2,
    }


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email (simulated)."""
    time.sleep(0.02)
    return {"sent": True, "to": to, "subject": subject, "message_id": "msg_12345"}


# ---------------------------------------------------------------------------
# Wrap with agentguard
# ---------------------------------------------------------------------------


config = GuardConfig(
    validate_input=True,
    max_retries=2,
    timeout=10.0,
    record=True,
    trace_dir="/tmp/agentguard_lc",
)


def build_guarded_tools() -> list[GuardedLangChainTool]:
    """Create guarded LangChain tools."""
    tools = guard_langchain_tools(
        [web_search, database_lookup, send_email],
        config=config,
    )
    return tools


# ---------------------------------------------------------------------------
# Mock agent loop
# ---------------------------------------------------------------------------


def mock_agent_loop(tools: list[GuardedLangChainTool]) -> None:
    """Simulate an agent tool-calling loop."""
    print("\n--- Mock Agent Loop ---")
    print(f"Agent has {len(tools)} tools: {[t.name for t in tools]}")

    # Build a name -> tool map
    tool_map = {t.name: t for t in tools}

    # Simulate some agent tool calls
    plan = [
        ("web_search", ("latest Python releases",), {}),
        ("database_lookup", ("users",), {"filter_by": "active=true"}),
        ("send_email", (), {"to": "user@example.com", "subject": "Report", "body": "See attached"}),
    ]

    for tool_name, args, kwargs in plan:
        tool = tool_map[tool_name]
        print(f"\n  Calling: {tool_name}")
        result = tool(*args, **kwargs)
        print(f"  Result: {str(result)[:80]}...")


# ---------------------------------------------------------------------------
# LangChain agent (if installed)
# ---------------------------------------------------------------------------


def run_langchain_agent(tools: list[GuardedLangChainTool]) -> None:
    """Run a real LangChain agent if dependencies are available."""
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        print("\n(langchain-core not installed; skipping live agent demo)")
        return

    try:
        from langchain_openai import ChatOpenAI
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("\n(OPENAI_API_KEY not set; skipping live LangChain agent)")
            return
    except ImportError:
        print("\n(langchain-openai not installed; skipping live agent)")
        return

    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Convert to proper LangChain tools (duck-typing works here)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = executor.invoke({"input": "Search for Python 3.13 features"})
    print(f"\nAgent result: {result['output']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard + LangChain Integration Demo")
    print("=" * 60)

    tools = build_guarded_tools()

    print("\nCreated guarded tools:")
    for t in tools:
        schema = t.to_openai_function()
        fn = schema["function"]
        print(f"  {t.name}: {t.description}")
        print(f"    params: {list(fn['parameters']['properties'].keys())}")

    mock_agent_loop(tools)
    run_langchain_agent(tools)

    # Individual tool creation with custom config
    print("\n--- Custom tool wrapping ---")
    custom_config = GuardConfig(validate_input=True, max_retries=1, timeout=5.0)
    search_tool = GuardedLangChainTool.from_function(
        web_search,
        config=custom_config,
        name="search",
        description="Search the internet for current information",
    )
    result = search_tool("agentguard middleware")
    print(f"  Search result: {str(result)[:60]}...")

    print("\n✓ LangChain example complete!")


if __name__ == "__main__":
    main()
