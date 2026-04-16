# LangChain Integration

## Overview

agentguard integrates with LangChain via `GuardedLangChainTool` — a `BaseTool`-compatible wrapper. Drop it into any LangChain agent that uses `tools=`.

## Installation

```bash
pip install awesome-agentguard langchain-core langchain-openai
```

---

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from agentguard.integrations import guard_langchain_tools, GuardedLangChainTool
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

# Define raw tool functions
def search_web(query: str) -> str:
    """Search the web for information."""
    import requests
    return requests.get(f"https://search.api.com?q={query}").text

def query_database(sql: str) -> str:
    """Query the database."""
    return str(db.execute(sql))

# Apply guards
config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=60),
)

guarded_tools = guard_langchain_tools([search_web, query_database], config=config)

# Build the LangChain agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, guarded_tools, prompt)
executor = AgentExecutor(agent=agent, tools=guarded_tools, verbose=True)

result = executor.invoke({"input": "What are the latest AI research papers?"})
```

---

## `GuardedLangChainTool` API

### From a function

```python
from agentguard.integrations import GuardedLangChainTool

guarded = GuardedLangChainTool.from_function(
    search_web,
    config=GuardConfig(validate_input=True),
    name="web_search",
    description="Search the web for current information",
)
```

### Direct construction

```python
guarded = GuardedLangChainTool(
    name="web_search",
    description="Search the web for current information",
    func=search_web,
    config=GuardConfig(validate_input=True),
)
```

### `guard_langchain_tools(functions, config=None)`

Bulk wrap:

```python
from agentguard.integrations import guard_langchain_tools

guarded = guard_langchain_tools([search_web, query_db], config=config)
```

---

## Converting to OpenAI Format

```python
tool = GuardedLangChainTool.from_function(search_web)
openai_schema = tool.to_openai_function()
```

---

## Using with LangGraph

```python
from langgraph.prebuilt import create_react_agent

guarded_tools = guard_langchain_tools([search_web, query_db], config=config)
llm = ChatOpenAI(model="gpt-4o")

graph = create_react_agent(llm, tools=guarded_tools)
result = graph.invoke({"messages": [("user", "Research quantum computing")]})
```

---

## Async Execution

`GuardedLangChainTool._arun` provides async execution:

```python
# LangChain calls _arun automatically in async agents
result = await guarded._arun(query="async search")
```

---

## Troubleshooting

### `langchain_core` not found

Install it: `pip install langchain-core`. The full `langchain` package is not required.

### Tool schema not generated correctly

LangChain uses `args_schema` (a Pydantic model) for schema generation. agentguard's `GuardedLangChainTool` returns `None` for `args_schema` — LangChain falls back to inspecting the function signature. Ensure your functions have proper type annotations.
