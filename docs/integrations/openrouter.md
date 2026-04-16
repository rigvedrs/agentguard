# OpenRouter & OpenAI-Compatible Providers

## Overview

agentguard's `guard_tools` function works with **any OpenAI-compatible API** — not just OpenAI. One integration, 10+ providers:

- **OpenRouter** — 300+ models via a single API
- **Groq** — ultra-low latency LLaMA and Mixtral
- **Together AI** — open-source models
- **Fireworks AI** — fast open-source inference
- **DeepInfra** — affordable open-source hosting
- **Mistral** — Mistral's own API
- **Perplexity** — sonar models with web search built in
- **xAI** — Grok models
- **Novita AI** — affordable LLaMA models

---

## Quick Start

```python
import os
from openai import OpenAI
from agentguard.integrations import guard_tools, Providers

# Define and guard your tools
from agentguard import guard

@guard(validate_input=True, detect_hallucination=True, max_retries=2)
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

executor = guard_tools([search_web])

# Use with OpenRouter (300+ models)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What is quantum entanglement?"}],
    tools=executor.tools,
)

results = executor.execute_all(response.choices[0].message.tool_calls)
```

---

## Real Cost Tracking for Compatible Providers

Use `guard_openai_compatible_client` for OpenRouter and other OpenAI-compatible providers when you want real usage-based cost tracking in addition to guarded tool execution.

```python
import os
from openai import OpenAI

from agentguard import TokenBudget
from agentguard.integrations import (
    Providers,
    create_client,
    guard_openai_compatible_client,
    guard_tools,
)

budget = TokenBudget(max_cost_per_session=10.00)
executor = guard_tools([search_web])

base_client = create_client(Providers.OPENROUTER)
client = guard_openai_compatible_client(
    base_client,
    provider=Providers.OPENROUTER,
    budget=budget,
)

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=messages,
    tools=executor.tools,
)
```

OpenAI-compatible extraction uses a registry-based plugin system internally, so provider-specific usage formats can be added without replacing the default generic extractor.

---

## Built-in Provider Presets

The `Providers` enum has presets for common providers:

```python
from agentguard.integrations import Providers, create_client

# Each preset reads the API key from environment variables
client = create_client(Providers.OPENROUTER)   # OPENROUTER_API_KEY
client = create_client(Providers.GROQ)          # GROQ_API_KEY
client = create_client(Providers.TOGETHER)      # TOGETHER_API_KEY
client = create_client(Providers.FIREWORKS)     # FIREWORKS_API_KEY
client = create_client(Providers.DEEPINFRA)     # DEEPINFRA_API_KEY
client = create_client(Providers.MISTRAL)       # MISTRAL_API_KEY
client = create_client(Providers.PERPLEXITY)    # PERPLEXITY_API_KEY
client = create_client(Providers.XAI)           # XAI_API_KEY
client = create_client(Providers.NOVITA)        # NOVITA_API_KEY
client = create_client(Providers.OPENAI)        # OPENAI_API_KEY

# Or use kwargs directly
kwargs = Providers.TOGETHER.client_kwargs()
client = OpenAI(**kwargs)
```

---

## Custom Providers

For any OpenAI-compatible endpoint:

```python
from agentguard.integrations import Provider, create_client

my_provider = Provider(
    name="my-llm",
    base_url="https://my-llm-api.example.com/v1",
    env_key="MY_LLM_API_KEY",
)

client = create_client(my_provider)
```

---

## `guard_tools` vs `OpenAIToolExecutor`

Both work identically. `guard_tools` is the shortcut:

```python
from agentguard.integrations import guard_tools

executor = guard_tools([search_web, get_weather])
# equivalent to:
from agentguard.integrations import OpenAIToolExecutor
executor = OpenAIToolExecutor()
executor.register(search_web).register(get_weather)
```

---

## Provider-Specific Notes

### OpenRouter

```python
client = create_client(Providers.OPENROUTER)

# OpenRouter supports extra headers for app identification
response = client.chat.completions.create(
    model="openai/gpt-4o",  # or "anthropic/claude-3-5-sonnet", etc.
    messages=messages,
    tools=executor.tools,
    extra_headers={
        "HTTP-Referer": "https://myapp.com",
        "X-Title": "My Agent App",
    },
)
```

### Groq (ultra-low latency)

Groq is ideal for high-frequency tool-calling agents. agentguard's rate limiter helps stay within Groq's limits:

```python
from agentguard import guard
from agentguard.core.types import RateLimitConfig

@guard(rate_limit=RateLimitConfig(calls_per_minute=30))  # Groq free tier limit
def search_web(query: str) -> str: ...

client = create_client(Providers.GROQ)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=executor.tools,
)
```

### Perplexity (web search built in)

Perplexity's sonar models include web search natively. Use agentguard for the API call guardrails:

```python
client = create_client(Providers.PERPLEXITY)
response = client.chat.completions.create(
    model="llama-3.1-sonar-large-128k-online",
    messages=messages,
    tools=executor.tools,
)
```

---

## Switching Providers

The same `executor.tools` works across providers — just change the `client`:

```python
executor = guard_tools([search_web, get_weather])

# Test with Groq (fast, free tier)
groq_client = create_client(Providers.GROQ)

# Deploy with OpenRouter (reliability, 300+ models)
openrouter_client = create_client(Providers.OPENROUTER)

# Same executor, different client
for client, model in [
    (groq_client, "llama-3.3-70b-versatile"),
    (openrouter_client, "openai/gpt-4o"),
]:
    response = client.chat.completions.create(
        model=model, messages=messages, tools=executor.tools,
    )
```
