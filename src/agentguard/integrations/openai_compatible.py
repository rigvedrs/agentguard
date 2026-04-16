"""OpenAI-compatible provider support for agentguard.

Most LLM API providers (OpenRouter, Groq, Together AI, Fireworks AI,
DeepInfra, Mistral, etc.) expose an OpenAI-compatible ``/chat/completions``
endpoint with tool calling support. This module provides:

- ``Provider`` — a lightweight config object for any OpenAI-compatible API.
- ``Providers`` — pre-built presets for popular providers.
- ``guard_tools`` — a universal wrapper that returns tool schemas + an executor.

Usage with any provider::

    from agentguard.integrations.openai_compatible import Providers, guard_tools

    # Option 1: Use a preset
    executor = guard_tools([search_web, query_db], provider=Providers.OPENROUTER)

    # Option 2: Custom provider
    from agentguard.integrations.openai_compatible import Provider
    my_provider = Provider(
        name="my-custom-api",
        base_url="https://api.my-llm.com/v1",
        env_key="MY_API_KEY",
    )
    executor = guard_tools([search_web], provider=my_provider)

Usage with OpenAI SDK (any provider)::

    from openai import OpenAI
    from agentguard.integrations.openai_compatible import Providers, guard_tools

    executor = guard_tools([search_web, get_weather], provider=Providers.GROQ)

    client = OpenAI(
        base_url=Providers.GROQ.base_url,
        api_key=os.getenv(Providers.GROQ.env_key),
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        tools=executor.tools,
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    )

    results = executor.execute_all(response.choices[0].message.tool_calls)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig
from agentguard.integrations.openai_integration import (
    OpenAIToolExecutor,
    function_to_openai_tool,
)


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    """Configuration for an OpenAI-compatible LLM API provider.

    Attributes:
        name: Human-readable provider name (e.g. "OpenRouter").
        base_url: The provider's API base URL.
        env_key: Environment variable name for the API key.
        default_model: A recommended default model identifier.
        default_headers: Extra headers required by the provider
            (e.g. ``HTTP-Referer`` for OpenRouter).
        supports_tools: Whether the provider supports tool/function calling.
        notes: Any usage notes or caveats.
    """

    name: str
    base_url: str
    env_key: str
    default_model: str = ""
    default_headers: dict[str, str] = field(default_factory=dict)
    supports_tools: bool = True
    notes: str = ""

    def get_api_key(self) -> Optional[str]:
        """Read the API key from the configured environment variable.

        Returns:
            The API key string, or None if not set.
        """
        return os.environ.get(self.env_key)

    def client_kwargs(self, api_key: Optional[str] = None) -> dict[str, Any]:
        """Return kwargs suitable for ``openai.OpenAI(...)`` construction.

        Args:
            api_key: Override the env-var key. If None, reads from env.

        Returns:
            Dict with ``base_url``, ``api_key``, and optionally
            ``default_headers``.
        """
        key = api_key or self.get_api_key()
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": key,
        }
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs

    def __repr__(self) -> str:
        return f"Provider({self.name!r}, base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------


class Providers:
    """Pre-built provider configurations for popular OpenAI-compatible APIs.

    Each attribute is a :class:`Provider` instance that can be passed to
    :func:`guard_tools` or used directly with the OpenAI SDK.

    Example::

        from agentguard.integrations.openai_compatible import Providers

        print(Providers.OPENROUTER.base_url)
        # "https://openrouter.ai/api/v1"

        print(Providers.GROQ.env_key)
        # "GROQ_API_KEY"
    """

    OPENAI = Provider(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o",
    )

    OPENROUTER = Provider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o",
        default_headers={"HTTP-Referer": "https://github.com/rigvedrs/agentguard"},
        notes="Unified gateway to 300+ models. Tool calling support varies by model.",
    )

    GROQ = Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        notes="Ultra-low latency inference. Supports tool calling on most models.",
    )

    TOGETHER = Provider(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        default_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        notes="High-performance open-source model hosting with native tool calling.",
    )

    FIREWORKS = Provider(
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        env_key="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        notes="Fast inference with FireFunction models for tool calling.",
    )

    DEEPINFRA = Provider(
        name="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        env_key="DEEPINFRA_API_KEY",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )

    MISTRAL = Provider(
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
        notes="Mistral's native API is OpenAI-compatible for chat completions.",
    )

    PERPLEXITY = Provider(
        name="Perplexity",
        base_url="https://api.perplexity.ai",
        env_key="PERPLEXITY_API_KEY",
        default_model="sonar-pro",
        supports_tools=False,
        notes="Perplexity models focus on search-augmented generation.",
    )

    NOVITA = Provider(
        name="Novita AI",
        base_url="https://api.novita.ai/v3/openai",
        env_key="NOVITA_API_KEY",
        default_model="meta-llama/llama-3.1-70b-instruct",
    )

    XAI = Provider(
        name="xAI",
        base_url="https://api.x.ai/v1",
        env_key="XAI_API_KEY",
        default_model="grok-2-latest",
        notes="xAI's Grok models with OpenAI-compatible API.",
    )

    @classmethod
    def all(cls) -> list[Provider]:
        """Return all pre-built provider presets.

        Returns:
            List of :class:`Provider` instances.
        """
        return [
            v for v in vars(cls).values()
            if isinstance(v, Provider)
        ]

    @classmethod
    def by_name(cls, name: str) -> Optional[Provider]:
        """Look up a provider by name (case-insensitive).

        Args:
            name: Provider name to search for.

        Returns:
            Matching :class:`Provider` or None.
        """
        lower = name.lower().replace(" ", "").replace("-", "").replace("_", "")
        for p in cls.all():
            pname = p.name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if pname == lower:
                return p
        return None


# ---------------------------------------------------------------------------
# Universal guard_tools helper
# ---------------------------------------------------------------------------


def guard_tools(
    functions: list[Callable[..., Any]],
    *,
    config: Optional[GuardConfig] = None,
    provider: Optional[Provider] = None,
) -> OpenAIToolExecutor:
    """Guard callables and build an executor for any OpenAI-compatible provider.

    This is the recommended entry point for using agentguard with providers
    like OpenRouter, Groq, Together AI, Fireworks, etc.

    Args:
        functions: Callables to guard and expose as tools.
        config: Guard config applied to all functions.
        provider: Optional provider preset (used for documentation only;
            the returned executor works with any provider's response format).

    Returns:
        An :class:`~agentguard.integrations.openai_integration.OpenAIToolExecutor`
        with all functions registered.

    Example::

        from agentguard.integrations.openai_compatible import guard_tools, Providers

        executor = guard_tools(
            [search_web, get_weather],
            config=GuardConfig(validate_input=True, max_retries=2),
            provider=Providers.GROQ,
        )

        # executor.tools → OpenAI-format tool schemas
        # executor.execute_all(tool_calls) → dispatch results
    """
    effective_config = config or GuardConfig()
    executor = OpenAIToolExecutor(config=effective_config)
    for fn in functions:
        executor.register(fn, config=effective_config)
    return executor


# ---------------------------------------------------------------------------
# Quick client factory (optional convenience)
# ---------------------------------------------------------------------------


def create_client(
    provider: Provider,
    api_key: Optional[str] = None,
) -> Any:
    """Create an OpenAI SDK client configured for the given provider.

    Requires ``openai`` to be installed.

    Args:
        provider: The provider to connect to.
        api_key: API key override. Defaults to reading from the
            provider's configured env var.

    Returns:
        An ``openai.OpenAI`` client instance.

    Raises:
        ImportError: If the ``openai`` package is not installed.
        ValueError: If no API key is available.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        )

    key = api_key or provider.get_api_key()
    if not key:
        raise ValueError(
            f"No API key found for {provider.name}. "
            f"Set the {provider.env_key} environment variable or pass api_key=."
        )

    return OpenAI(**provider.client_kwargs(api_key=key))
