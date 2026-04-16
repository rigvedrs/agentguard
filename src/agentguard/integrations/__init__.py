"""agentguard integrations package.

Provides adapters for OpenAI, Anthropic, LangChain, MCP, CrewAI, AutoGen,
and any OpenAI-compatible provider (OpenRouter, Groq, Together AI,
Fireworks AI, DeepInfra, Mistral, xAI, etc.).
"""

from agentguard.integrations.anthropic_integration import (
    AnthropicToolExecutor,
    function_to_anthropic_tool,
    guard_anthropic_tools,
)
from agentguard.integrations.langchain_integration import (
    GuardedLangChainTool,
    guard_langchain_tools,
)
from agentguard.integrations.mcp_integration import GuardedMCPClient, GuardedMCPServer
from agentguard.integrations.crewai_integration import (
    GuardedCrewAITool,
    guard_crewai_tools,
)
from agentguard.integrations.autogen_integration import (
    GuardedAutoGenTool,
    guard_autogen_tool,
    guard_autogen_tools,
    register_guarded_tools,
)
from agentguard.integrations.openai_compatible import (
    Provider,
    Providers,
    create_client,
    guard_tools,
)
from agentguard.integrations.openai_integration import (
    OpenAIToolExecutor,
    execute_openai_tool_call,
    function_to_openai_tool,
    guard_openai_tools,
)
from agentguard.integrations.tracked_clients import (
    guard_anthropic_client,
    guard_openai_client,
    guard_openai_compatible_client,
)

__all__ = [
    # OpenAI
    "guard_openai_tools",
    "function_to_openai_tool",
    "execute_openai_tool_call",
    "OpenAIToolExecutor",
    "guard_openai_client",
    # OpenAI-compatible providers (OpenRouter, Groq, Together, Fireworks, etc.)
    "Provider",
    "Providers",
    "guard_tools",
    "create_client",
    "guard_openai_compatible_client",
    # Anthropic
    "guard_anthropic_tools",
    "function_to_anthropic_tool",
    "AnthropicToolExecutor",
    "guard_anthropic_client",
    # LangChain
    "GuardedLangChainTool",
    "guard_langchain_tools",
    # MCP
    "GuardedMCPServer",
    "GuardedMCPClient",
    # CrewAI
    "GuardedCrewAITool",
    "guard_crewai_tools",
    # AutoGen
    "GuardedAutoGenTool",
    "guard_autogen_tool",
    "guard_autogen_tools",
    "register_guarded_tools",
]
