"""LLM usage extraction, pricing, and spend recording helpers."""

from __future__ import annotations

from agentguard.costs.types import (
    CostLedger,
    LLMCostBreakdown,
    LLMSpendEvent,
    LLMUsage,
    UsageKind,
)

__all__ = [
    "CompatibleUsageExtractor",
    "CompatibleUsageExtractorRegistry",
    "compatible_usage_extractors",
    "extract_anthropic_message_usage",
    "extract_compatible_usage",
    "extract_openai_chat_usage",
    "extract_openai_response_usage",
    "register_compatible_usage_extractor",
    "InMemoryCostLedger",
    "NullCostLedger",
    "UsageKind",
    "LLMUsage",
    "LLMCostBreakdown",
    "LLMSpendEvent",
    "CostLedger",
    "resolve_cost_breakdown",
    "LLMCallTracker",
]


def __getattr__(name: str) -> object:
    if name in {
        "CompatibleUsageExtractor",
        "CompatibleUsageExtractorRegistry",
        "compatible_usage_extractors",
        "extract_anthropic_message_usage",
        "extract_compatible_usage",
        "extract_openai_chat_usage",
        "extract_openai_response_usage",
        "register_compatible_usage_extractor",
    }:
        from agentguard.costs import extractors as _extractors

        return getattr(_extractors, name)
    if name in {"InMemoryCostLedger", "NullCostLedger"}:
        from agentguard.costs import ledger as _ledger

        return getattr(_ledger, name)
    if name == "resolve_cost_breakdown":
        from agentguard.costs import pricing as _pricing

        return _pricing.resolve_cost_breakdown
    if name == "LLMCallTracker":
        from agentguard.costs import tracker as _tracker

        return _tracker.LLMCallTracker
    raise AttributeError(name)
