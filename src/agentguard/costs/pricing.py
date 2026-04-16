"""Pricing resolution for tracked LLM calls."""

from __future__ import annotations

from datetime import datetime, timezone

from agentguard.core.types import LLMCostBreakdown, LLMUsage


def _override_cost(
    usage: LLMUsage,
    *,
    model: str,
    pricing_overrides: dict[str, tuple[float, float]] | None,
) -> LLMCostBreakdown | None:
    if not pricing_overrides:
        return None
    if model not in pricing_overrides:
        return None
    input_rate, output_rate = pricing_overrides[model]
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    input_cost = (input_tokens * input_rate) / 1_000_000
    output_cost = (output_tokens * output_rate) / 1_000_000
    return LLMCostBreakdown(
        total_cost_usd=input_cost + output_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        pricing_source="override",
        priced_model=model,
        pricing_as_of=datetime.now(tz=timezone.utc),
        estimated=False,
        cost_known=True,
    )


def _litellm_cost(usage: LLMUsage, *, model: str) -> LLMCostBreakdown | None:
    try:
        from litellm import cost_per_token
    except ImportError:
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    try:
        input_cost, output_cost = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception:
        return None

    return LLMCostBreakdown(
        total_cost_usd=float(input_cost) + float(output_cost),
        input_cost_usd=float(input_cost),
        output_cost_usd=float(output_cost),
        pricing_source="litellm",
        priced_model=model,
        pricing_as_of=datetime.now(tz=timezone.utc),
        estimated=False,
        cost_known=True,
    )


def resolve_cost_breakdown(
    usage: LLMUsage | None,
    *,
    model: str,
    pricing_overrides: dict[str, tuple[float, float]] | None = None,
    cost_per_call: float | None = None,
) -> LLMCostBreakdown:
    """Resolve cost metadata for a normalized usage payload."""
    if usage is None:
        if cost_per_call is not None:
            return LLMCostBreakdown(
                total_cost_usd=cost_per_call,
                pricing_source="budget.cost_per_call",
                priced_model=model,
                pricing_as_of=datetime.now(tz=timezone.utc),
                estimated=True,
                cost_known=True,
            )
        return LLMCostBreakdown(
            pricing_source="unknown",
            priced_model=model,
            estimated=False,
            cost_known=False,
        )

    override = _override_cost(usage, model=model, pricing_overrides=pricing_overrides)
    if override is not None:
        return override

    litellm_breakdown = _litellm_cost(usage, model=model)
    if litellm_breakdown is not None:
        return litellm_breakdown

    if cost_per_call is not None:
        return LLMCostBreakdown(
            total_cost_usd=cost_per_call,
            pricing_source="budget.cost_per_call",
            priced_model=model,
            pricing_as_of=datetime.now(tz=timezone.utc),
            estimated=True,
            cost_known=True,
        )

    return LLMCostBreakdown(
        pricing_source="unknown",
        priced_model=model,
        estimated=False,
        cost_known=False,
    )
