"""Cost and usage data models for LLM tracking."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class UsageKind(str, Enum):
    """Primary modality for a tracked model call."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class LLMUsage(BaseModel):
    """Normalized usage metadata for an LLM response."""

    kind: UsageKind = UsageKind.TEXT
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None
    image_input_units: Optional[int] = None
    raw_usage: Optional[dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


class LLMCostBreakdown(BaseModel):
    """Resolved cost metadata for a tracked LLM response."""

    total_cost_usd: Optional[float] = None
    input_cost_usd: Optional[float] = None
    cached_input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    reasoning_cost_usd: Optional[float] = None
    audio_cost_usd: Optional[float] = None
    image_cost_usd: Optional[float] = None
    pricing_source: Optional[str] = None
    priced_model: Optional[str] = None
    pricing_as_of: Optional[datetime] = None
    estimated: bool = False
    cost_known: bool = False

    model_config = {"arbitrary_types_allowed": True}


class LLMSpendEvent(BaseModel):
    """Single durable spend event emitted for an LLM model call."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: str
    request_kind: str = "chat_completion"
    request_id: Optional[str] = None
    stream: bool = False
    usage: Optional[LLMUsage] = None
    cost_breakdown: Optional[LLMCostBreakdown] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class CostLedger:
    """Interface for persisting and querying LLM spend events."""

    def record(self, event: LLMSpendEvent) -> None:
        raise NotImplementedError

    def query(
        self,
        *,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[LLMSpendEvent]:
        raise NotImplementedError
