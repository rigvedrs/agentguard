"""Usage extraction helpers for tracked LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agentguard.core.types import LLMUsage, UsageKind


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(usage, "dict"):
        dumped = usage.dict()
        if isinstance(dumped, dict):
            return dumped
    data: dict[str, Any] = {}
    for name in dir(usage):
        if name.startswith("_"):
            continue
        try:
            value = getattr(usage, name)
        except Exception:
            continue
        if callable(value):
            continue
        if hasattr(value, "model_dump"):
            try:
                data[name] = value.model_dump()
                continue
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                data[name] = {
                    k: v for k, v in vars(value).items()
                    if not k.startswith("_")
                }
                continue
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool, dict, list, tuple)) or value is None:
            data[name] = value
    return data


def _detail_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return _usage_to_dict(obj)


def _usage_kind_from_payload(data: dict[str, Any]) -> UsageKind:
    has_text = any(data.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"))
    has_audio = any(data.get(key) is not None for key in ("audio_input_tokens", "audio_output_tokens"))
    has_image = any(data.get(key) is not None for key in ("image_input_units", "image_tokens"))
    if has_audio and has_image:
        return UsageKind.MULTIMODAL
    if has_audio and has_text:
        return UsageKind.MULTIMODAL
    if has_image and has_text:
        return UsageKind.MULTIMODAL
    if has_audio:
        return UsageKind.AUDIO
    if has_image:
        return UsageKind.IMAGE
    return UsageKind.TEXT


def _build_usage(data: dict[str, Any]) -> LLMUsage | None:
    if not data:
        return None

    prompt_details = _detail_dict(data.get("prompt_tokens_details"))
    completion_details = _detail_dict(data.get("completion_tokens_details"))

    input_tokens = data.get("prompt_tokens", data.get("input_tokens"))
    output_tokens = data.get("completion_tokens", data.get("output_tokens"))
    total_tokens = data.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    cached_input_tokens = (
        prompt_details.get("cached_tokens")
        if prompt_details.get("cached_tokens") is not None
        else data.get("cache_read_input_tokens")
    )
    cache_creation_tokens = data.get("cache_creation_input_tokens")
    reasoning_tokens = completion_details.get("reasoning_tokens")
    audio_input_tokens = prompt_details.get("audio_tokens", data.get("audio_input_tokens"))
    audio_output_tokens = completion_details.get("audio_tokens", data.get("audio_output_tokens"))
    image_input_units = data.get("image_input_units", data.get("image_tokens"))

    return LLMUsage(
        kind=_usage_kind_from_payload(data),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_tokens=cache_creation_tokens,
        reasoning_tokens=reasoning_tokens,
        audio_input_tokens=audio_input_tokens,
        audio_output_tokens=audio_output_tokens,
        image_input_units=image_input_units,
        raw_usage=data,
    )


def extract_openai_chat_usage(response: Any) -> LLMUsage | None:
    """Extract usage from an OpenAI chat completions response or chunk."""
    return _build_usage(_usage_to_dict(getattr(response, "usage", None)))


def extract_openai_response_usage(response: Any) -> LLMUsage | None:
    """Extract usage from an OpenAI Responses API response."""
    return _build_usage(_usage_to_dict(getattr(response, "usage", None)))


def extract_anthropic_message_usage(response: Any) -> LLMUsage | None:
    """Extract usage from an Anthropic messages response."""
    return _build_usage(_usage_to_dict(getattr(response, "usage", None)))


class CompatibleUsageExtractor(Protocol):
    """Protocol for provider-specific OpenAI-compatible usage extractors."""

    def matches(self, response: Any, provider: str | None, model: str | None) -> bool:
        """Return True when this extractor should handle the response."""

    def extract(self, response: Any) -> LLMUsage | None:
        """Extract normalized usage data from a response."""


@dataclass
class _FunctionCompatibleExtractor:
    matches_fn: Callable[[Any, str | None, str | None], bool]
    extract_fn: Callable[[Any], LLMUsage | None]

    def matches(self, response: Any, provider: str | None, model: str | None) -> bool:
        return self.matches_fn(response, provider, model)

    def extract(self, response: Any) -> LLMUsage | None:
        return self.extract_fn(response)


@dataclass
class CompatibleUsageExtractorRegistry:
    """Registry of OpenAI-compatible usage extractors."""

    _extractors: list[CompatibleUsageExtractor] = field(default_factory=list)

    def register(
        self,
        extractor: CompatibleUsageExtractor | None = None,
        *,
        matches: Callable[[Any, str | None, str | None], bool] | None = None,
        extract: Callable[[Any], LLMUsage | None] | None = None,
    ) -> CompatibleUsageExtractor:
        if extractor is None:
            if matches is None or extract is None:
                raise ValueError("matches and extract are required when extractor is not provided")
            extractor = _FunctionCompatibleExtractor(matches, extract)
        self._extractors.insert(0, extractor)
        return extractor

    def extract(self, response: Any, provider: str | None = None, model: str | None = None) -> LLMUsage | None:
        for extractor in self._extractors:
            try:
                if extractor.matches(response, provider, model):
                    return extractor.extract(response)
            except Exception:
                continue
        return None


compatible_usage_extractors = CompatibleUsageExtractorRegistry()


def register_compatible_usage_extractor(
    extractor: CompatibleUsageExtractor | None = None,
    *,
    matches: Callable[[Any, str | None, str | None], bool] | None = None,
    extract: Callable[[Any], LLMUsage | None] | None = None,
) -> CompatibleUsageExtractor:
    """Register an OpenAI-compatible usage extractor."""
    return compatible_usage_extractors.register(extractor, matches=matches, extract=extract)


def extract_compatible_usage(response: Any, provider: str | None = None, model: str | None = None) -> LLMUsage | None:
    """Extract usage through the compatible-provider plugin registry."""
    return compatible_usage_extractors.extract(response, provider=provider, model=model)


register_compatible_usage_extractor(
    matches=lambda response, provider, model: getattr(response, "usage", None) is not None,
    extract=extract_openai_chat_usage,
)
