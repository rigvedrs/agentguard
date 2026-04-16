"""Tests for LLM cost tracking wrappers and helpers."""

from __future__ import annotations

from types import SimpleNamespace

from agentguard import GuardConfig
from agentguard.core.trace import TraceStore
from agentguard.core.types import (
    LLMCostBreakdown,
    LLMUsage,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    TraceEntry,
)
from agentguard.core.types import UsageKind
from agentguard.costs import InMemoryCostLedger, register_compatible_usage_extractor
from agentguard.costs.extractors import extract_openai_chat_usage
from agentguard.guardrails.budget import TokenBudget
from agentguard.integrations import (
    Provider,
    guard_anthropic_client,
    guard_openai_client,
    guard_openai_compatible_client,
)
from agentguard.reporting.json_report import JsonReporter


def _chat_response(*, prompt_tokens: int = 10, completion_tokens: int = 5, model: str = "gpt-4o") -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    return SimpleNamespace(id="resp_1", model=model, usage=usage)


def test_extract_openai_chat_usage_preserves_details():
    usage = extract_openai_chat_usage(_chat_response())
    assert usage is not None
    assert usage.kind == UsageKind.TEXT
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.reasoning_tokens == 2


def test_guard_openai_client_records_spend_with_override():
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return _chat_response(model=str(kwargs["model"]))

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    ledger = InMemoryCostLedger()
    budget = TokenBudget(max_cost_per_session=1.0)
    config = GuardConfig(
        session_id="sess-1",
        record=True,
        trace_dir="/tmp/agentguard-cost-tests",
        budget=budget.config.model_copy(
            update={
                "model_pricing_overrides": {"gpt-4o": (2.0, 4.0)},
                "cost_ledger": ledger,
            }
        ),
    )

    client = guard_openai_client(FakeClient(), config=config, budget=budget)
    response = client.chat.completions.create(model="gpt-4o", messages=[])
    assert response.id == "resp_1"
    assert budget.session_spend > 0
    events = ledger.query(session_id="sess-1")
    assert len(events) == 1
    assert events[0].cost_breakdown is not None
    assert events[0].cost_breakdown.pricing_source == "override"
    assert events[0].cost_breakdown.pricing_as_of is not None


def test_guard_openai_client_stream_records_once():
    class FakeStream:
        def __init__(self) -> None:
            self._items = iter([
                SimpleNamespace(id="chunk_1", usage=None),
                SimpleNamespace(id="chunk_2", usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)),
            ])

        def __iter__(self) -> "FakeStream":
            return self

        def __next__(self) -> SimpleNamespace:
            return next(self._items)

    class FakeCompletions:
        def create(self, **kwargs: object) -> FakeStream:
            assert kwargs["stream"] is True
            return FakeStream()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    ledger = InMemoryCostLedger()
    config = GuardConfig(
        session_id="stream-sess",
        budget=TokenBudget().config.model_copy(
            update={
                "model_pricing_overrides": {"gpt-4o": (1.0, 2.0)},
                "cost_ledger": ledger,
            }
        ),
    )

    client = guard_openai_client(FakeClient(), config=config)
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    chunks = list(stream)
    assert len(chunks) == 2
    assert len(ledger.query(session_id="stream-sess")) == 1


def test_guard_anthropic_client_tracks_message_usage():
    class FakeMessages:
        def create(self, **kwargs: object) -> SimpleNamespace:
            usage = SimpleNamespace(input_tokens=20, output_tokens=4, total_tokens=24)
            return SimpleNamespace(id="msg_1", usage=usage, model=kwargs["model"])

    class FakeClient:
        def __init__(self) -> None:
            self.messages = FakeMessages()

    ledger = InMemoryCostLedger()
    config = GuardConfig(
        session_id="anth-sess",
        budget=TokenBudget().config.model_copy(
            update={
                "model_pricing_overrides": {"claude-3-5-sonnet": (3.0, 15.0)},
                "cost_ledger": ledger,
            }
        ),
    )

    client = guard_anthropic_client(FakeClient(), config=config)
    client.messages.create(model="claude-3-5-sonnet", messages=[])
    events = ledger.query(session_id="anth-sess")
    assert len(events) == 1
    assert events[0].usage is not None
    assert events[0].usage.input_tokens == 20


def test_openai_compatible_registry_allows_provider_specific_extractor():
    class CustomResponse:
        def __init__(self) -> None:
            self.id = "custom-1"
            self.model = "special-model"
            self.usage = None
            self.provider_usage = {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}

    class CustomCompletions:
        def create(self, **kwargs: object) -> CustomResponse:
            return CustomResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=CustomCompletions())

    register_compatible_usage_extractor(
        matches=lambda response, provider, model: provider == "custom" and hasattr(response, "provider_usage"),
        extract=lambda response: extract_openai_chat_usage(SimpleNamespace(usage=SimpleNamespace(**response.provider_usage))),
    )

    provider = Provider(name="Custom", base_url="https://example.com/v1", env_key="CUSTOM_KEY")
    ledger = InMemoryCostLedger()
    config = GuardConfig(
        session_id="compat-sess",
        budget=TokenBudget().config.model_copy(
            update={
                "model_pricing_overrides": {"special-model": (1.0, 1.0)},
                "cost_ledger": ledger,
            }
        ),
    )

    client = guard_openai_compatible_client(FakeClient(), provider=provider, config=config)
    client.chat.completions.create(model="special-model", messages=[])
    events = ledger.query(session_id="compat-sess")
    assert len(events) == 1
    assert events[0].usage is not None
    assert events[0].usage.total_tokens == 10


def test_json_reporter_includes_llm_cost_metadata(tmp_path):
    store = TraceStore(str(tmp_path))
    entry = TraceEntry(
        call=ToolCall(tool_name="llm.openai.chat_completions", session_id="report-sess"),
        result=ToolResult(
            call_id="call-1",
            tool_name="llm.openai.chat_completions",
            status=ToolCallStatus.SUCCESS,
            cost=0.001,
            provider="openai",
            model="gpt-4o",
            cost_known=True,
            usage=LLMUsage(kind=UsageKind.TEXT, input_tokens=10, output_tokens=5, total_tokens=15),
            cost_breakdown=LLMCostBreakdown(
                total_cost_usd=0.001,
                pricing_source="override",
                priced_model="gpt-4o",
                cost_known=True,
            ),
        ),
    )
    store.write(entry, session_id="report-sess")

    report = JsonReporter(store, session_id="report-sess").generate(include_entries=True)
    assert report["summary"]["cost_known_calls"] == 1
    assert report["entries"][0]["model"] == "gpt-4o"
    assert report["entries"][0]["cost_breakdown"]["pricing_source"] == "override"
