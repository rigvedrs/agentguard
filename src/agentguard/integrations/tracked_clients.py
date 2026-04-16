"""Wrapped LLM clients that record real usage and spend."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from agentguard.core.types import BudgetConfig, GuardConfig, ToolCallStatus
from agentguard.costs.extractors import (
    extract_anthropic_message_usage,
    extract_compatible_usage,
    extract_openai_chat_usage,
    extract_openai_response_usage,
)
from agentguard.costs.tracker import LLMCallTracker
from agentguard.guardrails.budget import TokenBudget
from agentguard.integrations.openai_compatible import Provider


def _build_budget_runtime(config: GuardConfig | None, budget: Any) -> Any:
    if budget is not None:
        return budget
    if config is None or config.budget is None:
        return None
    budget_cfg = config.budget
    return TokenBudget(
        max_cost_per_call=budget_cfg.max_cost_per_call,
        max_cost_per_session=budget_cfg.max_cost_per_session,
        max_calls_per_session=budget_cfg.max_calls_per_session,
        alert_threshold=budget_cfg.alert_threshold,
        on_exceed=budget_cfg.on_exceed,
        cost_per_call=budget_cfg.cost_per_call,
    )


def _budget_config(config: GuardConfig | None) -> BudgetConfig:
    if config is None or config.budget is None:
        return BudgetConfig()
    return config.budget


def _request_id(response: Any) -> str | None:
    return getattr(response, "id", None)


class _TrackedSyncStream:
    def __init__(self, stream: Any, tracker: LLMCallTracker, extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._tracker = tracker
        self._extractor = extractor
        self._last_item: Any = None

    def __iter__(self) -> "_TrackedSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._stream)
        except StopIteration:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
            )
            raise
        self._last_item = item
        return item

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
            )

    def __enter__(self) -> "_TrackedSyncStream":
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            exit_fn = getattr(self._stream, "__exit__", None)
            if callable(exit_fn):
                exit_fn(exc_type, exc, tb)
        finally:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
                status=ToolCallStatus.FAILURE if exc is not None else ToolCallStatus.SUCCESS,
                exception=exc,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        try:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
            )
        except Exception:
            pass


class _TrackedAsyncStream:
    def __init__(self, stream: Any, tracker: LLMCallTracker, extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._tracker = tracker
        self._extractor = extractor
        self._last_item: Any = None

    def __aiter__(self) -> "_TrackedAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._stream.__anext__()
        except StopAsyncIteration:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
            )
            raise
        self._last_item = item
        return item

    async def aclose(self) -> None:
        try:
            close = getattr(self._stream, "aclose", None)
            if callable(close):
                await close()
        finally:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
            )

    async def __aenter__(self) -> "_TrackedAsyncStream":
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            exit_fn = getattr(self._stream, "__aexit__", None)
            if callable(exit_fn):
                await exit_fn(exc_type, exc, tb)
        finally:
            self._tracker.record(
                response=self._last_item,
                extract_usage=self._extractor,
                request_id=_request_id(self._last_item),
                status=ToolCallStatus.FAILURE if exc is not None else ToolCallStatus.SUCCESS,
                exception=exc,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_response(response: Any, *, tracker: LLMCallTracker, extractor: Callable[[Any], Any], stream: bool) -> Any:
    if stream:
        if hasattr(response, "__aiter__"):
            return _TrackedAsyncStream(response, tracker, extractor)
        return _TrackedSyncStream(iter(response), tracker, extractor)

    tracker.record(
        response=response,
        extract_usage=extractor,
        request_id=_request_id(response),
    )
    return response


def _invoke_tracked_create(
    create_fn: Callable[..., Any],
    *,
    provider: str,
    request_kind: str,
    extractor: Callable[[Any], Any],
    config: GuardConfig | None,
    budget: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    model = kwargs.get("model") or (args[0] if args else "unknown")
    stream = bool(kwargs.get("stream"))
    tracker = LLMCallTracker(
        provider=provider,
        model=str(model),
        request_kind=request_kind,
        session_id=(config.session_id if config is not None else None),
        budget=budget,
        budget_config=_budget_config(config),
        trace_dir=(config.trace_dir if config is not None else "./traces"),
        trace_backend=(config.trace_backend if config is not None else "sqlite"),
        trace_db_path=(config.trace_db_path if config is not None else None),
        record_trace=(config.record if config is not None else False),
        metadata={"stream": stream},
        stream=stream,
    )
    tracker.precheck()

    try:
        result = create_fn(*args, **kwargs)
    except Exception as exc:
        tracker.record(
            response=None,
            extract_usage=extractor,
            status=ToolCallStatus.FAILURE,
            exception=exc,
        )
        raise

    if inspect.isawaitable(result):
        async def _await_and_wrap() -> Any:
            try:
                response = await result
            except Exception as exc:
                tracker.record(
                    response=None,
                    extract_usage=extractor,
                    status=ToolCallStatus.FAILURE,
                    exception=exc,
                )
                raise
            return _wrap_response(response, tracker=tracker, extractor=extractor, stream=stream)

        return _await_and_wrap()

    return _wrap_response(result, tracker=tracker, extractor=extractor, stream=stream)


class _TrackedCreateProxy:
    def __init__(
        self,
        create_fn: Callable[..., Any],
        *,
        provider: str,
        request_kind: str,
        extractor: Callable[[Any], Any],
        config: GuardConfig | None,
        budget: Any,
    ) -> None:
        self._create_fn = create_fn
        self._provider = provider
        self._request_kind = request_kind
        self._extractor = extractor
        self._config = config
        self._budget = budget

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return _invoke_tracked_create(
            self._create_fn,
            provider=self._provider,
            request_kind=self._request_kind,
            extractor=self._extractor,
            config=self._config,
            budget=self._budget,
            args=args,
            kwargs=kwargs,
        )


class _Proxy:
    def __init__(self, target: Any) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class TrackedOpenAIClient(_Proxy):
    """Wrap an OpenAI client and track chat/responses usage and spend."""

    def __init__(self, client: Any, *, config: GuardConfig | None = None, budget: Any = None) -> None:
        super().__init__(client)
        budget_runtime = _build_budget_runtime(config, budget)
        self.chat = _Proxy(client.chat)
        self.chat.completions = _TrackedCreateProxy(
            client.chat.completions.create,
            provider="openai",
            request_kind="chat_completions",
            extractor=extract_openai_chat_usage,
            config=config,
            budget=budget_runtime,
        )
        if hasattr(client, "responses"):
            self.responses = _TrackedCreateProxy(
                client.responses.create,
                provider="openai",
                request_kind="responses",
                extractor=extract_openai_response_usage,
                config=config,
                budget=budget_runtime,
            )


class TrackedOpenAICompatibleClient(_Proxy):
    """Wrap an OpenAI-compatible client and track usage through the extractor registry."""

    def __init__(
        self,
        client: Any,
        *,
        provider: Provider,
        config: GuardConfig | None = None,
        budget: Any = None,
    ) -> None:
        super().__init__(client)
        budget_runtime = _build_budget_runtime(config, budget)
        provider_name = provider.name.lower().replace(" ", "_")
        extractor = lambda response: extract_compatible_usage(response, provider=provider_name, model=getattr(response, "model", None))
        self.chat = _Proxy(client.chat)
        self.chat.completions = _TrackedCreateProxy(
            client.chat.completions.create,
            provider=provider_name,
            request_kind="chat_completions",
            extractor=extractor,
            config=config,
            budget=budget_runtime,
        )
        if hasattr(client, "responses"):
            self.responses = _TrackedCreateProxy(
                client.responses.create,
                provider=provider_name,
                request_kind="responses",
                extractor=extractor,
                config=config,
                budget=budget_runtime,
            )


class TrackedAnthropicClient(_Proxy):
    """Wrap an Anthropic client and track message usage and spend."""

    def __init__(self, client: Any, *, config: GuardConfig | None = None, budget: Any = None) -> None:
        super().__init__(client)
        budget_runtime = _build_budget_runtime(config, budget)
        self.messages = _TrackedCreateProxy(
            client.messages.create,
            provider="anthropic",
            request_kind="messages",
            extractor=extract_anthropic_message_usage,
            config=config,
            budget=budget_runtime,
        )


def guard_openai_client(client: Any, *, config: GuardConfig | None = None, budget: Any = None) -> TrackedOpenAIClient:
    """Return a wrapped OpenAI client that records usage and spend."""
    return TrackedOpenAIClient(client, config=config, budget=budget)


def guard_openai_compatible_client(
    client: Any,
    *,
    provider: Provider,
    config: GuardConfig | None = None,
    budget: Any = None,
) -> TrackedOpenAICompatibleClient:
    """Return a wrapped OpenAI-compatible client that records usage and spend."""
    return TrackedOpenAICompatibleClient(client, provider=provider, config=config, budget=budget)


def guard_anthropic_client(client: Any, *, config: GuardConfig | None = None, budget: Any = None) -> TrackedAnthropicClient:
    """Return a wrapped Anthropic client that records usage and spend."""
    return TrackedAnthropicClient(client, config=config, budget=budget)
