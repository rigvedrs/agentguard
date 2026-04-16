"""Shared spend recorder used by wrapped LLM clients."""

from __future__ import annotations

import time
import uuid
from typing import Any

from agentguard.core.trace import TraceStore, get_active_recorders
from agentguard.core.types import (
    BudgetExceededError,
    BudgetConfig,
    LLMSpendEvent,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    TraceEntry,
)
from agentguard.costs.ledger import NullCostLedger
from agentguard.costs.pricing import resolve_cost_breakdown


def _raise_budget(detail: str) -> None:
    err = BudgetExceededError("llm_call", 0.0, 0.0)
    err.args = (detail,)
    raise err


def _budget_precheck(budget: Any) -> None:
    if budget is None:
        return
    if hasattr(budget, "check"):
        budget.check()
        return
    if hasattr(budget, "check_pre_call"):
        exceeded, detail = budget.check_pre_call()
        if exceeded:
            _raise_budget(detail)


def _budget_record_call(budget: Any) -> None:
    if budget is None:
        return
    if hasattr(budget, "record_call"):
        budget.record_call()


def _budget_record_spend(budget: Any, cost: float, *, tool_name: str) -> None:
    if budget is None:
        return
    if hasattr(budget, "record_spend"):
        try:
            budget.record_spend(cost, tool_name=tool_name)
        except TypeError:
            budget.record_spend(cost)


class LLMCallTracker:
    """Per-call recorder used by wrapped LLM client methods."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        request_kind: str,
        session_id: str | None,
        budget: Any = None,
        budget_config: BudgetConfig | None = None,
        trace_dir: str = "./traces",
        trace_backend: str = "sqlite",
        trace_db_path: str | None = None,
        record_trace: bool = False,
        metadata: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.request_kind = request_kind
        self.session_id = session_id
        self.budget = budget
        self.budget_config = budget_config or BudgetConfig()
        self.metadata = metadata or {}
        self.stream = stream
        self.event_id = str(uuid.uuid4())
        self._recorded = False
        self._trace_store = (
            TraceStore(directory=trace_dir, backend=trace_backend, db_path=trace_db_path)
            if record_trace
            else None
        )
        self._ledger = self.budget_config.cost_ledger or NullCostLedger()
        self._tool_name = f"llm.{provider}.{request_kind}"
        self._start = time.perf_counter()

    def precheck(self) -> None:
        _budget_precheck(self.budget)

    def record(
        self,
        *,
        usage: Any = None,
        request_id: str | None = None,
        extract_usage: Any = None,
        response: Any = None,
        status: ToolCallStatus = ToolCallStatus.SUCCESS,
        exception: Exception | None = None,
    ) -> LLMSpendEvent | None:
        if self._recorded:
            return None
        self._recorded = True

        normalized_usage = extract_usage(response) if extract_usage is not None else usage
        breakdown = resolve_cost_breakdown(
            normalized_usage,
            model=self.model,
            pricing_overrides=self.budget_config.model_pricing_overrides,
            cost_per_call=self.budget_config.cost_per_call,
        )

        event = LLMSpendEvent(
            event_id=self.event_id,
            session_id=self.session_id,
            provider=self.provider,
            model=self.model,
            request_kind=self.request_kind,
            request_id=request_id,
            stream=self.stream,
            usage=normalized_usage,
            cost_breakdown=breakdown,
            metadata=dict(self.metadata),
        )

        _budget_record_call(self.budget)
        if breakdown.cost_known and breakdown.total_cost_usd is not None:
            _budget_record_spend(self.budget, breakdown.total_cost_usd, tool_name=self._tool_name)

        self._ledger.record(event)
        self._write_trace(event=event, status=status, exception=exception)
        return event

    def _write_trace(
        self,
        *,
        event: LLMSpendEvent,
        status: ToolCallStatus,
        exception: Exception | None,
    ) -> None:
        call = ToolCall(
            tool_name=self._tool_name,
            session_id=self.session_id,
            metadata={
                "provider": self.provider,
                "model": self.model,
                "request_kind": self.request_kind,
                "event_id": event.event_id,
                **self.metadata,
            },
        )
        result = ToolResult(
            call_id=call.call_id,
            tool_name=self._tool_name,
            status=status,
            execution_time_ms=(time.perf_counter() - self._start) * 1000,
            exception=str(exception) if exception is not None else None,
            exception_type=type(exception).__qualname__ if exception is not None else None,
            cost=event.cost_breakdown.total_cost_usd if event.cost_breakdown else None,
            provider=self.provider,
            model=self.model,
            usage=event.usage,
            cost_breakdown=event.cost_breakdown,
            cost_estimated=(event.cost_breakdown.estimated if event.cost_breakdown else False),
            cost_known=(event.cost_breakdown.cost_known if event.cost_breakdown else False),
        )
        entry = TraceEntry(call=call, result=result)
        if self._trace_store:
            self._trace_store.write(entry, session_id=self.session_id)
        for recorder in get_active_recorders():
            recorder.record(entry)
