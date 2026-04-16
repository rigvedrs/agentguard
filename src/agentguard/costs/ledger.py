"""Spend ledger backends for tracked LLM calls."""

from __future__ import annotations

import threading

from agentguard.costs.types import CostLedger, LLMSpendEvent


class NullCostLedger(CostLedger):
    """No-op ledger used when persistence is disabled."""

    def record(self, event: LLMSpendEvent) -> None:
        del event

    def query(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[LLMSpendEvent]:
        del session_id, provider, model
        return []


class InMemoryCostLedger(CostLedger):
    """Thread-safe in-memory ledger for spend events."""

    def __init__(self) -> None:
        self._events: list[LLMSpendEvent] = []
        self._lock = threading.Lock()

    def record(self, event: LLMSpendEvent) -> None:
        with self._lock:
            self._events.append(event)

    def query(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[LLMSpendEvent]:
        with self._lock:
            events = list(self._events)
        if session_id is not None:
            events = [e for e in events if e.session_id == session_id]
        if provider is not None:
            events = [e for e in events if e.provider == provider]
        if model is not None:
            events = [e for e in events if e.model == model]
        return events
