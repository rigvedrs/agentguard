"""Hallucination detection for AI agent tool calls.

This module detects when an LLM *claims* it called a tool but the tool never
actually executed. Hallucinated responses are typically characterised by:

1. **Impossibly fast execution** — a real API call takes ≥100ms; a fabricated
   response takes <1ms.
2. **Missing required fields** — real tool responses always include certain keys
   (e.g. ``temperature`` for weather APIs).
3. **Response pattern mismatch** — real responses match known regex patterns;
   hallucinated ones may not.
4. **Suspiciously perfect structure** — values at round numbers, no variance.

The detector uses a weighted signal approach: each signal produces a score
(0.0–1.0) and the final ``confidence`` is a weighted average. A threshold
(default 0.6) determines ``is_hallucinated``.

Usage::

    from agentguard import HallucinationDetector

    detector = HallucinationDetector()
    detector.register_tool(
        "get_weather",
        expected_latency_ms=(100, 5000),
        required_fields=["temperature", "humidity"],
        response_patterns=[r'"temp":\\s*-?\\d+'],
    )

    result = detector.verify(
        tool_name="get_weather",
        execution_time_ms=0.4,
        response={"temperature": 72, "humidity": 50},
    )
    # result.is_hallucinated → True (execution was impossibly fast)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agentguard.core.types import HallucinationResult


# ---------------------------------------------------------------------------
# Tool profile
# ---------------------------------------------------------------------------


@dataclass
class ToolProfile:
    """Expected behaviour profile for a single tool.

    Attributes:
        tool_name: The registered tool name.
        expected_latency_ms: ``(min_ms, max_ms)`` range for real execution.
            Calls outside this range are suspicious.
        required_fields: Field names that *must* appear in the response.
        forbidden_fields: Field names that should *never* appear in a real
            response (e.g. ``"error"`` when the tool always succeeds).
        response_patterns: Regex patterns that at least one of must match the
            JSON-serialised response.
        min_response_length: Minimum length of the serialised response.
        max_response_length: Maximum length of the serialised response.
        latency_weight: Weight of the latency signal in the final score.
        fields_weight: Weight of the field-presence signal.
        patterns_weight: Weight of the regex-pattern signal.
    """

    tool_name: str
    expected_latency_ms: tuple[float, float] = (50.0, 30_000.0)
    required_fields: list[str] = field(default_factory=list)
    forbidden_fields: list[str] = field(default_factory=list)
    response_patterns: list[str] = field(default_factory=list)
    min_response_length: Optional[int] = None
    max_response_length: Optional[int] = None
    latency_weight: float = 0.50
    fields_weight: float = 0.30
    patterns_weight: float = 0.20


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class HallucinationDetector:
    """Multi-signal hallucination detector for guarded tool calls.

    Register tool profiles with :meth:`register_tool` to enable profile-based
    checks. Without a profile, the detector falls back to a universal latency
    heuristic (< 2ms is almost certainly hallucinated).

    Example::

        detector = HallucinationDetector(threshold=0.6)
        detector.register_tool(
            "search_web",
            expected_latency_ms=(200, 10_000),
            required_fields=["results"],
        )
        result = detector.verify("search_web", execution_time_ms=0.2, response={...})
    """

    #: Calls faster than this (ms) are universally suspect regardless of profile
    UNIVERSAL_MIN_LATENCY_MS: float = 2.0

    def __init__(self, threshold: float = 0.6) -> None:
        """Initialise the detector.

        Args:
            threshold: Confidence score at or above which a call is flagged as
                hallucinated (0–1, default 0.6).
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._profiles: dict[str, ToolProfile] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        tool_name: str,
        *,
        expected_latency_ms: tuple[float, float] = (50.0, 30_000.0),
        required_fields: Optional[list[str]] = None,
        forbidden_fields: Optional[list[str]] = None,
        response_patterns: Optional[list[str]] = None,
        min_response_length: Optional[int] = None,
        max_response_length: Optional[int] = None,
        latency_weight: float = 0.50,
        fields_weight: float = 0.30,
        patterns_weight: float = 0.20,
    ) -> "HallucinationDetector":
        """Register an expected-behaviour profile for *tool_name*.

        Args:
            tool_name: The name of the tool to profile.
            expected_latency_ms: ``(min_ms, max_ms)`` range for real execution.
            required_fields: Keys that must appear in real responses.
            forbidden_fields: Keys that must not appear in real responses.
            response_patterns: Regex patterns real responses match.
            min_response_length: Minimum serialised response length.
            max_response_length: Maximum serialised response length.
            latency_weight: Relative weight of the latency signal (0–1).
            fields_weight: Relative weight of the field-presence signal (0–1).
            patterns_weight: Relative weight of the regex-pattern signal (0–1).

        Returns:
            Self, to allow chaining.
        """
        self._profiles[tool_name] = ToolProfile(
            tool_name=tool_name,
            expected_latency_ms=expected_latency_ms,
            required_fields=required_fields or [],
            forbidden_fields=forbidden_fields or [],
            response_patterns=response_patterns or [],
            min_response_length=min_response_length,
            max_response_length=max_response_length,
            latency_weight=latency_weight,
            fields_weight=fields_weight,
            patterns_weight=patterns_weight,
        )
        return self

    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool profile.

        Args:
            tool_name: The tool to unregister.
        """
        self._profiles.pop(tool_name, None)

    def get_profile(self, tool_name: str) -> Optional[ToolProfile]:
        """Return the profile for *tool_name*, or None if unregistered."""
        return self._profiles.get(tool_name)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        tool_name: str,
        execution_time_ms: float,
        response: Any,
        *,
        call_stack_verified: bool = False,
    ) -> HallucinationResult:
        """Analyse a tool call for hallucination signals.

        Args:
            tool_name: The name of the tool that was called.
            execution_time_ms: Wall-clock time in milliseconds.
            response: The value returned by the tool.
            call_stack_verified: If True, the call was confirmed via stack
                inspection; latency signal weight is reduced.

        Returns:
            :class:`~agentguard.core.types.HallucinationResult` with
            ``is_hallucinated``, ``confidence``, ``reason``, and ``signals``.
        """
        signals: dict[str, Any] = {"execution_time_ms": execution_time_ms}
        scores: list[tuple[float, float]] = []  # (score, weight)
        reasons: list[str] = []

        # --- Universal latency check ---
        if execution_time_ms < self.UNIVERSAL_MIN_LATENCY_MS and not call_stack_verified:
            confidence = 1.0 - (execution_time_ms / self.UNIVERSAL_MIN_LATENCY_MS)
            signals["universal_latency_flag"] = True
            reasons.append(
                f"Execution time {execution_time_ms:.2f}ms is below the universal "
                f"minimum of {self.UNIVERSAL_MIN_LATENCY_MS:.0f}ms for any real I/O call"
            )
            return HallucinationResult(
                is_hallucinated=True,
                confidence=round(min(1.0, confidence), 4),
                reason="; ".join(reasons),
                signals=signals,
            )

        profile = self._profiles.get(tool_name)
        if profile is None:
            # No profile registered — use a simple latency heuristic only
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.0,
                reason="No profile registered; cannot perform detailed analysis",
                signals=signals,
            )

        # --- Latency signal ---
        latency_score = _latency_score(
            execution_time_ms,
            profile.expected_latency_ms,
            call_stack_verified=call_stack_verified,
        )
        signals["latency_score"] = latency_score
        signals["expected_latency_ms"] = profile.expected_latency_ms
        if latency_score > 0.0:
            reasons.append(
                f"Execution time {execution_time_ms:.2f}ms is outside expected "
                f"range {profile.expected_latency_ms[0]:.0f}–{profile.expected_latency_ms[1]:.0f}ms"
            )
        scores.append((latency_score, profile.latency_weight))

        # --- Required / forbidden fields signal ---
        if profile.required_fields or profile.forbidden_fields:
            fields_score, fields_reason = _fields_score(
                response,
                profile.required_fields,
                profile.forbidden_fields,
            )
            signals["fields_score"] = fields_score
            signals["required_fields"] = profile.required_fields
            if fields_score > 0.0:
                reasons.append(fields_reason)
            scores.append((fields_score, profile.fields_weight))

        # --- Response pattern signal ---
        if profile.response_patterns:
            pattern_score, pattern_reason = _pattern_score(response, profile.response_patterns)
            signals["pattern_score"] = pattern_score
            if pattern_score > 0.0:
                reasons.append(pattern_reason)
            scores.append((pattern_score, profile.patterns_weight))

        # --- Response length signal ---
        if profile.min_response_length or profile.max_response_length:
            length_score, length_reason = _length_score(
                response, profile.min_response_length, profile.max_response_length
            )
            signals["length_score"] = length_score
            if length_score > 0.0:
                reasons.append(length_reason)
            scores.append((length_score, 0.10))

        # --- Weighted confidence ---
        if scores:
            total_weight = sum(w for _, w in scores)
            confidence = sum(s * w for s, w in scores) / total_weight
        else:
            confidence = 0.0

        confidence = round(min(1.0, max(0.0, confidence)), 4)
        is_hallucinated = confidence >= self.threshold

        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            reason="; ".join(reasons) if reasons else "All signals within expected range",
            signals=signals,
        )

    def batch_verify(
        self,
        calls: list[dict[str, Any]],
    ) -> list[HallucinationResult]:
        """Verify multiple tool calls in a batch.

        Args:
            calls: List of dicts, each with keys matching :meth:`verify` parameters.

        Returns:
            List of :class:`~agentguard.core.types.HallucinationResult` in the same order.
        """
        return [self.verify(**call) for call in calls]


# ---------------------------------------------------------------------------
# Signal scoring helpers
# ---------------------------------------------------------------------------


def _latency_score(
    actual_ms: float,
    expected_range: tuple[float, float],
    *,
    call_stack_verified: bool = False,
) -> float:
    """Return a hallucination score (0–1) for the given latency.

    0 means the latency is perfectly normal; 1 means it is maximally suspicious.
    """
    if call_stack_verified:
        return 0.0
    min_ms, max_ms = expected_range
    if actual_ms < min_ms:
        # Below minimum — suspicious. Score scales from 0 (at min) to 1 (at 0)
        return max(0.0, 1.0 - (actual_ms / min_ms))
    if actual_ms > max_ms:
        # Above maximum — could be a slow real call; score is low
        excess_ratio = (actual_ms - max_ms) / max_ms
        return min(0.3, excess_ratio * 0.1)
    return 0.0


def _fields_score(
    response: Any,
    required: list[str],
    forbidden: list[str],
) -> tuple[float, str]:
    """Return (score, reason) based on required/forbidden field presence."""
    if not isinstance(response, dict):
        try:
            response = json.loads(json.dumps(response, default=str))
        except Exception:
            pass

    if not isinstance(response, dict):
        return 0.0, ""

    missing = [f for f in required if f not in response]
    present_forbidden = [f for f in forbidden if f in response]
    total_checks = len(required) + len(forbidden)
    if total_checks == 0:
        return 0.0, ""

    violations = len(missing) + len(present_forbidden)
    score = violations / total_checks
    parts: list[str] = []
    if missing:
        parts.append(f"Missing required fields: {missing}")
    if present_forbidden:
        parts.append(f"Forbidden fields present: {present_forbidden}")
    return score, "; ".join(parts)


def _pattern_score(response: Any, patterns: list[str]) -> tuple[float, str]:
    """Return (score, reason) based on regex pattern matching.

    Score is 0 if at least one pattern matches, 1 if none match.
    """
    if not patterns:
        return 0.0, ""
    try:
        serialised = json.dumps(response, default=str)
    except Exception:
        serialised = str(response)

    for pat in patterns:
        if re.search(pat, serialised):
            return 0.0, ""
    return 0.8, f"Response matched none of {len(patterns)} expected pattern(s)"


def _length_score(
    response: Any,
    min_length: Optional[int],
    max_length: Optional[int],
) -> tuple[float, str]:
    """Return (score, reason) based on response length."""
    try:
        serialised = json.dumps(response, default=str)
        length = len(serialised)
    except Exception:
        return 0.0, ""

    if min_length and length < min_length:
        return 0.5, f"Response length {length} < minimum {min_length}"
    if max_length and length > max_length:
        return 0.2, f"Response length {length} > maximum {max_length}"
    return 0.0, ""
