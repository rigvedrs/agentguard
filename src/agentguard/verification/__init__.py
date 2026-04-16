"""Intelligent verification engine for agentguard.

The verification package provides a Bayesian multi-signal hallucination
detection engine that replaces the simple heuristics-based detector.

Architecture (from research Section 7.2):
- Tier 0: Zero-cost schema, latency, pattern, and length checks
- Bayesian Signal Combiner using log-odds form
- Tier 1: SPC baselines, session consistency, cross-session consistency
- Adaptive thresholds that learn from user feedback

Quick start::

    from agentguard.verification import VerificationEngine

    engine = VerificationEngine()
    engine.register_tool_profile(
        "get_weather",
        expected_latency_ms=(100, 5000),
        required_fields=["temperature", "humidity"],
        has_network_io=True,
    )

    result = engine.verify(
        tool_name="get_weather",
        result={"temperature": 18, "humidity": 65},
        execution_time_ms=350.0,
        user_query="What's the weather in London?",
    )
    # result.verdict: "accept" | "flag" | "block"
    # result.confidence: 0.0–1.0 (P(hallucination))
"""

from __future__ import annotations

from agentguard.verification.engine import (
    GLOBAL_LIKELIHOOD_RATIOS,
    SignalResult,
    ToolProfile,
    VerificationEngine,
    VerificationResult,
    VerificationTier,
)
from agentguard.verification.baselines import (
    RunningStats,
    SPCAnomaly,
    SPCResult,
    ToolBaseline,
)
from agentguard.verification.consistency import (
    ConsistencyResult,
    ConsistencyTracker,
    ConsistencyViolation,
)
from agentguard.verification.adaptive import (
    AdaptiveThresholdManager,
    ThresholdStats,
)
from agentguard.verification.signals import (
    check_latency_anomaly,
    check_response_length,
    check_response_patterns,
    check_schema_compliance,
    check_session_consistency,
    check_value_plausibility,
)
from agentguard.verification.embeddings import (
    check_semantic_similarity,
    cosine_similarity,
    embed,
    is_available as embeddings_available,
)

__all__ = [
    # Engine
    "VerificationEngine",
    "VerificationResult",
    "VerificationTier",
    "SignalResult",
    "ToolProfile",
    "GLOBAL_LIKELIHOOD_RATIOS",
    # Baselines
    "ToolBaseline",
    "RunningStats",
    "SPCAnomaly",
    "SPCResult",
    # Consistency
    "ConsistencyTracker",
    "ConsistencyResult",
    "ConsistencyViolation",
    # Adaptive
    "AdaptiveThresholdManager",
    "ThresholdStats",
    # Signal functions
    "check_latency_anomaly",
    "check_schema_compliance",
    "check_response_patterns",
    "check_response_length",
    "check_value_plausibility",
    "check_session_consistency",
    # Embeddings
    "check_semantic_similarity",
    "cosine_similarity",
    "embed",
    "embeddings_available",
]
