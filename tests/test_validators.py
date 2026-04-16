"""Tests for validators: schema, hallucination, semantic, custom."""

from __future__ import annotations

from typing import Optional

import pytest

from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind
from agentguard.validators.custom import (
    CustomValidator,
    no_empty_string_args,
    run_custom_validators,
    validator_fn,
)
from agentguard.validators.hallucination import (
    HallucinationDetector,
    _fields_score,
    _latency_score,
    _pattern_score,
)
from agentguard.validators.schema import validate_inputs, validate_output
from agentguard.validators.semantic import (
    SemanticValidator,
    check_key_present,
    check_no_error_field,
    check_non_empty,
    check_status_ok,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_valid_str_input(self):
        def fn(name: str) -> None: ...
        results = validate_inputs(fn, ("alice",), {})
        assert all(r.valid for r in results)

    def test_invalid_type(self):
        def fn(count: int) -> None: ...
        results = validate_inputs(fn, ("not_an_int",), {})
        assert any(not r.valid for r in results)

    def test_optional_accepts_none(self):
        def fn(name: Optional[str] = None) -> None: ...
        results = validate_inputs(fn, (None,), {})
        assert all(r.valid for r in results)

    def test_list_type(self):
        def fn(items: list[str]) -> None: ...
        results = validate_inputs(fn, (["a", "b"],), {})
        assert all(r.valid for r in results)

    def test_dict_type(self):
        def fn(data: dict) -> None: ...
        results = validate_inputs(fn, ({"key": "val"},), {})
        assert all(r.valid for r in results)

    def test_output_validation_passes(self):
        def fn() -> dict: ...
        results = validate_output(fn, {"a": 1})
        assert all(r.valid for r in results)

    def test_output_validation_fails(self):
        def fn() -> dict: ...
        results = validate_output(fn, "not_a_dict")
        assert any(not r.valid for r in results)

    def test_no_annotations_passes_all(self):
        def fn(x, y): ...
        results = validate_inputs(fn, (1, 2), {})
        assert len(results) == 0  # No hints → nothing to validate

    def test_binding_failure(self):
        def fn(required: str) -> None: ...
        results = validate_inputs(fn, (), {})  # Missing required arg
        assert any(not r.valid for r in results)


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


class TestHallucinationDetector:
    def test_impossibly_fast_is_hallucinated(self):
        detector = HallucinationDetector()
        result = detector.verify("any_tool", execution_time_ms=0.1, response={})
        assert result.is_hallucinated
        assert result.confidence > 0.5

    def test_normal_speed_not_hallucinated(self):
        detector = HallucinationDetector()
        detector.register_tool(
            "weather_api",
            expected_latency_ms=(100, 5000),
            required_fields=["temperature"],
        )
        result = detector.verify(
            "weather_api",
            execution_time_ms=250,
            response={"temperature": 72, "humidity": 50},
        )
        assert not result.is_hallucinated

    def test_missing_required_fields(self):
        detector = HallucinationDetector(threshold=0.3)
        detector.register_tool(
            "db_tool",
            expected_latency_ms=(10, 5000),
            required_fields=["rows", "count"],
            fields_weight=1.0,
            latency_weight=0.0,
            patterns_weight=0.0,
        )
        result = detector.verify(
            "db_tool",
            execution_time_ms=50,
            response={"error": "not found"},
        )
        assert result.is_hallucinated

    def test_pattern_mismatch(self):
        detector = HallucinationDetector(threshold=0.5)
        detector.register_tool(
            "api",
            expected_latency_ms=(50, 5000),
            response_patterns=[r'"id":\s*\d+'],
            patterns_weight=1.0,
            latency_weight=0.0,
            fields_weight=0.0,
        )
        result = detector.verify(
            "api",
            execution_time_ms=200,
            response={"result": "no id field here"},
        )
        assert result.is_hallucinated

    def test_no_profile_returns_not_hallucinated(self):
        detector = HallucinationDetector()
        result = detector.verify("unknown_tool", execution_time_ms=500, response={})
        assert not result.is_hallucinated

    def test_call_stack_verified_bypasses_latency(self):
        detector = HallucinationDetector()
        detector.register_tool("fast_tool", expected_latency_ms=(100, 5000))
        result = detector.verify(
            "fast_tool",
            execution_time_ms=0.5,
            response={"ok": True},
            call_stack_verified=True,
        )
        # With call_stack_verified, latency signal is 0
        # Should have lower confidence
        assert result.confidence < 0.5

    def test_register_returns_self_for_chaining(self):
        d = HallucinationDetector()
        result = d.register_tool("a").register_tool("b")
        assert result is d

    def test_latency_score_helpers(self):
        assert _latency_score(0.5, (100, 5000)) > 0.5  # Very fast → suspicious
        assert _latency_score(500, (100, 5000)) == 0.0  # Normal → clean
        assert _latency_score(100000, (100, 5000)) > 0.0  # Very slow → slight flag

    def test_fields_score_helpers(self):
        score, _ = _fields_score({"temperature": 72}, ["temperature"], [])
        assert score == 0.0  # All required fields present

        score, msg = _fields_score({"other": 1}, ["temperature"], [])
        assert score > 0.0  # Missing required field

    def test_pattern_score_helpers(self):
        score, _ = _pattern_score({"temp": 72}, [r'"temp":\s*\d+'])
        assert score == 0.0  # Matches

        score, _ = _pattern_score({"other": "data"}, [r'"temp":\s*\d+'])
        assert score > 0.0  # No match


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


class TestSemanticValidation:
    def test_non_empty_check(self):
        assert check_non_empty({"results": [1, 2]}, field="results") is None
        assert check_non_empty({"results": []}, field="results") is not None

    def test_key_present_check(self):
        assert check_key_present({"a": 1, "b": 2}, keys=["a", "b"]) is None
        assert check_key_present({"a": 1}, keys=["a", "b"]) is not None

    def test_no_error_field(self):
        assert check_no_error_field({"data": 1}) is None
        assert check_no_error_field({"error": "something went wrong"}) is not None
        assert check_no_error_field({"error": None}) is None  # error=None is falsy

    def test_status_ok_check(self):
        assert check_status_ok({"status": 200}) is None
        assert check_status_ok({"status": 500}) is not None
        assert check_status_ok({"status_code": "ok"}) is None

    def test_semantic_validator_registration(self):
        sv = SemanticValidator()

        @sv.validator("test_tool")
        def check_result(query: str, result: dict) -> Optional[str]:
            if not result.get("data"):
                return "No data in result"
            return None

        results = sv.validate("test_tool", args=("my query",), kwargs={}, result={"data": [1, 2]})
        assert all(r.valid for r in results)

    def test_semantic_validator_failure(self):
        sv = SemanticValidator()

        @sv.validator("test_tool")
        def always_fails(query: str, result: dict) -> Optional[str]:
            return "always fails"

        results = sv.validate("test_tool", args=("q",), kwargs={}, result={})
        assert any(not r.valid for r in results)

    def test_no_validators_returns_empty(self):
        sv = SemanticValidator()
        results = sv.validate("unregistered", args=(), kwargs={}, result={})
        assert results == []


# ---------------------------------------------------------------------------
# Custom validators
# ---------------------------------------------------------------------------


class TestCustomValidators:
    def test_validator_fn_decorator(self):
        @validator_fn(name="test_v")
        def my_validator(call: ToolCall, result: object = None) -> ValidationResult:
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        assert hasattr(my_validator, "_agentguard_validator")
        assert my_validator._agentguard_name == "test_v"

    def test_custom_validator_class(self):
        def check(call: ToolCall, result: object = None) -> ValidationResult:
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        cv = CustomValidator("my_check", check)
        call = ToolCall(tool_name="test")
        vr = cv.validate(call)
        assert vr.valid

    def test_custom_validator_tool_filter(self):
        """Validator scoped to specific tool should skip other tools."""
        def check(call: ToolCall, result: object = None) -> ValidationResult:
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        cv = CustomValidator("scoped", check, apply_to=["allowed_tool"])
        wrong_call = ToolCall(tool_name="other_tool")
        vr = cv.validate(wrong_call)
        assert vr.valid  # Skipped, not failed

    def test_no_empty_string_args_validator(self):
        call = ToolCall(tool_name="t", kwargs={"query": ""})
        vr = no_empty_string_args(call)
        assert not vr.valid
        assert "query" in vr.message

    def test_run_custom_validators_list(self):
        def always_pass(call: ToolCall, result: object = None) -> ValidationResult:
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        call = ToolCall(tool_name="t")
        results = run_custom_validators([always_pass], call)
        assert len(results) == 1
        assert results[0].valid
