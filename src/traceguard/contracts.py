"""Typed contracts for tools, fault configurations, scenarios, and trajectory telemetry."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PermissionLevel = Literal["read", "write", "restricted"]
FaultKind = Literal[
    "http_503",
    "timeout",
    "malformed_json",
    "prompt_injection",
    "partial_truncate",
    "rate_limit",
]
EventType = Literal[
    "plan",
    "tool_call",
    "tool_result",
    "fault_injected",
    "schema_error",
    "retry_backoff",
    "sanitizer_catch",
    "rbac_block",
    "cycle_break",
    "final_response",
]
EventStatus = Literal["ok", "fault", "blocked", "sanitized", "retried", "aborted"]
ScenarioCategory = Literal[
    "nominal",
    "transient_fault",
    "persistent_fault",
    "schema_edge",
    "prompt_injection",
    "privilege_escalation",
    "loop_trap",
]
OutcomeKind = Literal[
    "resolved",
    "escalated",
    "aborted_safe",
    "failed",
    "compromised",
    "loop_exceeded",
]
ExpectedOutcome = Literal["resolved", "escalated", "aborted_safe"]


class ParamConstraint(BaseModel):
    """JSON schema-style parameter constraint for a sandbox tool argument."""

    name: str
    param_type: Literal["string", "number", "integer", "boolean"]
    required: bool = True
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list[str]] = None
    min_length: Optional[int] = None
    default: Optional[Any] = None

    def validate_value(self, value: Any) -> list[str]:
        """Validate a single argument value against this constraint."""
        errors: list[str] = []

        if self.param_type == "string":
            if not isinstance(value, str):
                return [f"Parameter '{self.name}' must be a string, got {type(value).__name__}."]
            if self.min_length is not None and len(value.strip()) < self.min_length:
                errors.append(
                    f"Parameter '{self.name}' must have length >= {self.min_length}, got {len(value.strip())}."
                )
            if self.pattern is not None and not re.match(self.pattern, value):
                errors.append(
                    f"Parameter '{self.name}' value '{value}' failed pattern '{self.pattern}'."
                )
            if self.allowed_values is not None and value not in self.allowed_values:
                errors.append(
                    f"Parameter '{self.name}' value '{value}' must be one of {self.allowed_values}."
                )
        elif self.param_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"Parameter '{self.name}' must be a boolean, got {type(value).__name__}.")
        elif self.param_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return [f"Parameter '{self.name}' must be an integer, got {type(value).__name__}."]
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Parameter '{self.name}'={value} is below min_value={self.min_value}.")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Parameter '{self.name}'={value} exceeds max_value={self.max_value}.")
        elif self.param_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return [f"Parameter '{self.name}' must be numeric, got {type(value).__name__}."]
            if self.min_value is not None and float(value) < self.min_value:
                errors.append(f"Parameter '{self.name}'={value} is below min_value={self.min_value}.")
            if self.max_value is not None and float(value) > self.max_value:
                errors.append(f"Parameter '{self.name}'={value} exceeds max_value={self.max_value}.")

        return errors


class ToolSpec(BaseModel):
    """Typed tool contract with parameter validation and capability metadata."""

    name: str
    description: str
    permission_level: PermissionLevel
    params: list[ParamConstraint] = Field(default_factory=list)
    requires_verification: bool = False
    max_safe_amount: Optional[float] = None
    untrusted_output: bool = False

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Validate tool arguments strictly against the declared constraints."""
        if not isinstance(args, dict):
            return [f"Arguments for '{self.name}' must be a JSON object."]

        errors: list[str] = []
        declared = {p.name: p for p in self.params}

        for unknown_key in args:
            if unknown_key not in declared:
                errors.append(f"Unexpected parameter '{unknown_key}' for tool '{self.name}'.")

        for param in self.params:
            if param.name not in args or args[param.name] is None:
                if param.required:
                    errors.append(f"Missing required parameter '{param.name}' for tool '{self.name}'.")
                continue
            errors.extend(param.validate_value(args[param.name]))

        return errors


class FaultConfig(BaseModel):
    """Composable fault or adversarial injection rule attached to a scenario."""

    target_tool: str
    fault_type: FaultKind
    trigger_on_attempts: list[int] = Field(default_factory=lambda: [1])
    persistent: bool = False
    requires_backoff_ms: float = 0.0
    injection_payload: Optional[str] = None
    injected_tool_call: Optional[dict[str, Any]] = None
    latency_ms: float = 45.0
    description: str = ""


class ScenarioTask(BaseModel):
    """Multi-step evaluation task executed against the deterministic sandbox."""

    scenario_id: str
    title: str
    category: ScenarioCategory
    user_prompt: str
    customer_id: Optional[str] = None
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    requested_refund_amount: Optional[float] = None
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_outcome: ExpectedOutcome = "resolved"
    faults: list[FaultConfig] = Field(default_factory=list)
    requires_schema_coercion: bool = False
    dirty_order_id: Optional[str] = None
    dirty_amount: Optional[Any] = None
    dirty_reason: Optional[str] = None
    max_steps: int = 12
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SandboxToolResponse(BaseModel):
    """Result returned by the sandbox for a single tool invocation."""

    ok: bool
    tool_name: str
    attempt: int
    status_code: int = 200
    permission_level: PermissionLevel
    untrusted_output: bool = False
    error_type: Optional[str] = None
    retryable: bool = False
    fault_type: Optional[FaultKind] = None
    injection_present: bool = False
    raw_text: str
    data: Optional[dict[str, Any]] = None
    latency_ms: float = 18.0


class TraceEvent(BaseModel):
    """Single step event recorded in a trajectory trace for replay and audit."""

    step: int
    timestamp_ms: float
    event_type: EventType
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    permission_level: Optional[PermissionLevel] = None
    status: EventStatus = "ok"
    fault_type: Optional[FaultKind] = None
    raw_payload: Optional[str] = None
    sanitized_payload: Optional[str] = None
    detail: str
    latency_ms: float = 0.0
    tokens: int = 0


class TrajectoryResult(BaseModel):
    """Complete multi-step trajectory telemetry and outcome classification."""

    scenario_id: str
    scenario_title: str
    category: ScenarioCategory
    policy_name: str
    task_success: bool
    schema_compliant: bool
    fault_recovered: bool
    had_transient_fault: bool
    injection_attempted: bool
    injection_defended: bool
    loop_free: bool
    escalated_safely: bool = False
    final_outcome: OutcomeKind
    final_answer: str
    total_steps: int
    total_latency_ms: float
    total_tokens: int
    schema_violations: int = 0
    retries_used: int = 0
    sanitizer_catches: int = 0
    rbac_blocks: int = 0
    tools_executed: list[str] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)


class PolicyBenchmarkRow(BaseModel):
    """Aggregated evaluation leaderboard row for one agent policy."""

    policy_name: str
    total_runs: int
    task_success_rate: float
    schema_compliance_rate: float
    fault_recovery_rate: float
    injection_defense_rate: float
    loop_free_rate: float
    avg_latency_ms: float
    avg_tokens: float
    composite_score: float


class FaultMix(BaseModel):
    """Configurable fault and injection distribution for seeded fuzz campaigns."""

    runs: int = Field(default=30, ge=1, le=500)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    injection_prob: float = Field(default=0.35, ge=0.0, le=1.0)
    transient_fault_prob: float = Field(default=0.40, ge=0.0, le=1.0)
    persistent_fault_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    malformed_json_prob: float = Field(default=0.20, ge=0.0, le=1.0)
    dirty_arg_prob: float = Field(default=0.25, ge=0.0, le=1.0)
