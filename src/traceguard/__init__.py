"""TraceGuard Eval — Multi-step AI agent trajectory simulator, fuzzer, and replay studio."""

from traceguard.agents import (
    AgentPolicy,
    CapabilityRBACGuard,
    DeterministicPlannerBackend,
    NaiveAgent,
    PromptInjectionSanitizer,
    SchemaRetryAgent,
    ToolCallingBackend,
    TraceGuardHardenedAgent,
    get_builtin_policies,
)
from traceguard.contracts import (
    FaultConfig,
    FaultMix,
    ParamConstraint,
    PolicyBenchmarkRow,
    ScenarioTask,
    ToolSpec,
    TraceEvent,
    TrajectoryResult,
)
from traceguard.evaluator import (
    BUILTIN_SCENARIOS,
    compute_policy_row,
    evaluate_policies,
    format_leaderboard_markdown,
    get_builtin_scenarios,
    render_html_report,
)
from traceguard.fuzzer import TraceFuzzer
from traceguard.sandbox import TOOL_REGISTRY, ExecutionSandbox

__version__ = "0.1.0"

__all__ = [
    "AgentPolicy",
    "BUILTIN_SCENARIOS",
    "CapabilityRBACGuard",
    "DeterministicPlannerBackend",
    "ExecutionSandbox",
    "FaultConfig",
    "FaultMix",
    "NaiveAgent",
    "ParamConstraint",
    "PolicyBenchmarkRow",
    "PromptInjectionSanitizer",
    "ScenarioTask",
    "SchemaRetryAgent",
    "TOOL_REGISTRY",
    "ToolCallingBackend",
    "ToolSpec",
    "TraceEvent",
    "TraceFuzzer",
    "TraceGuardHardenedAgent",
    "TrajectoryResult",
    "__version__",
    "compute_policy_row",
    "evaluate_policies",
    "format_leaderboard_markdown",
    "get_builtin_policies",
    "get_builtin_scenarios",
    "render_html_report",
]
