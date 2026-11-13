"""FastAPI web server and interactive dark-mode Trace Replay & Agent Reliability Studio UI."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from traceguard.agents import get_builtin_policies
from traceguard.contracts import FaultMix, PolicyBenchmarkRow, TrajectoryResult
from traceguard.evaluator import (
    evaluate_policies,
    get_builtin_scenarios,
    render_html_report,
)
from traceguard.fuzzer import TraceFuzzer
from traceguard.sandbox import TOOL_REGISTRY

app = FastAPI(
    title="traceguard-eval",
    description="Multi-Step AI Agent Trajectory Simulator, Prompt-Injection & Fault Fuzzer, and Trace Replay Studio",
    version="0.1.0",
)


class BenchmarkRequest(BaseModel):
    """Request payload for running the built-in evaluation suite."""

    policies: Optional[list[str]] = None
    scenario_ids: Optional[list[str]] = None


class FuzzRequest(FaultMix):
    """Request payload for running a seeded adversarial fuzz campaign."""

    policies: Optional[list[str]] = None


class ExportHtmlRequest(BaseModel):
    """Request payload for rendering a standalone HTML trajectory audit report."""

    title: str = "TraceGuard Trajectory Audit Report"
    mode: Literal["benchmark", "fuzz", "custom"] = "benchmark"
    policies: Optional[list[str]] = None
    fault_mix: Optional[FaultMix] = None
    leaderboard: Optional[list[PolicyBenchmarkRow]] = None
    trajectories: Optional[list[TrajectoryResult]] = None


@app.get("/", response_class=HTMLResponse)
def studio_ui() -> HTMLResponse:
    """Serve the interactive dark-mode Trace Replay & Agent Reliability Studio."""
    return HTMLResponse(content=_STUDIO_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service readiness, scenario count, and registered tools/policies."""
    scenarios = get_builtin_scenarios()
    policies = get_builtin_policies()
    return {
        "status": "ok",
        "service": "traceguard-eval",
        "version": "0.1.0",
        "scenario_count": len(scenarios),
        "tool_count": len(TOOL_REGISTRY),
        "policies": list(policies.keys()),
    }


@app.get("/api/scenarios")
def list_scenarios() -> dict[str, Any]:
    """Return the built-in 12-scenario evaluation catalog, tool specs, and policy descriptions."""
    scenarios = get_builtin_scenarios()
    policies = get_builtin_policies()
    return {
        "scenario_count": len(scenarios),
        "scenarios": [s.model_dump() for s in scenarios],
        "tools": [spec.model_dump() for spec in TOOL_REGISTRY.values()],
        "policies": [
            {"name": p.name, "description": p.description} for p in policies.values()
        ],
    }


@app.post("/api/benchmark")
def run_benchmark(payload: BenchmarkRequest = Body(default_factory=BenchmarkRequest)) -> dict[str, Any]:
    """Execute the multi-scenario benchmark across selected agent policies."""
    scenarios = get_builtin_scenarios()
    if payload.scenario_ids:
        wanted = set(payload.scenario_ids)
        scenarios = [s for s in scenarios if s.scenario_id in wanted]
        if not scenarios:
            raise HTTPException(status_code=400, detail="No matching scenario_ids found.")

    try:
        leaderboard, trajectories = evaluate_policies(scenarios, payload.policies)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "mode": "benchmark",
        "scenario_count": len(scenarios),
        "leaderboard": [row.model_dump() for row in leaderboard],
        "trajectories": [tr.model_dump() for tr in trajectories],
    }


@app.post("/api/fuzz")
def run_fuzz(payload: FuzzRequest = Body(default_factory=FuzzRequest)) -> dict[str, Any]:
    """Run a seeded adversarial fault & prompt-injection fuzz campaign."""
    fault_mix = FaultMix(
        runs=payload.runs,
        seed=payload.seed,
        injection_prob=payload.injection_prob,
        transient_fault_prob=payload.transient_fault_prob,
        persistent_fault_prob=payload.persistent_fault_prob,
        malformed_json_prob=payload.malformed_json_prob,
        dirty_arg_prob=payload.dirty_arg_prob,
    )
    fuzzer = TraceFuzzer(fault_mix)
    tasks = fuzzer.generate()
    breakdown = TraceFuzzer.summarize_mix(tasks)

    try:
        leaderboard, trajectories = evaluate_policies(tasks, payload.policies)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "mode": "fuzz",
        "fault_mix": fault_mix.model_dump(),
        "fuzz_breakdown": breakdown,
        "leaderboard": [row.model_dump() for row in leaderboard],
        "trajectories": [tr.model_dump() for tr in trajectories],
    }


@app.post("/api/export-html")
def export_html_report(payload: ExportHtmlRequest = Body(default_factory=ExportHtmlRequest)) -> HTMLResponse:
    """Render a standalone self-contained HTML trajectory audit report."""
    meta: dict[str, Any] = {"mode": payload.mode}

    if payload.leaderboard is not None and payload.trajectories is not None:
        leaderboard = payload.leaderboard
        trajectories = payload.trajectories
    elif payload.mode == "fuzz":
        mix = payload.fault_mix or FaultMix()
        tasks = TraceFuzzer(mix).generate()
        meta.update({"seed": mix.seed, "runs": mix.runs})
        try:
            leaderboard, trajectories = evaluate_policies(tasks, payload.policies)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        scenarios = get_builtin_scenarios()
        meta["scenarios"] = len(scenarios)
        try:
            leaderboard, trajectories = evaluate_policies(scenarios, payload.policies)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    html_doc = render_html_report(
        title=payload.title,
        leaderboard=leaderboard,
        trajectories=trajectories,
        metadata=meta,
    )
    return HTMLResponse(
        content=html_doc,
        headers={"Content-Disposition": 'attachment; filename="traceguard-audit-report.html"'},
    )


_STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TraceGuard Eval · Trace Replay &amp; Agent Reliability Studio</title>
<style>
  :root {
    --bg: #070b14;
    --bg-elev: #0e1525;
    --bg-card: #121b2f;
    --bg-soft: #18233c;
    --border: #253456;
    --text: #e6eeff;
    --muted: #8ea0c8;
    --cyan: #38bdf8;
    --indigo: #818cf8;
    --emerald: #22c55e;
    --amber: #f59e0b;
    --rose: #ef4444;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at top right, #111c36 0%, var(--bg) 52%);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.45;
  }
  .shell { max-width: 1440px; margin: 0 auto; padding: 24px; }
  .topbar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 16px; padding-bottom: 20px; border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 14px; }
  .shield {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    display: grid; place-items: center; font-weight: 800; font-size: 18px;
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25);
  }
  .brand h1 { margin: 0; font-size: 22px; letter-spacing: -0.02em; }
  .brand p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
  .status-group { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px;
    border-radius: 999px; font-size: 12px; font-weight: 600;
    background: var(--bg-card); border: 1px solid var(--border); color: var(--muted);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 8px var(--emerald); }
  .btn {
    border: 1px solid var(--border); background: var(--bg-soft); color: var(--text);
    padding: 9px 15px; border-radius: 10px; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all .15s ease;
  }
  .btn:hover { border-color: var(--cyan); transform: translateY(-1px); }
  .btn-primary {
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    border-color: transparent; box-shadow: 0 6px 18px rgba(14, 165, 233, 0.25);
  }
  .btn-emerald {
    background: linear-gradient(135deg, #059669, #0284c7);
    border-color: transparent;
  }
  .btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

  .controls-grid {
    display: grid; grid-template-columns: 1fr 1.55fr; gap: 16px; margin-top: 20px;
  }
  @media (max-width: 1020px) { .controls-grid { grid-template-columns: 1fr; } }

  .card {
    background: rgba(18, 27, 47, 0.9); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px; backdrop-filter: blur(8px);
  }
  .card-title {
    margin: 0 0 12px; font-size: 14px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--cyan); display: flex; justify-content: space-between; align-items: center;
  }
  .policy-checks { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
  .check-pill {
    display: inline-flex; align-items: center; gap: 7px; padding: 7px 12px;
    border-radius: 10px; background: var(--bg-elev); border: 1px solid var(--border);
    font-size: 13px; cursor: pointer;
  }
  .fuzz-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
    margin-bottom: 14px;
  }
  .field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .field input[type="number"], .field select {
    width: 100%; background: var(--bg-elev); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 8px; font-size: 13px;
  }
  .field input[type="range"] { width: 100%; accent-color: var(--cyan); }

  .kpi-row {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 18px;
  }
  @media (max-width: 860px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
  .kpi {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .kpi-label { font-size: 12px; color: var(--muted); }
  .kpi-val { font-size: 26px; font-weight: 800; margin-top: 4px; letter-spacing: -0.02em; }
  .kpi-sub { font-size: 12px; color: var(--cyan); margin-top: 2px; }

  .analytics-grid {
    display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; margin-top: 18px;
  }
  @media (max-width: 1120px) { .analytics-grid { grid-template-columns: 1fr; } }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 10px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
  .bar-group { margin-bottom: 12px; }
  .bar-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
  .bar-track { height: 8px; background: var(--bg-elev); border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width .35s ease; }

  .workspace {
    display: grid; grid-template-columns: 1.05fr 1.25fr; gap: 16px; margin-top: 18px;
  }
  @media (max-width: 1080px) { .workspace { grid-template-columns: 1fr; } }

  .filter-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .filter-bar select {
    background: var(--bg-elev); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; font-size: 12px;
  }
  .trace-list { max-height: 620px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding-right: 4px; }
  .trace-row {
    background: var(--bg-elev); border: 1px solid var(--border); border-radius: 10px;
    padding: 11px 13px; cursor: pointer; transition: border-color .15s, background .15s;
  }
  .trace-row:hover { border-color: var(--cyan); }
  .trace-row.active { border-color: var(--cyan); background: #16233f; box-shadow: 0 0 0 1px rgba(56,189,248,0.25); }
  .trace-row-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .trace-row-sub { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-top: 5px; }

  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700;
    text-transform: uppercase;
  }
  .b-resolved { background: rgba(34, 197, 94, 0.16); color: #4ade80; }
  .b-escalated { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; }
  .b-compromised { background: rgba(239, 68, 68, 0.18); color: #f87171; }
  .b-loop_exceeded, .b-failed { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
  .b-read { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; }
  .b-write { background: rgba(245, 158, 11, 0.16); color: #fcd34d; }
  .b-restricted { background: rgba(239, 68, 68, 0.18); color: #fca5a5; }

  .timeline { max-height: 620px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding-right: 4px; }
  .ev-card {
    border-left: 3px solid var(--border); background: var(--bg-elev);
    border-radius: 10px; padding: 11px 13px;
  }
  .ev-card.status-ok { border-left-color: var(--emerald); }
  .ev-card.status-fault { border-left-color: var(--rose); }
  .ev-card.status-retried { border-left-color: var(--indigo); }
  .ev-card.status-sanitized { border-left-color: var(--amber); }
  .ev-card.status-blocked { border-left-color: var(--cyan); }
  .ev-card.status-aborted { border-left-color: var(--rose); }

  .ev-meta { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
  .ev-title { font-weight: 700; color: var(--text); font-size: 13px; margin: 5px 0; }
  .codebox {
    margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: #070b14;
    border: 1px solid #1e2942; font-family: var(--mono); font-size: 11.5px;
    white-space: pre-wrap; word-break: break-word; color: #cbd5e1;
  }
  .codebox.raw { border-color: rgba(239, 68, 68, 0.45); color: #fecaca; }
  .codebox.san { border-color: rgba(34, 197, 94, 0.45); color: #bbf7d0; }
  .cost-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
  .mini-bar-label { font-size: 11px; color: var(--muted); display: flex; justify-content: space-between; margin-bottom: 3px; }
  .mini-track { height: 5px; background: #070b14; border-radius: 999px; overflow: hidden; }
  .mini-fill-lat { height: 100%; background: var(--cyan); }
  .mini-fill-tok { height: 100%; background: var(--indigo); }
  .verdict-strip { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="shield">TG</div>
      <div>
        <h1>TraceGuard Eval · Trace Replay &amp; Agent Reliability Studio</h1>
        <p>Multi-Step Trajectory Simulator · Indirect Prompt-Injection &amp; Fault Fuzzer · Capability RBAC Audit</p>
      </div>
    </div>
    <div class="status-group">
      <span class="pill"><span class="dot"></span><span id="healthText">Loading sandbox...</span></span>
      <button class="btn" id="btnExport" onclick="exportHtmlReport()">Export HTML Audit Report</button>
    </div>
  </header>

  <section class="controls-grid">
    <div class="card">
      <div class="card-title"><span>12-Scenario Evaluation Suite</span><span class="pill" id="modeBadge">Benchmark Mode</span></div>
      <div class="policy-checks">
        <label class="check-pill"><input type="checkbox" class="policy-cb" value="NaiveAgent" checked /> NaiveAgent</label>
        <label class="check-pill"><input type="checkbox" class="policy-cb" value="SchemaRetryAgent" checked /> SchemaRetryAgent</label>
        <label class="check-pill"><input type="checkbox" class="policy-cb" value="TraceGuardHardenedAgent" checked /> TraceGuardHardenedAgent</label>
      </div>
      <p style="font-size:13px; color:var(--muted); margin:0 0 14px;">
        Evaluates nominal support workflows, dirty JSON coercion, transient 503/429/timeout/truncation recovery,
        persistent loop traps, and indirect prompt-injection attacks across all selected policies.
      </p>
      <button class="btn btn-primary" id="btnBenchmark" onclick="runBenchmark()">Run 12-Scenario Suite</button>
    </div>

    <div class="card">
      <div class="card-title"><span>Adversarial Fault &amp; Injection Fuzzer</span><span id="fuzzSummaryPill" class="pill">Seeded Deterministic</span></div>
      <div class="fuzz-grid">
        <div class="field">
          <label><span>Runs</span><span id="lblRuns">20</span></label>
          <input type="number" id="fuzzRuns" min="1" max="200" value="20" oninput="document.getElementById('lblRuns').textContent=this.value" />
        </div>
        <div class="field">
          <label><span>Seed</span></label>
          <input type="number" id="fuzzSeed" min="0" value="42" />
        </div>
        <div class="field">
          <label><span>Injection Prob</span><span id="lblInj">35%</span></label>
          <input type="range" id="fuzzInj" min="0" max="100" value="35" oninput="document.getElementById('lblInj').textContent=this.value+'%'" />
        </div>
        <div class="field">
          <label><span>Transient Faults</span><span id="lblTrans">40%</span></label>
          <input type="range" id="fuzzTrans" min="0" max="100" value="40" oninput="document.getElementById('lblTrans').textContent=this.value+'%'" />
        </div>
        <div class="field">
          <label><span>Persistent Outage</span><span id="lblPers">15%</span></label>
          <input type="range" id="fuzzPers" min="0" max="100" value="15" oninput="document.getElementById('lblPers').textContent=this.value+'%'" />
        </div>
        <div class="field">
          <label><span>Malformed JSON</span><span id="lblMal">20%</span></label>
          <input type="range" id="fuzzMal" min="0" max="100" value="20" oninput="document.getElementById('lblMal').textContent=this.value+'%'" />
        </div>
        <div class="field">
          <label><span>Dirty Args</span><span id="lblDirty">25%</span></label>
          <input type="range" id="fuzzDirty" min="0" max="100" value="25" oninput="document.getElementById('lblDirty').textContent=this.value+'%'" />
        </div>
      </div>
      <button class="btn btn-emerald" id="btnFuzz" onclick="runFuzz()">Run Seeded Fuzz Campaign</button>
    </div>
  </section>

  <section class="kpi-row">
    <div class="kpi"><div class="kpi-label">Top Policy</div><div class="kpi-val" id="kpiTopPolicy">—</div><div class="kpi-sub" id="kpiTopScore">Composite —</div></div>
    <div class="kpi"><div class="kpi-label">Hardened Injection Defense</div><div class="kpi-val" id="kpiDefense">—</div><div class="kpi-sub">vs Baseline Vulnerability</div></div>
    <div class="kpi"><div class="kpi-label">Hardened Fault Recovery</div><div class="kpi-val" id="kpiRecovery">—</div><div class="kpi-sub">503 / 429 / Timeout / JSON</div></div>
    <div class="kpi"><div class="kpi-label">Trajectories Captured</div><div class="kpi-val" id="kpiTotalRuns">—</div><div class="kpi-sub" id="kpiModeNote">Click any trace to replay</div></div>
  </section>

  <section class="analytics-grid">
    <div class="card">
      <div class="card-title"><span>Policy Reliability Leaderboard</span></div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Policy</th><th>Runs</th><th>Success</th><th>Schema</th><th>Recovery</th>
              <th>Inj Defense</th><th>Loop-Free</th><th>Latency</th><th>Tokens</th><th>Composite</th>
            </tr>
          </thead>
          <tbody id="leaderboardBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><span>Composite &amp; Safety Comparison</span></div>
      <div id="comparisonBars"></div>
    </div>
  </section>

  <section class="workspace">
    <div class="card">
      <div class="card-title"><span>Trajectory Explorer</span><span class="pill" id="filteredCount">0 traces</span></div>
      <div class="filter-bar">
        <select id="filterPolicy" onchange="renderTraceList()"><option value="ALL">All Policies</option></select>
        <select id="filterOutcome" onchange="renderTraceList()">
          <option value="ALL">All Outcomes</option>
          <option value="resolved">resolved</option>
          <option value="escalated">escalated</option>
          <option value="compromised">compromised</option>
          <option value="loop_exceeded">loop_exceeded</option>
          <option value="failed">failed</option>
        </select>
        <select id="filterCategory" onchange="renderTraceList()"><option value="ALL">All Categories</option></select>
      </div>
      <div class="trace-list" id="traceList"></div>
    </div>

    <div class="card">
      <div class="card-title"><span>Step-by-Step Timeline Replay</span><span class="pill" id="activeTraceLabel">Select a trace</span></div>
      <div id="traceInspector">
        <p style="color:var(--muted); font-size:13px;">Select any trajectory on the left to inspect tool arguments, injected fault payloads, sanitizer redactions, RBAC blocks, and per-step latency/token bars.</p>
      </div>
    </div>
  </section>
</div>

<script>
let currentPayload = null;
let selectedIdx = 0;

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getSelectedPolicies() {
  const checked = Array.from(document.querySelectorAll(".policy-cb:checked")).map(el => el.value);
  return checked.length ? checked : ["NaiveAgent", "SchemaRetryAgent", "TraceGuardHardenedAgent"];
}

function setBusy(busy) {
  document.getElementById("btnBenchmark").disabled = busy;
  document.getElementById("btnFuzz").disabled = busy;
  document.getElementById("btnExport").disabled = busy;
}

async function init() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    document.getElementById("healthText").textContent =
      `Online · ${h.scenario_count} Scenarios · ${h.tool_count} Sandboxed Tools`;
  } catch (err) {
    document.getElementById("healthText").textContent = "Health check unavailable";
  }
  await runBenchmark();
}

async function runBenchmark() {
  setBusy(true);
  try {
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policies: getSelectedPolicies() }),
    });
    currentPayload = await res.json();
    document.getElementById("modeBadge").textContent = "Suite Mode (12 Scenarios)";
    document.getElementById("fuzzSummaryPill").textContent = "Built-in Suite";
    renderDashboard();
  } finally {
    setBusy(false);
  }
}

async function runFuzz() {
  setBusy(true);
  try {
    const body = {
      runs: Number(document.getElementById("fuzzRuns").value) || 20,
      seed: Number(document.getElementById("fuzzSeed").value) || 42,
      injection_prob: Number(document.getElementById("fuzzInj").value) / 100.0,
      transient_fault_prob: Number(document.getElementById("fuzzTrans").value) / 100.0,
      persistent_fault_prob: Number(document.getElementById("fuzzPers").value) / 100.0,
      malformed_json_prob: Number(document.getElementById("fuzzMal").value) / 100.0,
      dirty_arg_prob: Number(document.getElementById("fuzzDirty").value) / 100.0,
      policies: getSelectedPolicies(),
    };
    const res = await fetch("/api/fuzz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    currentPayload = await res.json();
    document.getElementById("modeBadge").textContent = `Fuzz Mode (seed=${body.seed})`;
    const br = currentPayload.fuzz_breakdown || {};
    const fc = br.fault_counts || {};
    document.getElementById("fuzzSummaryPill").textContent =
      `${br.total_scenarios || body.runs} cases · ${fc.prompt_injection || 0} inj · ${fc.http_503 || 0} 503s`;
    renderDashboard();
  } finally {
    setBusy(false);
  }
}

function renderDashboard() {
  if (!currentPayload) return;
  const lb = currentPayload.leaderboard || [];
  const traces = currentPayload.trajectories || [];

  const top = lb[0];
  const hardened = lb.find(r => r.policy_name === "TraceGuardHardenedAgent") || top;
  const naive = lb.find(r => r.policy_name === "NaiveAgent");

  document.getElementById("kpiTopPolicy").textContent = top ? top.policy_name : "—";
  document.getElementById("kpiTopScore").textContent = top ? `Composite Score ${top.composite_score.toFixed(2)} / 100` : "—";
  document.getElementById("kpiDefense").textContent = hardened ? `${(hardened.injection_defense_rate * 100).toFixed(1)}%` : "—";
  if (naive && hardened) {
    document.querySelector("#kpiDefense + .kpi-sub").textContent =
      `vs NaiveAgent ${(naive.injection_defense_rate * 100).toFixed(1)}%`;
  }
  document.getElementById("kpiRecovery").textContent = hardened ? `${(hardened.fault_recovery_rate * 100).toFixed(1)}%` : "—";
  document.getElementById("kpiTotalRuns").textContent = String(traces.length);

  document.getElementById("leaderboardBody").innerHTML = lb.map(r => `
    <tr>
      <td><strong>${escapeHtml(r.policy_name)}</strong></td>
      <td>${r.total_runs}</td>
      <td>${(r.task_success_rate * 100).toFixed(1)}%</td>
      <td>${(r.schema_compliance_rate * 100).toFixed(1)}%</td>
      <td>${(r.fault_recovery_rate * 100).toFixed(1)}%</td>
      <td>${(r.injection_defense_rate * 100).toFixed(1)}%</td>
      <td>${(r.loop_free_rate * 100).toFixed(1)}%</td>
      <td>${r.avg_latency_ms.toFixed(1)} ms</td>
      <td>${r.avg_tokens.toFixed(0)}</td>
      <td style="color:var(--cyan); font-weight:800;">${r.composite_score.toFixed(2)}</td>
    </tr>
  `).join("");

  document.getElementById("comparisonBars").innerHTML = lb.map(r => {
    const color = r.policy_name === "TraceGuardHardenedAgent"
      ? "linear-gradient(90deg, #22c55e, #38bdf8)"
      : (r.policy_name === "SchemaRetryAgent" ? "linear-gradient(90deg, #6366f1, #38bdf8)" : "linear-gradient(90deg, #f59e0b, #ef4444)");
    return `
      <div class="bar-group">
        <div class="bar-header">
          <strong>${escapeHtml(r.policy_name)}</strong>
          <span>Composite ${r.composite_score.toFixed(2)} · Success ${(r.task_success_rate*100).toFixed(0)}% · InjDef ${(r.injection_defense_rate*100).toFixed(0)}%</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, r.composite_score)}%; background:${color};"></div></div>
      </div>
    `;
  }).join("");

  populateFilterOptions(traces);
  selectedIdx = 0;
  renderTraceList();
}

function populateFilterOptions(traces) {
  const policies = Array.from(new Set(traces.map(t => t.policy_name)));
  const categories = Array.from(new Set(traces.map(t => t.category)));

  const fp = document.getElementById("filterPolicy");
  const prevP = fp.value;
  fp.innerHTML = `<option value="ALL">All Policies</option>` + policies.map(p => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("");
  if (policies.includes(prevP)) fp.value = prevP;

  const fc = document.getElementById("filterCategory");
  const prevC = fc.value;
  fc.innerHTML = `<option value="ALL">All Categories</option>` + categories.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  if (categories.includes(prevC)) fc.value = prevC;
}

function getFilteredTraces() {
  if (!currentPayload) return [];
  const p = document.getElementById("filterPolicy").value;
  const o = document.getElementById("filterOutcome").value;
  const c = document.getElementById("filterCategory").value;
  return (currentPayload.trajectories || []).filter(t =>
    (p === "ALL" || t.policy_name === p) &&
    (o === "ALL" || t.final_outcome === o) &&
    (c === "ALL" || t.category === c)
  );
}

function renderTraceList() {
  const list = getFilteredTraces();
  document.getElementById("filteredCount").textContent = `${list.length} traces`;
  if (selectedIdx >= list.length) selectedIdx = 0;

  document.getElementById("traceList").innerHTML = list.map((t, i) => `
    <div class="trace-row ${i === selectedIdx ? 'active' : ''}" onclick="selectTrace(${i})">
      <div class="trace-row-top">
        <span><strong>${escapeHtml(t.scenario_id)}</strong> · ${escapeHtml(t.policy_name)}</span>
        <span class="badge b-${escapeHtml(t.final_outcome)}">${escapeHtml(t.final_outcome)}</span>
      </div>
      <div style="font-size:12.5px; margin-top:3px;">${escapeHtml(t.scenario_title)}</div>
      <div class="trace-row-sub">
        <span>${escapeHtml(t.category)} · ${t.total_steps} steps</span>
        <span>${t.total_latency_ms.toFixed(1)} ms · ${t.total_tokens} tok</span>
      </div>
    </div>
  `).join("");

  renderTraceInspector(list[selectedIdx] || null);
}

function selectTrace(idx) {
  selectedIdx = idx;
  renderTraceList();
}

function renderTraceInspector(tr) {
  const container = document.getElementById("traceInspector");
  if (!tr) {
    document.getElementById("activeTraceLabel").textContent = "No trace selected";
    container.innerHTML = `<p style="color:var(--muted);">No trajectories match the selected filters.</p>`;
    return;
  }

  document.getElementById("activeTraceLabel").textContent = `${tr.scenario_id} · ${tr.policy_name}`;
  const maxLat = Math.max(1, ...tr.events.map(e => e.latency_ms || 0));
  const maxTok = Math.max(1, ...tr.events.map(e => e.tokens || 0));

  const pill = (ok, label) =>
    `<span class="badge ${ok ? 'b-resolved' : 'b-compromised'}">${ok ? 'PASS' : 'FAIL'} · ${escapeHtml(label)}</span>`;

  const eventsHtml = tr.events.map(ev => {
    const argsBox = ev.tool_args
      ? `<div class="codebox">ARGS: ${escapeHtml(JSON.stringify(ev.tool_args))}</div>`
      : "";
    const rawBox = ev.raw_payload
      ? `<div class="codebox raw">RAW PAYLOAD: ${escapeHtml(ev.raw_payload)}</div>`
      : "";
    const sanBox = ev.sanitized_payload
      ? `<div class="codebox san">SANITIZED CONTEXT: ${escapeHtml(ev.sanitized_payload)}</div>`
      : "";
    const permBadge = ev.permission_level
      ? `<span class="badge b-${escapeHtml(ev.permission_level)}">${escapeHtml(ev.permission_level)}</span>`
      : "";
    const faultBadge = ev.fault_type
      ? `<span class="badge b-compromised">${escapeHtml(ev.fault_type)}</span>`
      : "";

    const latPct = Math.min(100, ((ev.latency_ms || 0) / maxLat) * 100);
    const tokPct = Math.min(100, ((ev.tokens || 0) / maxTok) * 100);

    return `
      <div class="ev-card status-${escapeHtml(ev.status)}">
        <div class="ev-meta">
          <span><strong>Step ${ev.step}</strong> · ${escapeHtml(ev.event_type)} ${ev.tool_name ? '· ' + escapeHtml(ev.tool_name) : ''}</span>
          <span>${permBadge} ${faultBadge} @ ${ev.timestamp_ms.toFixed(1)} ms</span>
        </div>
        <div class="ev-title">${escapeHtml(ev.detail)}</div>
        ${argsBox}${rawBox}${sanBox}
        <div class="cost-bars">
          <div>
            <div class="mini-bar-label"><span>Latency</span><span>${ev.latency_ms.toFixed(1)} ms</span></div>
            <div class="mini-track"><div class="mini-fill-lat" style="width:${latPct}%;"></div></div>
          </div>
          <div>
            <div class="mini-bar-label"><span>Tokens</span><span>${ev.tokens}</span></div>
            <div class="mini-track"><div class="mini-fill-tok" style="width:${tokPct}%;"></div></div>
          </div>
        </div>
      </div>
    `;
  }).join("");

  container.innerHTML = `
    <div class="verdict-strip">
      <span class="badge b-${escapeHtml(tr.final_outcome)}">${escapeHtml(tr.final_outcome)}</span>
      ${pill(tr.task_success, "Task Success")}
      ${pill(tr.schema_compliant, "Schema")}
      ${pill(tr.fault_recovered, "Fault Recovery")}
      ${pill(tr.injection_defended, "Injection Defense")}
      ${pill(tr.loop_free, "Loop-Free")}
      <span class="pill">Sanitizer Catches: ${tr.sanitizer_catches} · RBAC Blocks: ${tr.rbac_blocks} · Retries: ${tr.retries_used}</span>
    </div>
    <div class="codebox" style="margin-bottom:12px;">FINAL ANSWER: ${escapeHtml(tr.final_answer)}</div>
    <div class="timeline">${eventsHtml}</div>
  `;
}

async function exportHtmlReport() {
  if (!currentPayload) return;
  const res = await fetch("/api/export-html", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: `TraceGuard Audit Report (${currentPayload.mode || 'benchmark'})`,
      mode: currentPayload.mode || "benchmark",
      leaderboard: currentPayload.leaderboard,
      trajectories: currentPayload.trajectories,
    }),
  });
  const htmlText = await res.text();
  const blob = new Blob([htmlText], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "traceguard-audit-report.html";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

init();
</script>
</body>
</html>
"""
