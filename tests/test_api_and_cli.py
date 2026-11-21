"""Tests for FastAPI studio endpoints and CLI runner execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from traceguard.server import app

client = TestClient(app)


def test_studio_home_and_health_endpoints() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers["content-type"]
    assert "TraceGuard Eval" in home.text

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["scenario_count"] == 12
    assert set(body["policies"]) == {"NaiveAgent", "SchemaRetryAgent", "TraceGuardHardenedAgent"}


def test_scenarios_catalog_endpoint() -> None:
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_count"] == 12
    assert len(data["scenarios"]) == 12
    assert len(data["tools"]) == 6
    assert len(data["policies"]) == 3


def test_benchmark_endpoint_and_validation() -> None:
    resp = client.post("/api/benchmark", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_count"] == 12
    assert len(data["leaderboard"]) == 3
    assert len(data["trajectories"]) == 36
    assert data["leaderboard"][0]["policy_name"] == "TraceGuardHardenedAgent"
    assert data["leaderboard"][0]["injection_defense_rate"] == 1.0

    bad = client.post("/api/benchmark", json={"policies": ["NonExistentPolicy"]})
    assert bad.status_code == 400


def test_fuzz_and_export_html_endpoints() -> None:
    fuzz_resp = client.post(
        "/api/fuzz",
        json={
            "runs": 8,
            "seed": 77,
            "injection_prob": 0.5,
            "transient_fault_prob": 0.5,
            "persistent_fault_prob": 0.1,
            "malformed_json_prob": 0.2,
            "dirty_arg_prob": 0.25,
        },
    )
    assert fuzz_resp.status_code == 200
    fuzz_data = fuzz_resp.json()
    assert fuzz_data["mode"] == "fuzz"
    assert fuzz_data["fuzz_breakdown"]["total_scenarios"] == 8
    assert len(fuzz_data["trajectories"]) == 24

    export_resp = client.post(
        "/api/export-html",
        json={
            "title": "API Test Audit",
            "mode": "fuzz",
            "leaderboard": fuzz_data["leaderboard"],
            "trajectories": fuzz_data["trajectories"],
        },
    )
    assert export_resp.status_code == 200
    assert "text/html" in export_resp.headers["content-type"]
    assert "attachment" in export_resp.headers.get("content-disposition", "")
    assert "API Test Audit" in export_resp.text
    assert "TraceGuardHardenedAgent" in export_resp.text


def test_cli_benchmark_fuzz_and_html_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    report_path = tmp_path / "cli-report.html"

    bench_proc = subprocess.run(
        [sys.executable, "main.py", "--benchmark", "--json", "--export-html", str(report_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    bench_json = json.loads(bench_proc.stdout)
    assert bench_json["benchmark"]["scenario_count"] == 12
    assert len(bench_json["benchmark"]["leaderboard"]) == 3
    assert report_path.exists()
    assert "TraceGuardHardenedAgent" in report_path.read_text(encoding="utf-8")

    fuzz_proc = subprocess.run(
        [sys.executable, "main.py", "--fuzz", "--runs", "6", "--seed", "42", "--json"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    fuzz_json = json.loads(fuzz_proc.stdout)
    assert fuzz_json["fuzz"]["fault_mix"]["runs"] == 6
    assert fuzz_json["fuzz"]["fault_mix"]["seed"] == 42
    assert len(fuzz_json["fuzz"]["trajectories"]) == 18


def _row(composite: float):
    from traceguard.contracts import PolicyBenchmarkRow

    return PolicyBenchmarkRow(
        policy_name="p",
        total_runs=1,
        task_success_rate=0.0,
        schema_compliance_rate=0.0,
        fault_recovery_rate=0.0,
        injection_defense_rate=0.0,
        loop_free_rate=0.0,
        avg_latency_ms=0.0,
        avg_tokens=0.0,
        composite_score=composite,
    )


def test_exit_code_no_threshold_always_succeeds() -> None:
    from traceguard.cli import exit_code_for

    assert exit_code_for([], None) == 0


def test_exit_code_gates_on_best_composite_score() -> None:
    from traceguard.cli import exit_code_for

    assert exit_code_for([_row(40.0), _row(90.0)], 80.0) == 0
    assert exit_code_for([_row(40.0), _row(70.0)], 80.0) == 1
    assert exit_code_for([_row(80.0)], 80.0) == 0
    assert exit_code_for([], 80.0) == 1


def test_exit_code_rejects_invalid_threshold() -> None:
    import pytest

    from traceguard.cli import exit_code_for

    for bad in (float("nan"), float("inf"), -0.1, 100.1):
        with pytest.raises(ValueError, match="min_composite must be a finite value"):
            exit_code_for([_row(100.0)], bad)


def test_cli_min_composite_gates_the_benchmark() -> None:
    from traceguard.cli import main

    # NaiveAgent scores well below the hardened policy on the built-in suite.
    assert main(["--benchmark", "--policies", "NaiveAgent", "--min-composite", "30.0"]) == 0
    assert main(["--benchmark", "--policies", "NaiveAgent", "--min-composite", "80.0"]) == 1
