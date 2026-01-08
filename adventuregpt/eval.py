"""
Evaluation harness for AdventureGPT.

This runs multiple sessions and produces an aggregate report so progress can be
tracked over time (steps, loops, tasks completed, etc.).

Usage:
    python -m adventuregpt.eval --runs 10 --dry_run

Outputs:
    - report.json (aggregate + per-run)
    - report.csv  (per-run rows)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .game_loop import GameLoop
from .llm_client import CappedLLMClient, CachedLLMClient, OpenAIResponsesClient, OpenAIResponsesConfig
from .run_artifacts import RunArtifacts, RunArtifactsConfig


@dataclass(frozen=True)
class EvalConfig:
    runs: int
    dry_run: bool
    out_dir: Path
    model: str
    temperature: float
    max_output_tokens: int
    max_steps: int
    max_seconds: float
    cost_per_1k_input_usd: Optional[float]
    cost_per_1k_output_usd: Optional[float]
    models: List[str]
    temperatures: List[float]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args(argv: Optional[list[str]] = None) -> EvalConfig:
    parser = argparse.ArgumentParser(prog="adventuregpt-eval", description="AdventureGPT eval harness")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to execute.")
    parser.add_argument("--dry_run", action="store_true", help="Run without calling OpenAI.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write eval outputs (default: ./eval_runs/<timestamp>).",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (non-dry-run).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (non-dry-run).")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated models to run as a matrix (overrides --model).",
    )
    parser.add_argument(
        "--temperatures",
        default=None,
        help="Comma-separated temperatures to run as a matrix (overrides --temperature).",
    )
    parser.add_argument("--max_output_tokens", type=int, default=2000, help="Max output tokens cap.")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per run before stopping.")
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=60.0,
        help="Max wall-clock seconds per run before stopping.",
    )
    parser.add_argument(
        "--cost_per_1k_input_usd",
        type=float,
        default=None,
        help="Optional cost estimate: USD per 1k input tokens.",
    )
    parser.add_argument(
        "--cost_per_1k_output_usd",
        type=float,
        default=None,
        help="Optional cost estimate: USD per 1k output tokens.",
    )
    ns = parser.parse_args(argv)

    out_dir = Path(ns.out_dir) if ns.out_dir else Path("eval_runs") / _utc_stamp()

    models = [str(ns.model)]
    if ns.models:
        models = [m.strip() for m in str(ns.models).split(",") if m.strip()]
    temperatures = [float(ns.temperature)]
    if ns.temperatures:
        temperatures = [float(t.strip()) for t in str(ns.temperatures).split(",") if t.strip()]

    return EvalConfig(
        runs=int(ns.runs),
        dry_run=bool(ns.dry_run),
        out_dir=out_dir,
        model=str(ns.model),
        temperature=float(ns.temperature),
        max_output_tokens=int(ns.max_output_tokens),
        max_steps=int(ns.max_steps),
        max_seconds=float(ns.max_seconds),
        cost_per_1k_input_usd=ns.cost_per_1k_input_usd,
        cost_per_1k_output_usd=ns.cost_per_1k_output_usd,
        models=models,
        temperatures=temperatures,
    )


def aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-run metrics into a summary.
    """

    def _sum(key: str) -> int:
        return sum(int(m.get(key, 0) or 0) for m in all_metrics)

    runs = len(all_metrics)
    total_steps = _sum("steps")
    total_commands = _sum("commands_sent")
    total_tasks = _sum("tasks_completed")
    total_input_tokens = sum(int((m.get("tokens") or {}).get("input_tokens", 0) or 0) for m in all_metrics)
    total_output_tokens = sum(int((m.get("tokens") or {}).get("output_tokens", 0) or 0) for m in all_metrics)
    total_total_tokens = sum(int((m.get("tokens") or {}).get("total_tokens", 0) or 0) for m in all_metrics)
    total_cost = sum(float(m.get("estimated_cost_usd", 0.0) or 0.0) for m in all_metrics)

    return {
        "runs": runs,
        "total_steps": total_steps,
        "total_commands_sent": total_commands,
        "total_tasks_completed": total_tasks,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_total_tokens,
        "total_estimated_cost_usd": total_cost,
        "avg_steps": (total_steps / runs) if runs else 0.0,
        "avg_commands_sent": (total_commands / runs) if runs else 0.0,
        "avg_tasks_completed": (total_tasks / runs) if runs else 0.0,
        "avg_total_tokens": (total_total_tokens / runs) if runs else 0.0,
        "avg_estimated_cost_usd": (total_cost / runs) if runs else 0.0,
        "errors_count": sum(len(m.get("errors", []) or []) for m in all_metrics),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """
    Write a per-run CSV report.
    """

    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Stable columns (plus optional extras)
    fieldnames = [
        "run_id",
        "started_at_utc",
        "ended_at_utc",
        "dry_run",
        "steps",
        "commands_sent",
        "tasks_completed",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "errors_count",
    ]

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "run_id": r.get("run_id"),
                    "started_at_utc": r.get("started_at_utc"),
                    "ended_at_utc": r.get("ended_at_utc"),
                    "dry_run": r.get("dry_run"),
                    "steps": r.get("steps", 0),
                    "commands_sent": r.get("commands_sent", 0),
                    "tasks_completed": r.get("tasks_completed", 0),
                    "input_tokens": (r.get("tokens") or {}).get("input_tokens", 0),
                    "output_tokens": (r.get("tokens") or {}).get("output_tokens", 0),
                    "total_tokens": (r.get("tokens") or {}).get("total_tokens", 0),
                    "estimated_cost_usd": r.get("estimated_cost_usd", 0.0),
                    "errors_count": len(r.get("errors", []) or []),
                }
            )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    per_run: List[Dict[str, Any]] = []
    stamp = _utc_stamp()

    matrix_reports: List[Dict[str, Any]] = []

    combos = [(m, t) for m in cfg.models for t in cfg.temperatures]
    for combo_idx, (model, temperature) in enumerate(combos):
        per_run = []
        combo_dir = cfg.out_dir / f"matrix_{combo_idx:02d}_{model.replace('/', '_')}_t{temperature}"
        combo_dir.mkdir(parents=True, exist_ok=True)

        for i in range(cfg.runs):
            run_id = f"{i:03d}_{stamp}"
            run_dir = combo_dir / run_id

            artifacts = RunArtifacts(RunArtifactsConfig(base_run_dir=str(run_dir)))
            artifacts.set_llm_config(
                model=model,
                temperature=temperature,
                max_output_tokens=cfg.max_output_tokens,
                dry_run=cfg.dry_run,
            )
            artifacts.set_cost_rates(
                cost_per_1k_input_usd=cfg.cost_per_1k_input_usd,
                cost_per_1k_output_usd=cfg.cost_per_1k_output_usd,
            )

            llm = None
            if not cfg.dry_run:
                import os

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set (required for non-dry-run eval).")
                base = OpenAIResponsesClient(
                    api_key=api_key,
                    config=OpenAIResponsesConfig(model=model, temperature=temperature),
                    on_usage=artifacts.add_usage,
                )
                llm = CachedLLMClient(CappedLLMClient(base, max_output_tokens_cap=cfg.max_output_tokens))

            loop = GameLoop(
                walkthrough_path=None,
                artifacts=artifacts,
                dry_run=cfg.dry_run,
                llm=llm,
                max_steps=cfg.max_steps,
                max_seconds=cfg.max_seconds,
            )

            try:
                loop.run()
            finally:
                artifacts.close()

            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["run_id"] = run_id
            per_run.append(metrics)

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "runs": cfg.runs,
                "dry_run": cfg.dry_run,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": cfg.max_output_tokens,
                "max_steps": cfg.max_steps,
                "max_seconds": cfg.max_seconds,
                "cost_per_1k_input_usd": cfg.cost_per_1k_input_usd,
                "cost_per_1k_output_usd": cfg.cost_per_1k_output_usd,
            },
            "summary": aggregate_metrics(per_run),
            "runs": per_run,
        }

        (combo_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_csv(per_run, combo_dir / "report.csv")
        matrix_reports.append(report)

    # Write a top-level index report for the whole matrix.
    index = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "runs": cfg.runs,
            "dry_run": cfg.dry_run,
            "models": cfg.models,
            "temperatures": cfg.temperatures,
            "max_output_tokens": cfg.max_output_tokens,
            "max_steps": cfg.max_steps,
            "max_seconds": cfg.max_seconds,
            "cost_per_1k_input_usd": cfg.cost_per_1k_input_usd,
            "cost_per_1k_output_usd": cfg.cost_per_1k_output_usd,
        },
        "matrix": [
            {
                "model": r["config"]["model"],
                "temperature": r["config"]["temperature"],
                "summary": r["summary"],
            }
            for r in matrix_reports
        ],
    }
    (cfg.out_dir / "matrix_report.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

