"""
Trend report generator for AdventureGPT eval runs.

This reads multiple eval run directories (each containing report.json) and
produces a consolidated trend CSV/JSON.

Usage:
    python -m adventuregpt.trend --eval_root eval_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def discover_reports(eval_root: Path) -> List[Path]:
    return sorted(eval_root.glob("**/report.json"))


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_trend(reports: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (out_dir / "trend.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    # CSV
    import csv

    with (out_dir / "trend.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "generated_at_utc",
                "runs",
                "dry_run",
                "model",
                "temperature",
                "max_steps",
                "max_seconds",
                "avg_steps",
                "avg_total_tokens",
                "avg_estimated_cost_usd",
                "errors_count",
            ]
        )
        for r in reports:
            cfg = r.get("config") or {}
            s = r.get("summary") or {}
            w.writerow(
                [
                    r.get("generated_at_utc"),
                    cfg.get("runs"),
                    cfg.get("dry_run"),
                    cfg.get("model"),
                    cfg.get("temperature"),
                    cfg.get("max_steps"),
                    cfg.get("max_seconds"),
                    s.get("avg_steps"),
                    s.get("avg_total_tokens"),
                    s.get("avg_estimated_cost_usd"),
                    s.get("errors_count"),
                ]
            )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adventuregpt-trend", description="Trend report for eval runs")
    p.add_argument("--eval_root", default="eval_runs", help="Root directory containing eval outputs.")
    p.add_argument("--out_dir", default="eval_trends", help="Directory to write trend outputs.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    eval_root = Path(ns.eval_root)
    out_dir = Path(ns.out_dir)

    report_paths = discover_reports(eval_root)
    reports = [load_report(p) for p in report_paths]
    write_trend(reports, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

