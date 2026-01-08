"""
Tests for eval report aggregation.
"""

from __future__ import annotations

from adventuregpt.eval import aggregate_metrics


def test_aggregate_metrics_basic() -> None:
    metrics = [
        {"steps": 2, "commands_sent": 2, "tasks_completed": 1, "errors": []},
        {"steps": 4, "commands_sent": 3, "tasks_completed": 0, "errors": ["x"]},
    ]
    summary = aggregate_metrics(metrics)
    assert summary["runs"] == 2
    assert summary["total_steps"] == 6
    assert summary["total_commands_sent"] == 5
    assert summary["total_tasks_completed"] == 1
    assert summary["errors_count"] == 1

