"""
Tests for eval report aggregation.
"""

from __future__ import annotations

from adventuregpt.eval import aggregate_metrics


def test_aggregate_metrics_basic() -> None:
    metrics = [
        {
            "steps": 2,
            "commands_sent": 2,
            "tasks_completed": 1,
            "errors": [],
            "tokens": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "estimated_cost_usd": 0.01,
        },
        {
            "steps": 4,
            "commands_sent": 3,
            "tasks_completed": 0,
            "errors": ["x"],
            "tokens": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            "estimated_cost_usd": 0.0,
        },
    ]
    summary = aggregate_metrics(metrics)
    assert summary["runs"] == 2
    assert summary["total_steps"] == 6
    assert summary["total_commands_sent"] == 5
    assert summary["total_tasks_completed"] == 1
    assert summary["errors_count"] == 1
    assert summary["total_input_tokens"] == 12
    assert summary["total_output_tokens"] == 8
    assert summary["total_tokens"] == 20
    assert summary["total_estimated_cost_usd"] == 0.01

