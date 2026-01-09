"""
Tests for task list parsing utilities.

Run:
    python3 -m pytest
"""

from __future__ import annotations

from adventuregpt.agent import openai_task_response_to_list


def test_openai_task_response_to_list_parses_numbered_lines() -> None:
    response = "1. Get lamp\n2. Go north\n3. Take keys\n"
    tasks = openai_task_response_to_list(response)
    assert tasks == [
        {"task_name": "Get lamp"},
        {"task_name": "Go north"},
        {"task_name": "Take keys"},
    ]


def test_openai_task_response_to_list_ignores_bad_lines() -> None:
    response = "\nHello\nX. Not numbered\n1) Wrong format\n2. Valid task!\n"
    tasks = openai_task_response_to_list(response)
    assert tasks == [{"task_name": "Valid task"}]

