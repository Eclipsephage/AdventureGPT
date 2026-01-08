"""
Tests for SingleTaskListStorage behavior.

Run:
    python3 -m pytest
"""

from __future__ import annotations

from adventuregpt.agent import SingleTaskListStorage


def test_task_storage_popleft_empty_returns_none() -> None:
    storage = SingleTaskListStorage()
    assert storage.popleft() is None


def test_task_storage_concat_accepts_storage() -> None:
    a = SingleTaskListStorage([{"task_name": "a"}])
    b = SingleTaskListStorage([{"task_name": "b"}])
    a.concat(b)
    assert a.get_task_names() == ["a", "b"]


def test_task_storage_concat_accepts_iterable() -> None:
    a = SingleTaskListStorage([{"task_name": "a"}])
    a.concat([{"task_name": "b"}, {"task_name": "c"}])
    assert a.get_task_names() == ["a", "b", "c"]

