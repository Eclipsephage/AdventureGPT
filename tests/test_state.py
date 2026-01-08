"""
Tests for heuristic state extraction.
"""

from __future__ import annotations

from adventuregpt.state import GameStateTracker


def test_state_tracker_extracts_inventory_and_directions() -> None:
    tracker = GameStateTracker()
    tracker.observe_command("inventory")
    tracker.observe_system_output(
        "You are carrying:\n"
        "a lamp\n"
        "a bottle\n"
        "\n"
        "You can go north or east.\n"
    )

    summary = tracker.format_summary()
    assert "a lamp" in summary
    assert "a bottle" in summary
    assert "north" in summary
    assert "east" in summary

