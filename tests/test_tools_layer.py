"""
Tests for tool snapshot construction.
"""

from __future__ import annotations

from adventuregpt.map_agent import MapAgent
from adventuregpt.state import GameStateTracker
from adventuregpt.tools_layer import build_tool_snapshot


def test_build_tool_snapshot_contains_frontier() -> None:
    state = GameStateTracker()
    map_agent = MapAgent()
    map_agent.observe_output("Room A\nYou can go north.\n")
    snap = build_tool_snapshot(state, map_agent)
    assert snap.current_room is not None
    assert isinstance(snap.frontier, list)

