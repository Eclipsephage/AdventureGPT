"""
Lightweight regression fixtures for core heuristics.

These are deterministic, fast tests that catch behavior regressions without
requiring live OpenAI calls.
"""

from __future__ import annotations

from adventuregpt.map_graph import MapGraph
from adventuregpt.state import detect_blockers
from adventuregpt.navigator import find_route


def test_map_room_dedupes_by_canonical_key() -> None:
    g = MapGraph()
    r1 = g.observe_room("Small Chamber")
    r2 = g.observe_room("small   chamber!!")
    assert r1 == r2
    assert len(g.rooms) == 1


def test_blocker_detection_dark() -> None:
    blockers = detect_blockers("It is now pitch dark. If you proceed you will likely fall.")
    assert "dark" in blockers


def test_navigator_route_smoke() -> None:
    g = MapGraph()
    a = g.observe_room("A")
    b = g.observe_room("B")
    c = g.observe_room("C")
    g.record_transition(a, "north", b)
    g.record_transition(b, "west", c)
    route = find_route(g, from_room=a, to_room=c)
    assert route is not None
    assert route.directions == ["north", "west"]

