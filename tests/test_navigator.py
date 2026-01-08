"""
Tests for map navigation utilities.
"""

from __future__ import annotations

from adventuregpt.map_graph import MapGraph
from adventuregpt.navigator import find_route


def test_find_route_bfs() -> None:
    g = MapGraph()
    a = g.observe_room("A")
    b = g.observe_room("B")
    c = g.observe_room("C")
    g.record_transition(a, "north", b)
    g.record_transition(b, "east", c)

    route = find_route(g, from_room=a, to_room=c)
    assert route is not None
    assert route.directions == ["north", "east"]

