"""
Tests for map navigation utilities.
"""

from __future__ import annotations

from adventuregpt.map_graph import MapGraph
from adventuregpt.navigator import find_route
from adventuregpt.navigation_tasks import next_command_for_objective


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


def test_next_command_for_objective_routes_then_explores() -> None:
    g = MapGraph()
    a = g.observe_room("A")
    b = g.observe_room("B")
    g.record_transition(a, "north", b)
    g.current_room = a
    # objective wants to explore east from B; first we need to go north to B
    d1 = next_command_for_objective("Explore east from B", graph=g)
    assert d1 is not None
    assert d1.command == "north"
    # now pretend we are at B
    g.current_room = b
    d2 = next_command_for_objective("Explore east from B", graph=g)
    assert d2 is not None
    assert d2.command == "east"

