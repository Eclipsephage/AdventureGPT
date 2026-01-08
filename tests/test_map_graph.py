"""
Tests for MapGraph and MapAgent heuristics.
"""

from __future__ import annotations

from adventuregpt.map_agent import MapAgent
from adventuregpt.map_graph import MapGraph
from adventuregpt.planner import heuristic_plan


def test_map_graph_frontier_detects_untried_exits() -> None:
    g = MapGraph()
    r1 = g.observe_room("Room A", exits_mentioned={"north", "east"})
    r2 = g.observe_room("Room B", exits_mentioned=set())
    g.record_transition(r1, "north", r2)
    frontier = g.frontier()
    assert (r1, "east") in frontier
    assert (r1, "north") not in frontier


def test_map_agent_records_transition_on_move() -> None:
    a = MapAgent()
    a.observe_output("Start Room\nYou can go north.\n")
    start = a.graph.current_room
    a.observe_command("north")
    a.observe_output("North Room\nYou can go south.\n")
    end = a.graph.current_room
    assert start is not None and end is not None
    assert a.graph.known_exits(start).get("north") == end


def test_planner_heuristic_includes_frontier_tasks() -> None:
    g = MapGraph()
    g.observe_room("Room A", exits_mentioned={"north"})
    plan = heuristic_plan(g)
    names = plan.get_task_names()
    assert any("Explore north" in n for n in names)

