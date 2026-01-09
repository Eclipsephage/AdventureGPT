"""
Deterministic navigation task execution for AdventureGPT.

Phase 3 improvement:
    When planner outputs objectives like "Explore east from Room A", the agent
    should be able to navigate to Room A using the discovered map edges and then
    perform the exploration step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .map_graph import MapGraph
from .navigator import find_route


_EXPLORE_FROM_RE = re.compile(r"^explore\s+(north|south|east|west|up|down|in|out)\s+from\s+(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class NavigationDecision:
    """
    Represents a deterministic next command derived from an objective.
    """

    command: str
    reason: str


def next_command_for_objective(objective: str, *, graph: MapGraph) -> Optional[NavigationDecision]:
    """
    If objective matches known patterns, return a deterministic next command.
    Otherwise return None.
    """

    if not graph.current_room:
        return None

    obj = (objective or "").strip()
    m = _EXPLORE_FROM_RE.match(obj)
    if not m:
        return None

    direction = m.group(1).lower()
    target = m.group(2).strip()
    # Normalize target to the graph's internal room id if possible.
    # We allow both room_id and label-ish strings since planner may output either.
    target_id = None
    if target in graph.rooms:
        target_id = target
    else:
        # Try canonical index lookup by using MapGraph's canonical key behavior
        # indirectly via observe_room-like normalization. Here we just do a best-effort match.
        for rid, room in graph.rooms.items():
            if room.label.lower().strip() == target.lower().strip():
                target_id = rid
                break

    if target_id is None:
        # Can't route deterministically; let the LLM handle it.
        return None

    if graph.current_room != target_id:
        route = find_route(graph, from_room=graph.current_room, to_room=target_id)
        if route and route.directions:
            return NavigationDecision(command=route.directions[0], reason=f"route_to:{target_id}")
        return None

    # We are at the target room: execute exploration direction.
    return NavigationDecision(command=direction, reason="explore_from_target")

