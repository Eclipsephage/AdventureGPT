"""
Map navigation utilities for AdventureGPT.

Provides route finding over the discovered MapGraph (BFS).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

from .map_graph import Direction, MapGraph, RoomId


@dataclass(frozen=True)
class Route:
    """
    A route as a sequence of directions to follow.
    """

    from_room: RoomId
    to_room: RoomId
    directions: List[Direction]


def find_route(graph: MapGraph, *, from_room: RoomId, to_room: RoomId, max_depth: int = 50) -> Optional[Route]:
    """
    Find a route between two rooms using BFS over known transitions.
    """

    if from_room == to_room:
        return Route(from_room=from_room, to_room=to_room, directions=[])

    # Build adjacency: room -> [(neighbor, direction)]
    adj: Dict[RoomId, List[Tuple[RoomId, Direction]]] = {}
    for (rid, d), dst in graph.edges.items():
        adj.setdefault(rid, []).append((dst, d))

    q: Deque[RoomId] = deque([from_room])
    prev: Dict[RoomId, Tuple[RoomId, Direction]] = {}
    seen: Set[RoomId] = {from_room}
    depth: Dict[RoomId, int] = {from_room: 0}

    while q:
        cur = q.popleft()
        if depth[cur] >= max_depth:
            continue
        for nxt, d in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = (cur, d)
            depth[nxt] = depth[cur] + 1
            if nxt == to_room:
                # Reconstruct
                dirs: List[Direction] = []
                node = to_room
                while node != from_room:
                    p, pd = prev[node]
                    dirs.append(pd)
                    node = p
                dirs.reverse()
                return Route(from_room=from_room, to_room=to_room, directions=dirs)
            q.append(nxt)

    return None


def suggest_frontier_move(graph: MapGraph) -> Optional[Direction]:
    """
    Suggest a move direction based on frontier exits from the current room.
    """

    if not graph.current_room:
        return None
    room = graph.rooms.get(graph.current_room)
    if not room:
        return None
    known = set(graph.known_exits(graph.current_room).keys())
    candidates = sorted(room.exits_mentioned - known)
    return candidates[0] if candidates else None


def route_to_nearest_frontier(graph: MapGraph, *, max_depth: int = 50) -> Optional[Route]:
    """
    Find a route from current room to any room with frontier exits.

    Returns the shortest route found, or None.
    """

    if not graph.current_room:
        return None
    start = graph.current_room
    frontier_rooms = {rid for rid, _d in graph.frontier()}
    if start in frontier_rooms:
        # A route to self doesn't help; rely on suggest_frontier_move().
        return Route(from_room=start, to_room=start, directions=[])

    best: Optional[Route] = None
    for target in frontier_rooms:
        r = find_route(graph, from_room=start, to_room=target, max_depth=max_depth)
        if r is None:
            continue
        if best is None or len(r.directions) < len(best.directions):
            best = r
    return best

