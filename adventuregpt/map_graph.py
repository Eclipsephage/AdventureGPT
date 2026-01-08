"""
Map graph representation for AdventureGPT.

This is a lightweight world model of Colossal Cave derived from observed text.
It intentionally uses conservative heuristics because the game's output isn't
strictly structured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


Direction = str
RoomId = str


def _normalize_room_id(label: str) -> str:
    """
    Normalize a room label into a stable ID.

    We keep it simple: strip and collapse whitespace. This isn't perfect but
    keeps IDs stable across a run.
    """

    return " ".join((label or "").strip().split()) or "UNKNOWN"


@dataclass
class Room:
    """
    A discovered room node.
    """

    room_id: RoomId
    label: str
    seen_count: int = 0
    exits_mentioned: Set[Direction] = field(default_factory=set)


@dataclass
class MapGraph:
    """
    Directed graph of rooms and transitions.
    """

    rooms: Dict[RoomId, Room] = field(default_factory=dict)
    edges: Dict[Tuple[RoomId, Direction], RoomId] = field(default_factory=dict)
    current_room: Optional[RoomId] = None

    def observe_room(self, label: str, *, exits_mentioned: Optional[Set[Direction]] = None) -> RoomId:
        """
        Mark a room as visited/observed.
        """

        rid = _normalize_room_id(label)
        room = self.rooms.get(rid)
        if room is None:
            room = Room(room_id=rid, label=label)
            self.rooms[rid] = room
        room.seen_count += 1
        if exits_mentioned:
            room.exits_mentioned |= set(exits_mentioned)
        self.current_room = rid
        return rid

    def record_transition(self, from_room: RoomId, direction: Direction, to_room: RoomId) -> None:
        """
        Record a directed edge from one room to another via a direction.
        """

        self.edges[(from_room, direction)] = to_room

    def known_exits(self, room_id: RoomId) -> Dict[Direction, RoomId]:
        """
        Return known exits from a given room.
        """

        out: Dict[Direction, RoomId] = {}
        for (rid, d), dst in self.edges.items():
            if rid == room_id:
                out[d] = dst
        return out

    def frontier(self) -> List[Tuple[RoomId, Direction]]:
        """
        Return (room, direction) pairs that were mentioned but not yet traversed.
        """

        out: List[Tuple[RoomId, Direction]] = []
        for rid, room in self.rooms.items():
            known = set(self.known_exits(rid).keys())
            for d in sorted(room.exits_mentioned - known):
                out.append((rid, d))
        return out

    def format_summary(self, *, max_frontier: int = 8) -> str:
        """
        Format a compact map summary for prompts.
        """

        current = self.current_room or "unknown"
        n_rooms = len(self.rooms)
        n_edges = len(self.edges)
        frontier = self.frontier()[:max_frontier]
        frontier_str = ", ".join([f"{rid} -> {d}" for rid, d in frontier]) if frontier else "none"

        return (
            "## Map (heuristic)\n"
            f"- Rooms discovered: {n_rooms}\n"
            f"- Transitions recorded: {n_edges}\n"
            f"- Current room id: {current}\n"
            f"- Frontier (untried exits): {frontier_str}\n"
        )

    def to_dict(self) -> dict:
        """
        Serialize graph to a JSON-friendly dict.
        """

        return {
            "current_room": self.current_room,
            "rooms": {
                rid: {
                    "label": r.label,
                    "seen_count": r.seen_count,
                    "exits_mentioned": sorted(r.exits_mentioned),
                }
                for rid, r in self.rooms.items()
            },
            "edges": [
                {"from": rid, "direction": d, "to": dst}
                for (rid, d), dst in self.edges.items()
            ],
            "frontier": [{"room": rid, "direction": d} for rid, d in self.frontier()],
        }

