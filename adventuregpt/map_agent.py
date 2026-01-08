"""
Map agent for AdventureGPT.

This module updates a MapGraph based on:
- The last command we sent (directional movement)
- The next system output (which we treat as "arrived at a room")

It is heuristic: Colossal Cave doesn't expose a stable room ID in text output.
"""

from __future__ import annotations

import re
from typing import Optional, Set

from .map_graph import MapGraph


_MOVE_COMMANDS = {"north", "south", "east", "west", "up", "down", "in", "out"}


def _first_nonempty_line(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return None


def _extract_directions(text: str) -> Set[str]:
    """
    Extract direction words from output as a hint for exits/frontier.
    """

    lowered = (text or "").lower()
    directions = set()
    for d in _MOVE_COMMANDS:
        if re.search(rf"\b{re.escape(d)}\b", lowered):
            directions.add(d)
    return directions


def _is_meta_line(line: str) -> bool:
    """
    Filter out obviously non-room header lines.
    """

    l = (line or "").strip().lower()
    if not l:
        return True
    if "welcome to adventure" in l:
        return True
    if l.endswith("?") and "would you like" in l:
        return True
    return False


class MapAgent:
    """
    Maintains a map graph during play.
    """

    def __init__(self) -> None:
        self.graph = MapGraph()
        self._pending_move_dir: Optional[str] = None
        self._pending_from_room: Optional[str] = None

    def observe_command(self, command: str) -> None:
        """
        Observe the normalized command we sent to the game.
        """

        cmd = (command or "").strip().lower()
        first = cmd.split(" ", 1)[0] if cmd else ""
        if first in _MOVE_COMMANDS:
            self._pending_move_dir = first
            self._pending_from_room = self.graph.current_room
        else:
            self._pending_move_dir = None
            self._pending_from_room = None

    def observe_output(self, output: str) -> None:
        """
        Observe system output after a command and update map.
        """

        line = _first_nonempty_line(output) or "UNKNOWN"
        if _is_meta_line(line):
            # Still record directions mentioned, but don't treat as a room transition.
            exits = _extract_directions(output)
            if self.graph.current_room and exits:
                self.graph.rooms[self.graph.current_room].exits_mentioned |= exits
            return

        exits = _extract_directions(output)
        new_room = self.graph.observe_room(line, exits_mentioned=exits)

        if self._pending_move_dir and self._pending_from_room:
            self.graph.record_transition(self._pending_from_room, self._pending_move_dir, new_room)

        self._pending_move_dir = None
        self._pending_from_room = None

    def format_prompt_addendum(self) -> str:
        """
        Map context to inject into prompts.
        """

        return self.graph.format_summary()

