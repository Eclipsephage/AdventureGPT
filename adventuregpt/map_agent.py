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
from .navigator import route_to_nearest_frontier, suggest_frontier_move


_MOVE_COMMANDS = {"north", "south", "east", "west", "up", "down", "in", "out"}
_META_PATTERNS = (
    "welcome to adventure",
    "please answer the question",
    "i don't understand",
    "i dont understand",
    "you are not carrying anything",
)


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
    if any(p in l for p in _META_PATTERNS):
        return True
    if l.endswith("?") and ("would you like" in l or "do you want" in l):
        return True
    return False


def _extract_room_label(output: str) -> Optional[str]:
    """
    Best-effort extraction of a stable-ish room label from output.

    Strategy:
    - Prefer "You are in/at ..." sentences if present.
    - Otherwise, use the first non-empty line that isn't meta.
    """

    text = output or ""
    lowered = text.lower()

    # Prefer explicit "You are in/at ..." sentences if present.
    m = re.search(r"\byou are (in|at)\b\s+(.+?)([.!]\s|$)", lowered)
    if m:
        # Use the described place as a room label key.
        place = m.group(2).strip()
        # Avoid extremely long labels.
        place = " ".join(place.split())[:120]
        return place

    # Prefer a short, title-like first line (often room name).
    first = _first_nonempty_line(text)
    if first and not _is_meta_line(first):
        # Heuristic: short line, no trailing punctuation, and either Title Case or ALL CAPS.
        is_short = len(first) <= 60
        no_punct_tail = not re.search(r"[.!?]$", first.strip())
        is_titleish = first.isupper() or first.istitle()
        if is_short and no_punct_tail and is_titleish:
            return first.strip()

    # fallback: first non-empty non-meta line
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _is_meta_line(s):
            continue
        return s
    return None


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

        label = _extract_room_label(output)
        if not label:
            label = _first_nonempty_line(output) or "UNKNOWN"

        if _is_meta_line(label):
            # Still record directions mentioned, but don't treat as a room transition.
            exits = _extract_directions(output)
            if self.graph.current_room and exits:
                self.graph.rooms[self.graph.current_room].exits_mentioned |= exits
            return

        exits = _extract_directions(output)
        new_room = self.graph.observe_room(label, exits_mentioned=exits)

        if self._pending_move_dir and self._pending_from_room:
            self.graph.record_transition(self._pending_from_room, self._pending_move_dir, new_room)

        self._pending_move_dir = None
        self._pending_from_room = None

    def format_prompt_addendum(self) -> str:
        """
        Map context to inject into prompts.
        """

        suggestion = suggest_frontier_move(self.graph)
        route = route_to_nearest_frontier(self.graph)
        if route and route.directions:
            route_str = " -> ".join(route.directions[:6])
        else:
            route_str = "none"

        return (
            self.graph.format_summary()
            + "## Navigation hints\n"
            + f"- Suggested frontier move (current room): {suggestion or 'unknown'}\n"
            + f"- Route to nearest frontier (known edges): {route_str}\n"
        )

