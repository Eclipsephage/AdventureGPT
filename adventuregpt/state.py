"""
Game state tracking for AdventureGPT.

This module maintains a lightweight, heuristic state derived from the game's
text output and the commands we send. It's intentionally conservative: the game
parser/output format can vary, so we avoid brittle parsing.

State is used to:
- Provide a compact "what we know" summary to agent prompts
- Support loop detection and safer command selection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


_DIRECTIONS = ("north", "south", "east", "west", "up", "down", "in", "out")


def _extract_directions(text: str) -> Set[str]:
    lowered = text.lower()
    found = {d for d in _DIRECTIONS if re.search(rf"\b{re.escape(d)}\b", lowered)}
    return found


def _first_nonempty_line(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return None


def _extract_inventory(text: str) -> Optional[List[str]]:
    """
    Heuristically extract an inventory listing.

    Many interactive fiction games output inventory like:
        You are carrying:
        a lamp
        a bottle
    """

    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\byou are carrying\b", line.lower()):
            items: List[str] = []
            # read subsequent non-empty lines as items until a blank line
            for j in range(i + 1, len(lines)):
                item = lines[j].strip()
                if not item:
                    break
                items.append(item)
            return items
    return None


@dataclass
class GameState:
    """
    Minimal state representation derived from output.
    """

    turn: int = 0
    last_command: Optional[str] = None
    last_system_output: str = ""

    # Heuristics
    last_scene_line: Optional[str] = None
    possible_directions: Set[str] = field(default_factory=set)
    inventory: Optional[List[str]] = None
    blockers: List[str] = field(default_factory=list)


class GameStateTracker:
    """
    Updates GameState from commands and game output.
    """

    def __init__(self) -> None:
        self.state = GameState()

    def observe_command(self, command: str) -> None:
        """
        Record the command we decided to send to the game.
        """

        self.state.last_command = command
        self.state.turn += 1

    def observe_system_output(self, output: str) -> None:
        """
        Update state based on the latest game output.
        """

        self.state.last_system_output = output or ""
        self.state.last_scene_line = _first_nonempty_line(output)
        self.state.possible_directions |= _extract_directions(output)

        inv = _extract_inventory(output)
        if inv is not None:
            self.state.inventory = inv

        self.state.blockers = detect_blockers(output)

    def format_summary(self, *, max_chars: int = 700) -> str:
        """
        Format a compact state summary suitable for a system prompt.
        """

        inv = self.state.inventory
        inv_str = ", ".join(inv) if inv else "unknown"
        dirs = ", ".join(sorted(self.state.possible_directions)) if self.state.possible_directions else "unknown"
        scene = self.state.last_scene_line or "unknown"
        blockers = ", ".join(self.state.blockers) if self.state.blockers else "none"

        snippet = (self.state.last_system_output or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."

        return (
            "## Game state (heuristic)\n"
            f"- Turn: {self.state.turn}\n"
            f"- Last command: {self.state.last_command or 'none'}\n"
            f"- Scene line: {scene}\n"
            f"- Possible directions mentioned: {dirs}\n"
            f"- Inventory: {inv_str}\n"
            f"- Blockers: {blockers}\n"
            f"- Last output snippet: {snippet}\n"
        )


def detect_blockers(text: str) -> List[str]:
    """
    Detect common "blocked progress" signals from game output.

    Returns a list of short blocker tags.
    """

    t = (text or "").lower()
    blockers: List[str] = []
    if "too dark" in t or "it is now pitch dark" in t:
        blockers.append("dark")
    if "locked" in t:
        blockers.append("locked")
    if "i don't know that word" in t or "i dont know that word" in t:
        blockers.append("unknown_word")
    if "you can't" in t or "you cant" in t:
        blockers.append("cant")
    if "i see no" in t or "you see no" in t:
        blockers.append("not_present")
    if "please answer the question" in t:
        blockers.append("needs_yes_no")
    # de-dupe, stable order
    out: List[str] = []
    for b in blockers:
        if b not in out:
            out.append(b)
    return out

