"""
Structured actions for AdventureGPT.

Phase 3 tool-structured planning:
    The planner emits structured actions (MOVE/TAKE/LOOK/etc.) rather than free-form tasks.
    The game loop can often execute these deterministically without the player LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


ActionType = Literal[
    "MOVE",
    "LOOK",
    "INVENTORY",
    "TAKE",
    "DROP",
    "EXAMINE",
    "READ",
    "USE",
    "SAY",
    "EXPLORE_FROM",
]


_MOVE_DIRS = {"north", "south", "east", "west", "up", "down", "in", "out"}


@dataclass(frozen=True)
class Action:
    """
    A structured action with optional arguments.
    """

    type: ActionType
    arg: Optional[str] = None
    extra: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "arg": self.arg, "extra": self.extra}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Action":
        return Action(type=d["type"], arg=d.get("arg"), extra=d.get("extra"))

    def to_task_name(self) -> str:
        if self.type == "MOVE":
            return f"Move {self.arg}"
        if self.type == "LOOK":
            return "Look around carefully"
        if self.type == "INVENTORY":
            return "Check inventory"
        if self.type == "TAKE":
            return f"Take {self.arg}"
        if self.type == "DROP":
            return f"Drop {self.arg}"
        if self.type == "EXAMINE":
            return f"Examine {self.arg}"
        if self.type == "READ":
            return f"Read {self.arg}"
        if self.type == "USE":
            return f"Use {self.arg}"
        if self.type == "SAY":
            return f"Say {self.arg}"
        if self.type == "EXPLORE_FROM":
            # arg=direction, extra=room
            return f"Explore {self.arg} from {self.extra}"
        return f"{self.type} {self.arg or ''}".strip()


def action_to_command(action: Action) -> Optional[str]:
    """
    Convert a structured action into a single game command string when possible.
    """

    t = action.type
    if t == "LOOK":
        return "look"
    if t == "INVENTORY":
        return "inventory"
    if t == "MOVE":
        d = (action.arg or "").lower().strip()
        if d in _MOVE_DIRS:
            return d
        return None
    if t == "TAKE":
        if action.arg:
            return f"take {action.arg}".strip()
        return None
    if t == "DROP":
        if action.arg:
            return f"drop {action.arg}".strip()
        return None
    if t == "EXAMINE":
        if action.arg:
            return f"examine {action.arg}".strip()
        return None
    if t == "READ":
        if action.arg:
            return f"read {action.arg}".strip()
        return None
    if t == "USE":
        if action.arg:
            return f"use {action.arg}".strip()
        return None
    if t == "SAY":
        if action.arg:
            return str(action.arg).strip().lower()
        return None
    if t == "EXPLORE_FROM":
        # Needs routing; GameLoop handles it via navigation_tasks.
        return None
    return None


_ACTION_LINE_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(MOVE|LOOK|INVENTORY|TAKE|DROP|EXAMINE|READ|USE|SAY|EXPLORE_FROM)\b\s*(.*)$",
    re.IGNORECASE,
)


def parse_action_line(line: str) -> Optional[Action]:
    """
    Parse a single action line like:
      1. MOVE north
      2. TAKE lamp
      3. EXPLORE_FROM east | Room A
    """

    m = _ACTION_LINE_RE.match(line or "")
    if not m:
        return None
    typ = m.group(1).upper()
    rest = (m.group(2) or "").strip()

    if typ in ("LOOK", "INVENTORY"):
        return Action(type=typ)  # type: ignore[arg-type]

    if typ == "EXPLORE_FROM":
        # Format: "<dir> | <room>"
        if "|" not in rest:
            return None
        dir_part, room_part = [p.strip() for p in rest.split("|", 1)]
        if not dir_part or not room_part:
            return None
        return Action(type="EXPLORE_FROM", arg=dir_part.lower(), extra=room_part)

    if rest:
        return Action(type=typ, arg=rest.lower())  # type: ignore[arg-type]
    return None


def parse_actions(text: str, *, max_actions: int = 12) -> List[Action]:
    """
    Parse an LLM response into a list of actions.
    """

    actions: List[Action] = []
    for line in (text or "").splitlines():
        a = parse_action_line(line)
        if a is None:
            continue
        actions.append(a)
        if len(actions) >= max_actions:
            break
    return actions

