"""
Command normalization + loop breaking for AdventureGPT.

The Colossal Cave parser is limited, and the player agent sometimes emits
multiple sentences or long explanations. This module turns model output into a
single short command and provides simple loop detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


_ALIASES = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "u": "up",
    "d": "down",
    "i": "inventory",
    "inv": "inventory",
}

_SAFE_FALLBACKS = ("help", "look", "inventory", "north", "south", "east", "west")


def normalize_command_text(text: str, *, max_words: int = 3) -> str:
    """
    Convert arbitrary model output into a single short command.

    Strategy:
    - Take the first non-empty line
    - Take the first sentence-ish segment
    - Extract word tokens
    - Apply common aliases
    - Cap the number of words
    """

    if not text:
        return "help"

    # First non-empty line
    line = ""
    for raw in text.splitlines():
        raw = raw.strip()
        if raw:
            line = raw
            break
    if not line:
        return "help"

    # First sentence-ish segment
    line = re.split(r"[.!?]", line, maxsplit=1)[0]

    # Extract word tokens (letters/numbers/underscore)
    words = re.findall(r"\w+", line.lower())
    if not words:
        return "help"

    # Apply alias for first token
    words[0] = _ALIASES.get(words[0], words[0])

    # Cap length
    words = words[:max_words]

    return " ".join(words)


@dataclass
class LoopBreaker:
    """
    Tracks recent commands to detect simple loops.
    """

    window: int = 6
    repeat_threshold: int = 3
    recent: List[str] = field(default_factory=list)

    def record(self, command: str) -> None:
        command = command.strip().lower()
        if not command:
            return
        self.recent.append(command)
        if len(self.recent) > self.window:
            self.recent = self.recent[-self.window :]

    def suggest_alternative(self, command: str) -> Optional[str]:
        """
        If the command appears to be looping, suggest a safe alternative.
        """

        cmd = command.strip().lower()
        if not cmd:
            return "help"

        repeats = sum(1 for c in self.recent[-self.repeat_threshold :] if c == cmd)
        if repeats < self.repeat_threshold:
            return None

        # Simple heuristics: break out with help/inventory/look.
        if cmd == "look":
            return "help"
        if cmd in ("help", "inventory"):
            return "look"

        # Prefer a generic safe fallback that differs from the looping command.
        for alt in _SAFE_FALLBACKS:
            if alt != cmd:
                return alt
        return "help"

