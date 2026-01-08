"""
Game outcome detection for AdventureGPT.

This module provides heuristic detection of:
- victory/game-over signals
- score lines (e.g., "You have scored 123 out of 350.")

It is intentionally conservative and should never crash the game loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Outcome:
    """
    Detected outcome information from game output.
    """

    is_game_over: bool
    is_victory: bool
    score: Optional[int] = None
    max_score: Optional[int] = None
    reason: Optional[str] = None


_SCORE_RE = re.compile(r"\byou have scored\s+(\d+)\s+out of\s+(\d+)\b", re.IGNORECASE)


def detect_outcome(text: str) -> Optional[Outcome]:
    """
    Detect an outcome from a chunk of game output.

    Returns:
        Outcome if something meaningful is detected, else None.
    """

    t = (text or "").strip()
    if not t:
        return None

    lowered = t.lower()

    # Score line detection (usually appears at end of game).
    m = _SCORE_RE.search(lowered)
    if m:
        score = int(m.group(1))
        max_score = int(m.group(2))
        # If score line appears, it's very likely end-of-game.
        return Outcome(
            is_game_over=True,
            is_victory=score >= max_score,  # conservative guess
            score=score,
            max_score=max_score,
            reason="score_reported",
        )

    # Generic end signals (heuristic).
    if "you have won" in lowered or "you win" in lowered:
        return Outcome(is_game_over=True, is_victory=True, reason="win_text")
    if "game over" in lowered:
        return Outcome(is_game_over=True, is_victory=False, reason="game_over_text")
    if "you are dead" in lowered or "you have died" in lowered:
        return Outcome(is_game_over=True, is_victory=False, reason="death_text")

    return None

