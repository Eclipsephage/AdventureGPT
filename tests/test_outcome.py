"""
Tests for outcome detection heuristics.
"""

from __future__ import annotations

from adventuregpt.outcome import detect_outcome


def test_detect_outcome_score_line() -> None:
    o = detect_outcome("You have scored 123 out of 350.\n")
    assert o is not None
    assert o.is_game_over is True
    assert o.score == 123
    assert o.max_score == 350


def test_detect_outcome_win_text() -> None:
    o = detect_outcome("You have won!\n")
    assert o is not None
    assert o.is_victory is True

