"""
Tests for command normalization and loop breaking.
"""

from __future__ import annotations

from adventuregpt.command_safety import LoopBreaker, normalize_command_text


def test_normalize_command_text_prefers_first_line_sentence() -> None:
    text = "Look around.\nThen go north."
    assert normalize_command_text(text) == "look around"


def test_normalize_command_text_caps_words_and_applies_alias() -> None:
    text = "N please go north quickly"
    assert normalize_command_text(text) == "north please go"


def test_loop_breaker_suggests_alternative_on_repeats() -> None:
    lb = LoopBreaker(window=6, repeat_threshold=3)
    lb.record("look")
    lb.record("look")
    lb.record("look")
    assert lb.suggest_alternative("look") == "help"

