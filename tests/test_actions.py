"""
Tests for structured action parsing and deterministic command conversion.
"""

from __future__ import annotations

from adventuregpt.actions import Action, action_to_command, parse_actions


def test_parse_actions_basic() -> None:
    text = "1. LOOK\n2. MOVE north\n3. TAKE lamp\n"
    actions = parse_actions(text)
    assert [a.type for a in actions] == ["LOOK", "MOVE", "TAKE"]
    assert actions[1].arg == "north"


def test_parse_actions_explore_from() -> None:
    text = "1. EXPLORE_FROM east | Room A\n"
    actions = parse_actions(text)
    assert len(actions) == 1
    assert actions[0].type == "EXPLORE_FROM"
    assert actions[0].arg == "east"
    assert actions[0].extra == "Room A"


def test_action_to_command() -> None:
    assert action_to_command(Action(type="LOOK")) == "look"
    assert action_to_command(Action(type="INVENTORY")) == "inventory"
    assert action_to_command(Action(type="MOVE", arg="north")) == "north"
    assert action_to_command(Action(type="TAKE", arg="lamp")) == "take lamp"

