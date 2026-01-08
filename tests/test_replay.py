"""
Tests for replay assembly.
"""

from __future__ import annotations

from adventuregpt.replay import replay


def test_replay_interleaves_outputs_and_commands() -> None:
    outputs = ["WELCOME\n", "OK\n", "DONE\n"]
    commands = ["look", "north"]
    text = replay(outputs, commands)
    assert "WELCOME" in text
    assert "> look" in text
    assert "OK" in text
    assert "> north" in text
    assert "DONE" in text

