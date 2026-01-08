"""
UI hook interfaces for AdventureGPT.

Phase 4: allow alternate UIs (curses TUI, web UI) without changing game logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class UIHooks(Protocol):
    """
    Interface for receiving game loop events.
    """

    def on_task_list(self, text: str) -> None: ...
    def on_output(self, text: str) -> None: ...
    def on_command(self, command: str) -> None: ...
    def on_status(self, text: str) -> None: ...


@dataclass
class StdoutUI:
    """
    Default UI hook implementation (prints to stdout).
    """

    def on_task_list(self, text: str) -> None:
        print("***************** TASK LIST *******************")
        print(text)
        print()

    def on_output(self, text: str) -> None:
        print(text, end="")

    def on_command(self, command: str) -> None:
        print(f"> {command}\n")

    def on_status(self, text: str) -> None:
        # Keep status minimal to avoid spam.
        if text:
            print(text)

