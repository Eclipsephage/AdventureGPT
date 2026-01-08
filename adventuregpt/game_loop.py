"""
AdventureGPT game loop orchestration.

This module owns the runtime loop that:
- Initializes the Colossal Cave game engine
- Uses LLM agents to select commands
- Records artifacts (history, commands, metrics)

Usage:
    from adventuregpt.game_loop import GameLoop
    GameLoop(...).run()
"""

from __future__ import annotations

import functools
import operator
import re
import sys
from time import sleep
from typing import Dict, List, Optional

from adventure import load_advent_dat
from adventure.game import Game

from .agent import (
    SingleTaskListStorage,
    gametask_creation_agent,
    player_agent,
    prioritization_agent,
    task_completion_agent,
    walkthrough_gametask_creation_agent,
)
from .run_artifacts import RunArtifacts


BAUD = 1200


class GameLoop:
    """
    Run the Adventure game loop using LLM agents.
    """

    def __init__(self, walkthrough_path: Optional[str], artifacts: RunArtifacts):
        self.walkthrough_path = walkthrough_path or None
        self.artifacts = artifacts

        self.history: List[Dict[str, str]] = []
        self.game_tasks = SingleTaskListStorage()
        self.completed_tasks = SingleTaskListStorage()
        self.current_task: Optional[str] = None

        self.game: Optional[Game] = None

    def _baudout(self, s: str) -> None:
        """
        Slowly print output to stdout to emulate terminal baud output.
        """

        out = sys.stdout
        for c in s:
            sleep(9.0 / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            out.write(c)
            out.flush()

    def _append_history(self, role: str, content: str) -> None:
        """
        Append to in-memory history and to JSONL artifacts.
        """

        event = {"role": role, "content": content}
        self.history.append(event)
        self.artifacts.write_history_event(event)

    def _next_game_task(self) -> None:
        """
        Pop the next task, recording the previous as completed.
        """

        print("***************** TASK LIST *******************")
        print(self.game_tasks)
        print()

        if self.current_task:
            self.completed_tasks.append({"task_name": self.current_task})
            self.artifacts.increment("tasks_completed", 1)

        next_task = self.game_tasks.popleft()
        if next_task:
            self.current_task = next_task.get("task_name")
        else:
            self.current_task = None

    def _init_tasks(self) -> None:
        """
        Initialize task list from walkthrough or from initial history.
        """

        self.artifacts.set_walkthrough_enabled(bool(self.walkthrough_path))

        if self.walkthrough_path:
            # Read walkthrough in approximate token chunks.
            text_chunks: list[str] = []
            chunk_size = 500
            curr_chunk: list[str] = []

            with open(self.walkthrough_path, "r", encoding="utf-8") as f:
                for line in f:
                    curr_chunk += line.split()
                    if len(curr_chunk) >= chunk_size:
                        text_chunks.append(" ".join(curr_chunk))
                        curr_chunk = []

            if curr_chunk:
                text_chunks.append(" ".join(curr_chunk))

            for chunk in text_chunks:
                tasks = walkthrough_gametask_creation_agent(chunk)
                self.game_tasks.concat(tasks)
        else:
            self.game_tasks = gametask_creation_agent(self.history)

        self._next_game_task()

    def run(self) -> None:
        """
        Main game loop.
        """

        print("***************** INITIALIZING GAME *******************")

        self._init_tasks()

        self.game = Game()
        load_advent_dat(self.game)
        self.game.start()

        next_input = self.game.output
        self._baudout(next_input)
        self._append_history("system", next_input)

        while not self.game.is_finished:
            if not self.current_task:
                # If tasks are exhausted, generate more from history.
                if self.walkthrough_path:
                    self.artifacts.record_error("Task list exhausted while using walkthrough mode.")
                    break
                self.game_tasks = gametask_creation_agent(self.history)
                self._next_game_task()
                if not self.current_task:
                    self.artifacts.record_error("Unable to generate a non-empty task list.")
                    break

            # Ask Player Agent what to do next
            result = player_agent(self.current_task, self.history, self.completed_tasks)
            self._append_history("assistant", result)

            # Split lines by newlines and periods and flatten list
            newline_split = result.lower().split("\n")
            period_split = [line.split(".") for line in newline_split]
            split_lines = functools.reduce(operator.iconcat, period_split, [])

            # We got input! Act on it.
            for line in split_lines:
                words = re.findall(r"\w+", line)
                if not words:
                    continue

                self.artifacts.increment("steps", 1)

                command_output = self.game.do_command(words)
                self._append_history("system", command_output)

                self._baudout(f"> {line}\n\n")
                self._baudout(command_output)

                self.artifacts.write_command(line)
                self.artifacts.increment("commands_sent", 1)

                # If not using a walkthrough, come up with more tasks and prioritize
                if not self.walkthrough_path:
                    new_tasks = gametask_creation_agent(self.history)
                    self.game_tasks.concat(new_tasks)
                    self.game_tasks = prioritization_agent(self.game_tasks, self.history)

                completed = task_completion_agent(self.current_task, self.history)
                if completed:
                    self._next_game_task()

        # Always write a final pretty dump for human inspection.
        self.artifacts.write_history_dump(self.history)

