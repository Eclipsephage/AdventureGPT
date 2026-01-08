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
from .command_safety import LoopBreaker, normalize_command_text
from .llm_client import LLMClient
from .memory import MemoryConfig, MemoryManager
from .run_artifacts import RunArtifacts
from .state import GameStateTracker


BAUD = 1200


class GameLoop:
    """
    Run the Adventure game loop using LLM agents.
    """

    def __init__(
        self,
        walkthrough_path: Optional[str],
        artifacts: RunArtifacts,
        dry_run: bool = False,
        llm: Optional[LLMClient] = None,
        max_steps: Optional[int] = None,
    ):
        self.walkthrough_path = walkthrough_path or None
        self.artifacts = artifacts
        self.dry_run = bool(dry_run)
        self.llm = llm
        self.max_steps = int(max_steps) if max_steps is not None else None

        self.history: List[Dict[str, str]] = []
        self.game_tasks = SingleTaskListStorage()
        self.completed_tasks = SingleTaskListStorage()
        self.current_task: Optional[str] = None

        self.game: Optional[Game] = None
        self.state = GameStateTracker()
        self.memory = MemoryManager(MemoryConfig())
        self.loop_breaker = LoopBreaker()

    def _baudout(self, s: str) -> None:
        """
        Slowly print output to stdout to emulate terminal baud output.
        """

        out = sys.stdout
        if self.dry_run:
            out.write(s)
            out.flush()
            return
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

        # Update heuristic state from game output.
        if role == "system":
            self.state.observe_system_output(content)

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

        if self.dry_run:
            self.game_tasks = SingleTaskListStorage([{"task_name": "Dry run: initialize"}])
            self._next_game_task()
            return

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
                tasks = walkthrough_gametask_creation_agent(chunk, llm=self.llm)
                self.game_tasks.concat(tasks)
        else:
            self.game_tasks = gametask_creation_agent(self.history, llm=self.llm)

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

        dry_stop = False
        while not self.game.is_finished:
            if self.max_steps is not None and int(self.artifacts.metrics.get("steps", 0)) >= self.max_steps:
                self.artifacts.record_error(f"Stopped due to max_steps limit ({self.max_steps}).")
                break
            if not self.current_task:
                # If tasks are exhausted, generate more from history.
                if self.walkthrough_path:
                    self.artifacts.record_error("Task list exhausted while using walkthrough mode.")
                    break
                self.game_tasks = gametask_creation_agent(self.history, llm=self.llm)
                self._next_game_task()
                if not self.current_task:
                    self.artifacts.record_error("Unable to generate a non-empty task list.")
                    break

            # Ask Player Agent what to do next
            if self.dry_run:
                result = "look"
            else:
                state_summary = self.state.format_summary()
                context_history = self.memory.build_context(
                    self.history,
                    llm=self.llm,
                    state_summary=state_summary,
                )
                result = player_agent(
                    self.current_task,
                    context_history,
                    self.completed_tasks,
                    llm=self.llm,
                )
            self._append_history("assistant", result)

            # Normalize to a single safe command.
            command = normalize_command_text(result)
            alt = self.loop_breaker.suggest_alternative(command)
            if alt:
                command = alt

            words = re.findall(r"\w+", command)
            if not words:
                words = ["help"]

            self.loop_breaker.record(command)
            self.state.observe_command(command)
            self.artifacts.increment("steps", 1)
            if self.max_steps is not None and int(self.artifacts.metrics.get("steps", 0)) >= self.max_steps:
                self.artifacts.record_error(f"Stopped due to max_steps limit ({self.max_steps}).")
                break

            command_output = self.game.do_command(words)
            self._append_history("system", command_output)

            self._baudout(f"> {command}\n\n")
            self._baudout(command_output)

            self.artifacts.write_command(command)
            self.artifacts.increment("commands_sent", 1)

            # If not using a walkthrough, come up with more tasks and prioritize
            if self.dry_run:
                # Don't call OpenAI in dry-run mode.
                self._next_game_task()
                dry_stop = True
                break

            if not self.walkthrough_path:
                state_summary = self.state.format_summary()
                context_history = self.memory.build_context(
                    self.history,
                    llm=self.llm,
                    state_summary=state_summary,
                )
                new_tasks = gametask_creation_agent(context_history, llm=self.llm)
                self.game_tasks.concat(new_tasks)
                self.game_tasks = prioritization_agent(self.game_tasks, context_history, llm=self.llm)

            state_summary = self.state.format_summary()
            context_history = self.memory.build_context(
                self.history,
                llm=self.llm,
                state_summary=state_summary,
            )
            completed = task_completion_agent(self.current_task, context_history, llm=self.llm)
            if completed:
                self._next_game_task()

            if dry_stop:
                break

        # Always write a final pretty dump for human inspection.
        self.artifacts.write_history_dump(self.history)

