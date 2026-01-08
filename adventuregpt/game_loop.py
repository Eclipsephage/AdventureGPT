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
import time
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
from .map_agent import MapAgent
from .memory import MemoryConfig, MemoryManager
from .planner import WinPlanner
from .run_artifacts import RunArtifacts
from .state import GameStateTracker
from .navigator import suggest_frontier_move
from .ui_hooks import StdoutUI, UIHooks
from .outcome import detect_outcome
from .navigation_tasks import next_command_for_objective


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
        planner_llm: Optional[LLMClient] = None,
        max_steps: Optional[int] = None,
        max_seconds: Optional[float] = None,
        ui: Optional[UIHooks] = None,
    ):
        self.walkthrough_path = walkthrough_path or None
        self.artifacts = artifacts
        self.dry_run = bool(dry_run)
        self.llm = llm
        self.planner_llm = planner_llm
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.max_seconds = float(max_seconds) if max_seconds is not None else None
        self._started_monotonic: Optional[float] = None
        self.ui: UIHooks = ui or StdoutUI()

        self.history: List[Dict[str, str]] = []
        self.game_tasks = SingleTaskListStorage()
        self.completed_tasks = SingleTaskListStorage()
        self.current_task: Optional[str] = None

        self.game: Optional[Game] = None
        self.state = GameStateTracker()
        self.memory = MemoryManager(MemoryConfig())
        self.loop_breaker = LoopBreaker()
        self.map_agent = MapAgent()
        self.planner = WinPlanner()
        self._task_attempts: dict[str, int] = {}
        self._max_task_attempts: int = 3

    def _baudout(self, s: str) -> None:
        """
        Slowly print output to stdout to emulate terminal baud output.
        """

        if self.dry_run:
            self.ui.on_output(s)
            return
        for c in s:
            sleep(9.0 / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            self.ui.on_output(c)

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
            self.map_agent.observe_output(content)
            oc = detect_outcome(content)
            if oc is not None:
                self.artifacts.set_outcome(
                    is_game_over=oc.is_game_over,
                    is_victory=oc.is_victory,
                    score=oc.score,
                    max_score=oc.max_score,
                    reason=oc.reason,
                )

    def _next_game_task(self) -> None:
        """
        Pop the next task, recording the previous as completed.
        """

        self.ui.on_task_list(str(self.game_tasks))

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
            # Phase 3: win-oriented planner (map/state aware).
            self.game_tasks = self.planner.plan(
                context_history=self.history,
                map_graph=self.map_agent.graph,
                completed_tasks=str(self.completed_tasks),
                blockers=self.state.state.blockers,
                llm=self.planner_llm or self.llm,
            )

        self._next_game_task()

    def run(self) -> None:
        """
        Main game loop.
        """

        self.ui.on_status("***************** INITIALIZING GAME *******************")

        self._init_tasks()

        self.game = Game()
        load_advent_dat(self.game)
        self.game.start()
        self._started_monotonic = time.monotonic()

        next_input = self.game.output
        self._baudout(next_input)
        self._append_history("system", next_input)

        dry_stop = False
        while not self.game.is_finished:
            if (
                self.max_seconds is not None
                and self._started_monotonic is not None
                and (time.monotonic() - self._started_monotonic) >= self.max_seconds
            ):
                self.artifacts.record_error(f"Stopped due to max_seconds limit ({self.max_seconds}).")
                break
            if self.max_steps is not None and int(self.artifacts.metrics.get("steps", 0)) >= self.max_steps:
                self.artifacts.record_error(f"Stopped due to max_steps limit ({self.max_steps}).")
                break
            if not self.current_task:
                # If tasks are exhausted, generate more from history.
                if self.walkthrough_path:
                    self.artifacts.record_error("Task list exhausted while using walkthrough mode.")
                    break
                self.game_tasks = self.planner.plan(
                    context_history=self.history,
                    map_graph=self.map_agent.graph,
                    completed_tasks=str(self.completed_tasks),
                    blockers=self.state.state.blockers,
                    llm=self.planner_llm or self.llm,
                )
                self._next_game_task()
                if not self.current_task:
                    self.artifacts.record_error("Unable to generate a non-empty task list.")
                    break

            # Deterministic navigation step if objective matches a known pattern.
            if not self.dry_run and self.current_task:
                nav = next_command_for_objective(self.current_task, graph=self.map_agent.graph)
                if nav is not None:
                    # Treat this as the chosen command; skip LLM for this turn.
                    result = nav.command
                    self._append_history("assistant", f"[navigation:{nav.reason}] {result}")
                else:
                    result = None
            else:
                result = None

            # Ask Player Agent what to do next
            if self.dry_run:
                result = "look"
            elif result is None:
                state_summary = self.state.format_summary()
                map_summary = self.map_agent.format_prompt_addendum()
                self.ui.on_info(
                    objective=self.current_task or "",
                    state_summary=state_summary,
                    map_summary=map_summary,
                )
                context_history = self.memory.build_context(
                    self.history,
                    llm=self.llm,
                    state_summary=state_summary + "\n" + map_summary,
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
            self.map_agent.observe_command(command)
            self.artifacts.increment("steps", 1)
            if self.max_steps is not None and int(self.artifacts.metrics.get("steps", 0)) >= self.max_steps:
                self.artifacts.record_error(f"Stopped due to max_steps limit ({self.max_steps}).")
                break
            if (
                self.max_seconds is not None
                and self._started_monotonic is not None
                and (time.monotonic() - self._started_monotonic) >= self.max_seconds
            ):
                self.artifacts.record_error(f"Stopped due to max_seconds limit ({self.max_seconds}).")
                break

            command_output = self.game.do_command(words)
            self._append_history("system", command_output)

            self.ui.on_command(command)
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
                map_summary = self.map_agent.format_prompt_addendum()
                context_history = self.memory.build_context(
                    self.history,
                    llm=self.llm,
                    state_summary=state_summary + "\n" + map_summary,
                )
                # Phase 3: refresh task list from planner (map/state aware).
                self.game_tasks = self.planner.plan(
                    context_history=context_history,
                    map_graph=self.map_agent.graph,
                    completed_tasks=str(self.completed_tasks),
                    blockers=self.state.state.blockers,
                    llm=self.planner_llm or self.llm,
                )

            state_summary = self.state.format_summary()
            map_summary = self.map_agent.format_prompt_addendum()
            context_history = self.memory.build_context(
                self.history,
                llm=self.llm,
                state_summary=state_summary + "\n" + map_summary,
            )
            completed = task_completion_agent(self.current_task, context_history, llm=self.llm)
            if completed:
                self._next_game_task()
                continue

            # If we keep failing the same task, mark it blocked and move on.
            if self.current_task:
                self._task_attempts[self.current_task] = self._task_attempts.get(self.current_task, 0) + 1
                if self._task_attempts[self.current_task] >= self._max_task_attempts:
                    self.completed_tasks.append({"task_name": f"BLOCKED: {self.current_task}"})
                    self.artifacts.record_error(f"Objective blocked after {self._max_task_attempts} attempts: {self.current_task}")
                    self._next_game_task()

            if dry_stop:
                break

        # Always write a final pretty dump for human inspection.
        self.artifacts.write_history_dump(self.history)
        # Persist map artifact for replay/debugging.
        self.artifacts.write_json("map.json", self.map_agent.graph.to_dict())
        # Record map stats in metrics.
        self.artifacts.set_map_stats(
            rooms_discovered=len(self.map_agent.graph.rooms),
            transitions_recorded=len(self.map_agent.graph.edges),
            frontier_size=len(self.map_agent.graph.frontier()),
        )

