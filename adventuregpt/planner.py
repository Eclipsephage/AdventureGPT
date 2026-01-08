"""
Win-oriented planner for AdventureGPT.

This planner produces game objectives using:
- Heuristic state summary (from GameStateTracker)
- Map summary and frontier (from MapAgent / MapGraph)
- Completed tasks list

It supports two modes:
- LLM-backed planning (preferred): produce a small, ordered objective list
- Heuristic fallback: explore map frontier and gather info (look/inventory/examine)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .agent import SingleTaskListStorage, openai_task_response_to_list, prompt_to_history, openai_call
from .llm_client import LLMClient
from .map_graph import MapGraph
from .navigator import suggest_frontier_move


@dataclass(frozen=True)
class PlannerConfig:
    """
    Configuration for planning behavior.
    """

    max_tasks: int = 8


def heuristic_plan(graph: MapGraph, *, blockers: Sequence[str] = ()) -> SingleTaskListStorage:
    """
    Deterministic fallback planning: explore untried exits and gather info.
    """

    tasks: List[Dict[str, str]] = []

    blockers = list(blockers or [])
    if "dark" in blockers:
        tasks.append({"task_name": "Find and take a lamp or light source"})
        tasks.append({"task_name": "Turn on the lamp if you have it"})
    if "needs_yes_no" in blockers:
        tasks.append({"task_name": "Answer the prompt with yes or no"})

    # Prefer current-room frontier first.
    if graph.current_room:
        known = graph.known_exits(graph.current_room)
        room = graph.rooms.get(graph.current_room)
        if room:
            for d in sorted(room.exits_mentioned - set(known.keys())):
                tasks.append({"task_name": f"Explore {d} from current room"})

    # Then any other frontier.
    for rid, d in graph.frontier():
        if len(tasks) >= 6:
            break
        tasks.append({"task_name": f"Explore {d} from {rid}"})

    # Always keep a couple generic “information gathering” tasks.
    tasks.extend(
        [
            {"task_name": f"Try moving {suggest_frontier_move(graph) or 'north'} to explore"} ,
            {"task_name": "Look around carefully"},
            {"task_name": "Check inventory"},
            {"task_name": "Examine visible objects"},
        ]
    )

    return SingleTaskListStorage(tasks)


class WinPlanner:
    """
    Plan objectives using map/state/tools.
    """

    def __init__(self, config: PlannerConfig | None = None):
        self._config = config or PlannerConfig()

    def plan(
        self,
        *,
        context_history: List[Dict[str, str]],
        map_graph: MapGraph,
        completed_tasks: str,
        blockers: Sequence[str] = (),
        llm: Optional[LLMClient],
    ) -> SingleTaskListStorage:
        """
        Produce a prioritized list of next objectives.
        """

        if llm is None:
            return heuristic_plan(map_graph, blockers=blockers)

        prompt = f"""
You are an expert Colossal Cave Adventure planner.

Your job is to produce the next {self._config.max_tasks} objectives to win the game.

Use the provided MAP summary and conversation context. Prefer objectives that:
- unlock progress (pre-requisites)
- explore untried exits (frontier)
- collect useful items and treasures
- avoid repeating completed or blocked objectives

If an objective appears blocked, include it only if you also include a prerequisite objective that may unblock it.

Return a numbered list in the format:
1. Do X
2. Do Y

No headers, no extra text.

MAP:
{map_graph.format_summary()}

Previously completed objectives:
{completed_tasks}

Detected blockers from recent output:
{", ".join(blockers) if blockers else "none"}
"""

        messages = prompt_to_history(prompt) + context_history[-30:]
        response = openai_call(messages, max_tokens=700, llm=llm)
        tasks = openai_task_response_to_list(response)
        if not tasks:
            return heuristic_plan(map_graph, blockers=blockers)

        # Cap number of tasks defensively.
        return SingleTaskListStorage(tasks[: self._config.max_tasks])

