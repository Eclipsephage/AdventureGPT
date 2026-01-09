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

from .actions import Action, parse_actions
from .agent import SingleTaskListStorage, prompt_to_history, openai_call
from .llm_client import LLMClient
from .map_graph import MapGraph
from .navigator import suggest_frontier_move
from .tools_layer import ToolSnapshot


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
        tasks.append({"task_name": Action(type="TAKE", arg="lamp").to_task_name(), "action": Action(type="TAKE", arg="lamp").to_dict()})
        tasks.append({"task_name": Action(type="USE", arg="lamp").to_task_name(), "action": Action(type="USE", arg="lamp").to_dict()})
    if "needs_yes_no" in blockers:
        tasks.append({"task_name": Action(type="SAY", arg="yes").to_task_name(), "action": Action(type="SAY", arg="yes").to_dict()})

    # Prefer current-room frontier first.
    if graph.current_room:
        known = graph.known_exits(graph.current_room)
        room = graph.rooms.get(graph.current_room)
        if room:
            for d in sorted(room.exits_mentioned - set(known.keys())):
                a = Action(type="MOVE", arg=d)
                tasks.append({"task_name": a.to_task_name(), "action": a.to_dict()})

    # Then any other frontier.
    for rid, d in graph.frontier():
        if len(tasks) >= 6:
            break
        a = Action(type="EXPLORE_FROM", arg=d, extra=rid)
        tasks.append({"task_name": a.to_task_name(), "action": a.to_dict()})

    # Always keep a couple generic “information gathering” tasks.
    tasks.extend(
        [
            {"task_name": Action(type="MOVE", arg=suggest_frontier_move(graph) or "north").to_task_name(), "action": Action(type="MOVE", arg=suggest_frontier_move(graph) or "north").to_dict()},
            {"task_name": Action(type="LOOK").to_task_name(), "action": Action(type="LOOK").to_dict()},
            {"task_name": Action(type="INVENTORY").to_task_name(), "action": Action(type="INVENTORY").to_dict()},
            {"task_name": Action(type="EXAMINE", arg="objects").to_task_name(), "action": Action(type="EXAMINE", arg="objects").to_dict()},
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
        tools: Optional[ToolSnapshot] = None,
        llm: Optional[LLMClient],
    ) -> SingleTaskListStorage:
        """
        Produce a prioritized list of next objectives.
        """

        if llm is None:
            return heuristic_plan(map_graph, blockers=blockers)

        prompt = f"""
You are an expert Colossal Cave Adventure planner.

Your job is to produce the next {self._config.max_tasks} ACTIONS to win the game.

Use the provided TOOLS + MAP summary and conversation context. Prefer actions that:
- unlock progress (pre-requisites)
- explore untried exits (frontier)
- collect useful items and treasures
- avoid repeating completed or blocked objectives

If progress appears blocked, include actions that unblock it.

Return a numbered list of ACTIONS using ONLY these action types:

- MOVE <direction>
- LOOK
- INVENTORY
- TAKE <object>
- DROP <object>
- EXAMINE <object>
- READ <object>
- USE <object>
- SAY <yes|no>
- EXPLORE_FROM <direction> | <room_id>

Examples:
1. LOOK
2. MOVE north
3. TAKE lamp
4. EXPLORE_FROM east | Small Chamber

No headers, no extra text.

TOOLS:
{tools.to_prompt() if tools else "(none)"}

MAP:
{map_graph.format_summary()}

Previously completed objectives:
{completed_tasks}

Detected blockers from recent output:
{", ".join(blockers) if blockers else "none"}
"""

        messages = prompt_to_history(prompt) + context_history[-30:]
        response = openai_call(messages, max_tokens=700, llm=llm)
        actions = parse_actions(response, max_actions=self._config.max_tasks)
        if not actions:
            return heuristic_plan(map_graph, blockers=blockers)

        tasks: List[Dict[str, str]] = []
        for a in actions:
            tasks.append({"task_name": a.to_task_name(), "action": a.to_dict()})
        return SingleTaskListStorage(tasks)

