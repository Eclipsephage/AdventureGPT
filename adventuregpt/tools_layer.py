"""
Tool snapshot layer for AdventureGPT.

Phase 3 tool-structured planning:
    Provide a structured snapshot of "tools" (map/state/memory) to the planner/player,
    and keep the interface stable as internals evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .map_agent import MapAgent
from .state import GameStateTracker


@dataclass(frozen=True)
class ToolSnapshot:
    """
    Stable, prompt-friendly view of the current run state.
    """

    blockers: List[str]
    inventory: Optional[List[str]]
    current_room: Optional[str]
    frontier: List[Tuple[str, str]]
    navigation_hint: Optional[str]
    last_output_snippet: str

    def to_prompt(self) -> str:
        inv = ", ".join(self.inventory) if self.inventory else "unknown"
        fr = ", ".join([f"{r}->{d}" for r, d in self.frontier[:8]]) if self.frontier else "none"
        bl = ", ".join(self.blockers) if self.blockers else "none"
        return (
            "## Tools snapshot\n"
            f"- Current room: {self.current_room or 'unknown'}\n"
            f"- Inventory: {inv}\n"
            f"- Blockers: {bl}\n"
            f"- Frontier: {fr}\n"
            f"- Navigation hint: {self.navigation_hint or 'none'}\n"
            f"- Last output snippet: {self.last_output_snippet}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blockers": list(self.blockers),
            "inventory": list(self.inventory) if self.inventory else None,
            "current_room": self.current_room,
            "frontier": [{"room": r, "direction": d} for r, d in self.frontier],
            "navigation_hint": self.navigation_hint,
            "last_output_snippet": self.last_output_snippet,
        }


def build_tool_snapshot(state: GameStateTracker, map_agent: MapAgent) -> ToolSnapshot:
    """
    Build a ToolSnapshot from internal trackers.
    """

    last = (state.state.last_system_output or "").strip().replace("\n", " ")
    if len(last) > 200:
        last = last[:197] + "..."

    frontier = map_agent.graph.frontier()
    nav_hint = None
    # Use existing prompt addendum nav section as a hint for now.
    # This remains deterministic and avoids calling the LLM.
    if frontier and map_agent.graph.current_room:
        nav_hint = f"try {frontier[0][1]} (frontier)"

    return ToolSnapshot(
        blockers=list(state.state.blockers),
        inventory=state.state.inventory,
        current_room=map_agent.graph.current_room,
        frontier=frontier,
        navigation_hint=nav_hint,
        last_output_snippet=last,
    )

