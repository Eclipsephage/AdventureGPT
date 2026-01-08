"""
Prompt memory management for AdventureGPT.

Goals:
- Keep prompt context bounded (rolling window)
- Optionally summarize older history using the LLM to preserve key facts
- Provide a single system message that contains:
  - a running summary
  - the current game-state summary (provided externally)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm_client import LLMClient


@dataclass
class MemoryConfig:
    """
    Configuration for memory behavior.
    """

    # Keep the last N messages verbatim in context.
    window_messages: int = 30

    # When total messages exceed this, summarize older content.
    summarize_when_over: int = 60

    # Max output tokens for summarization calls.
    summary_max_output_tokens: int = 400

    # Hard cap prompt size by character count (rough safety budget).
    max_prompt_chars: int = 12000


class MemoryManager:
    """
    Maintains a rolling window + an optional LLM-generated summary.
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._summary: str = ""

    @property
    def summary(self) -> str:
        return self._summary

    def build_context(
        self,
        history: List[Dict[str, str]],
        *,
        llm: Optional[LLMClient],
        state_summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Return a bounded history suitable to pass to agents.

        If an LLM is available and history is long, update the summary by
        summarizing the "older" chunk.
        """

        if len(history) > self._config.summarize_when_over and llm is not None:
            # Summarize older content, keep only the last window.
            keep_n = self._config.window_messages
            older = history[:-keep_n]
            recent = history[-keep_n:]
            self._summary = self._summarize(older, recent, llm=llm)
            history_for_prompt = recent
        else:
            history_for_prompt = history[-self._config.window_messages :]

        # Inject a single system message with state + summary at the start.
        injected_parts: list[str] = []
        if state_summary:
            injected_parts.append(state_summary.strip())
        if self._summary:
            injected_parts.append("## Memory summary\n" + self._summary.strip())

        if injected_parts:
            injected = {"role": "system", "content": "\n\n".join(injected_parts) + "\n"}
            combined = [injected] + history_for_prompt
        else:
            combined = history_for_prompt

        # Prompt budgeting: ensure total content stays under a rough char cap.
        total = sum(len(m.get("content", "") or "") for m in combined)
        if total <= self._config.max_prompt_chars:
            return combined

        # Drop oldest messages until under budget (keeping injected message).
        kept = combined[:1] if combined and combined[0].get("role") == "system" else []
        tail = combined[1:] if kept else combined
        # keep last messages
        for m in reversed(tail):
            if sum(len(x.get("content", "") or "") for x in kept) + len(m.get("content", "") or "") > self._config.max_prompt_chars:
                continue
            kept.insert(1 if kept else 0, m)
        return kept

        return combined

    def _summarize(
        self,
        older: List[Dict[str, str]],
        recent: List[Dict[str, str]],
        *,
        llm: LLMClient,
    ) -> str:
        """
        Summarize older history into a compact running memory.
        """

        # Keep the prompt short: we summarize older, referencing recent only for context.
        prompt = (
            "You are maintaining a compact running summary of a Colossal Cave playthrough.\n"
            "Summarize the older conversation into bullet points capturing:\n"
            "- discovered facts about location/objects/inventory\n"
            "- goals attempted and results\n"
            "- any puzzles or blockers\n"
            "Keep it under ~200 words.\n"
            "Do not include any commands to run.\n"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]

        # Include prior summary (if any) to maintain continuity.
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": "Previous summary:\n" + self._summary.strip(),
                }
            )

        messages.append({"role": "system", "content": "Older events to summarize:"})
        messages.extend(older)
        messages.append({"role": "system", "content": "Recent context (do not re-summarize in detail):"})
        messages.extend(recent[-10:])

        return llm.respond(messages, max_output_tokens=self._config.summary_max_output_tokens)

