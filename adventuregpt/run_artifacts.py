"""
Run artifacts + logging utilities for AdventureGPT.

This module creates a per-run directory (default: ./runs/<timestamp>/) and writes:
- history.jsonl: JSON-lines of {role, content, ...}
- commands.log: plain-text commands sent to the game
- metrics.json: summary stats about the run

It is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import logging
import pprint
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_timestamp() -> str:
    """
    Return a filesystem-friendly UTC timestamp.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunArtifactsConfig:
    """
    Configuration for run artifact output locations.
    """

    base_run_dir: Optional[str] = None
    history_dump_path: Optional[str] = None


class RunArtifacts:
    """
    Manages per-run artifact files and metrics.
    """

    def __init__(self, config: RunArtifactsConfig):
        self._config = config

        base = Path(config.base_run_dir) if config.base_run_dir else Path("runs") / _utc_timestamp()
        self.run_dir = base
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history_jsonl_path = self.run_dir / "history.jsonl"
        self.commands_log_path = self.run_dir / "commands.log"
        self.metrics_path = self.run_dir / "metrics.json"

        # Optional pretty-printed dump. If not provided, default into run dir.
        self.history_dump_path = (
            Path(config.history_dump_path)
            if config.history_dump_path
            else self.run_dir / "history_dump.txt"
        )

        self._history_fp = self.history_jsonl_path.open("a", encoding="utf-8")
        self._commands_fp = self.commands_log_path.open("a", encoding="utf-8")
        self.log_path = self.run_dir / "run.log"

        self.metrics: Dict[str, Any] = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "ended_at_utc": None,
            "walkthrough_enabled": False,
            "dry_run": False,
            "llm": {
                "model": None,
                "temperature": None,
                "max_output_tokens": None,
                "cost_per_1k_input_usd": None,
                "cost_per_1k_output_usd": None,
            },
            "tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "estimated_cost_usd": 0.0,
            "map": {
                "rooms_discovered": 0,
                "transitions_recorded": 0,
                "frontier_size": 0,
            },
            "outcome": {
                "is_game_over": False,
                "is_victory": False,
                "score": None,
                "max_score": None,
                "reason": None,
            },
            "steps": 0,
            "commands_sent": 0,
            "tasks_completed": 0,
            "errors": [],
        }

    def set_llm_config(
        self,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        dry_run: bool,
    ) -> None:
        """
        Record LLM configuration for this run.
        """

        self.metrics["dry_run"] = bool(dry_run)
        self.metrics["llm"] = {
            "model": model,
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
            "cost_per_1k_input_usd": None,
            "cost_per_1k_output_usd": None,
        }

    def set_cost_rates(
        self,
        *,
        cost_per_1k_input_usd: Optional[float],
        cost_per_1k_output_usd: Optional[float],
    ) -> None:
        """
        Configure optional cost estimation rates.
        """

        self.metrics["llm"]["cost_per_1k_input_usd"] = (
            float(cost_per_1k_input_usd) if cost_per_1k_input_usd is not None else None
        )
        self.metrics["llm"]["cost_per_1k_output_usd"] = (
            float(cost_per_1k_output_usd) if cost_per_1k_output_usd is not None else None
        )

    def add_usage(self, usage: Dict[str, Any]) -> None:
        """
        Add token usage from an LLM call and update estimated cost if configured.

        Expected keys:
            - input_tokens
            - output_tokens
            - total_tokens
        """

        tokens = self.metrics.get("tokens") or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        tot_tok = int(usage.get("total_tokens", 0) or 0)

        tokens["input_tokens"] = int(tokens.get("input_tokens", 0) or 0) + in_tok
        tokens["output_tokens"] = int(tokens.get("output_tokens", 0) or 0) + out_tok
        tokens["total_tokens"] = int(tokens.get("total_tokens", 0) or 0) + tot_tok
        self.metrics["tokens"] = tokens

        rate_in = self.metrics.get("llm", {}).get("cost_per_1k_input_usd")
        rate_out = self.metrics.get("llm", {}).get("cost_per_1k_output_usd")
        if rate_in is not None and rate_out is not None:
            self.metrics["estimated_cost_usd"] = float(self.metrics.get("estimated_cost_usd", 0.0) or 0.0) + (
                (in_tok / 1000.0) * float(rate_in) + (out_tok / 1000.0) * float(rate_out)
            )

    def set_map_stats(self, *, rooms_discovered: int, transitions_recorded: int, frontier_size: int) -> None:
        """
        Record map statistics for the run.
        """

        self.metrics["map"] = {
            "rooms_discovered": int(rooms_discovered),
            "transitions_recorded": int(transitions_recorded),
            "frontier_size": int(frontier_size),
        }

    def set_outcome(
        self,
        *,
        is_game_over: bool,
        is_victory: bool,
        score: Optional[int] = None,
        max_score: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Record detected game outcome.
        """

        self.metrics["outcome"] = {
            "is_game_over": bool(is_game_over),
            "is_victory": bool(is_victory),
            "score": int(score) if score is not None else None,
            "max_score": int(max_score) if max_score is not None else None,
            "reason": reason,
        }

    def configure_logging(self, level: int = logging.INFO) -> None:
        """
        Configure basic file logging into the run directory.

        This is intentionally simple: one file handler + a readable format.
        """

        root = logging.getLogger()
        root.setLevel(level)

        # Avoid double-handlers if called more than once.
        if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(self.log_path) for h in root.handlers):
            return

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)

    def record_error(self, message: str) -> None:
        """
        Record a non-fatal error message in metrics.
        """

        self.metrics["errors"].append(message)

    def set_walkthrough_enabled(self, enabled: bool) -> None:
        """
        Record whether walkthrough mode was enabled.
        """

        self.metrics["walkthrough_enabled"] = bool(enabled)

    def increment(self, key: str, amount: int = 1) -> None:
        """
        Increment a numeric metric counter.
        """

        self.metrics[key] = int(self.metrics.get(key, 0)) + int(amount)

    def write_history_event(self, event: Dict[str, Any]) -> None:
        """
        Append a history event as JSONL.
        """

        self._history_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._history_fp.flush()

    def write_command(self, command: str) -> None:
        """
        Append a command to the command log.
        """

        self._commands_fp.write(command.rstrip("\n") + "\n")
        self._commands_fp.flush()

    def write_history_dump(self, history: list[Dict[str, Any]]) -> None:
        """
        Write a pretty-printed history dump for human reading.
        """

        self.history_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_dump_path.open("w", encoding="utf-8") as fp:
            pprint.pprint(history, stream=fp)

    def write_json(self, relative_name: str, data: Any) -> None:
        """
        Write a JSON file under the run directory.
        """

        path = self.run_dir / relative_name
        with path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def finalize(self) -> None:
        """
        Write metrics.json.
        """

        self.metrics["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(self.metrics, fp, ensure_ascii=False, indent=2)

    def close(self) -> None:
        """
        Finalize and close file handles.
        """

        try:
            self.finalize()
        finally:
            try:
                self._history_fp.close()
            finally:
                self._commands_fp.close()

