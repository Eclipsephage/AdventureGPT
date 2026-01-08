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
            "steps": 0,
            "commands_sent": 0,
            "tasks_completed": 0,
            "errors": [],
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

