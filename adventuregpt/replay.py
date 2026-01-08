"""
Replay a previous AdventureGPT run.

This replays gameplay text using artifacts written by the game loop:
- history.jsonl (system outputs)
- commands.log  (normalized commands sent)

Usage:
    python -m adventuregpt.replay --run_dir runs/20260108_123456
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def load_system_outputs(history_jsonl: Path) -> List[str]:
    outputs: List[str] = []
    for line in history_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        evt = json.loads(line)
        if evt.get("role") == "system":
            outputs.append(str(evt.get("content", "")))
    return outputs


def load_commands(commands_log: Path) -> List[str]:
    return [l.strip() for l in commands_log.read_text(encoding="utf-8").splitlines() if l.strip()]


def replay(outputs: List[str], commands: List[str]) -> str:
    """
    Build a replay string by interleaving outputs and commands.
    """

    parts: List[str] = []
    if outputs:
        parts.append(outputs[0])
    # For each command, print it then the subsequent output (if present).
    for i, cmd in enumerate(commands):
        parts.append(f"> {cmd}\n")
        if i + 1 < len(outputs):
            parts.append(outputs[i + 1])
    return "".join(parts)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adventuregpt-replay", description="Replay an AdventureGPT run")
    p.add_argument("--run_dir", required=True, help="Run directory containing artifacts.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    run_dir = Path(ns.run_dir)
    history = run_dir / "history.jsonl"
    commands = run_dir / "commands.log"

    if not history.exists():
        raise FileNotFoundError(f"Missing {history}")
    if not commands.exists():
        raise FileNotFoundError(f"Missing {commands}")

    out = replay(load_system_outputs(history), load_commands(commands))
    print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

