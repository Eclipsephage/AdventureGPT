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
import time
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
    p.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Seconds to sleep between each output chunk (default: 0).",
    )
    p.add_argument(
        "--overlay_map_stats",
        action="store_true",
        help="Print map statistics at the end if map.json is present.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    run_dir = Path(ns.run_dir)
    history = run_dir / "history.jsonl"
    commands = run_dir / "commands.log"
    map_path = run_dir / "map.json"

    if not history.exists():
        raise FileNotFoundError(f"Missing {history}")
    if not commands.exists():
        raise FileNotFoundError(f"Missing {commands}")

    outputs = load_system_outputs(history)
    cmds = load_commands(commands)
    text = replay(outputs, cmds)
    if ns.speed and ns.speed > 0:
        # Print progressively by system output boundaries.
        # This is a simple, deterministic cadence rather than character-level replay.
        for chunk in text.splitlines(True):
            print(chunk, end="")
            # only sleep on line breaks to keep it responsive
            if chunk.endswith("\n"):
                time.sleep(float(ns.speed))
    else:
        print(text, end="")

    if ns.overlay_map_stats and map_path.exists():
        try:
            data = json.loads(map_path.read_text(encoding="utf-8"))
            rooms = len((data.get("rooms") or {}).keys())
            edges = len(data.get("edges") or [])
            frontier = len(data.get("frontier") or [])
            print("\n\n--- MAP STATS ---")
            print(f"Rooms: {rooms}  Edges: {edges}  Frontier: {frontier}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

