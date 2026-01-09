"""
Tests for step/time limit stopping behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from adventuregpt.game_loop import GameLoop
from adventuregpt.run_artifacts import RunArtifacts, RunArtifactsConfig


def test_game_loop_max_seconds_stops_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifacts = RunArtifacts(RunArtifactsConfig(base_run_dir=str(run_dir)))
    loop = GameLoop(
        walkthrough_path=None,
        artifacts=artifacts,
        dry_run=True,
        llm=None,
        max_seconds=0.0,
    )
    try:
        loop.run()
    finally:
        artifacts.close()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert any("max_seconds" in e for e in (metrics.get("errors") or []))

