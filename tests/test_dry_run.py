"""
Smoke test for the CLI dry-run path.

This validates that the game engine can start and that run artifacts are written
without calling OpenAI.
"""

from __future__ import annotations

from pathlib import Path

from adventuregpt.cli import main


def test_cli_dry_run_writes_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code = main(["--dry_run", "--run_dir", str(run_dir)])
    assert code == 0

    # Core artifacts
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "history_dump.txt").exists()
    assert (run_dir / "metrics.json").exists()

    # Commands should be recorded (one command in dry run)
    assert (run_dir / "commands.log").exists()

