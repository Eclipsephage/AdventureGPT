"""
AdventureGPT CLI entrypoint.

Usage:
    python -m adventuregpt [--walkthrough_path PATH] [--output_path PATH] [--run_dir DIR]

This module centralizes argument parsing and delegates execution to the game loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from .game_loop import GameLoop
from .run_artifacts import RunArtifacts, RunArtifactsConfig


@dataclass(frozen=True)
class CLIArgs:
    """
    Parsed CLI arguments for AdventureGPT.
    """

    walkthrough_path: Optional[str]
    output_path: Optional[str]
    run_dir: Optional[str]
    dry_run: bool


def parse_args(argv: Optional[list[str]] = None) -> CLIArgs:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        prog="AdventureGPT",
        description="The game ADVENTURE played by ChatGPT",
    )
    parser.add_argument(
        "-w",
        "--walkthrough_path",
        default=None,
        help="Optional path to a walkthrough text file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=None,
        help=(
            "Optional path to write a pretty-printed history dump. "
            "If omitted, the history is written under the run directory."
        ),
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help=(
            "Optional directory to store run artifacts. "
            "If omitted, a timestamped directory under ./runs is created."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run without calling OpenAI. Uses a minimal task + a single safe command "
            "to validate the game loop and artifact writing."
        ),
    )
    ns = parser.parse_args(argv)
    return CLIArgs(
        walkthrough_path=ns.walkthrough_path,
        output_path=ns.output_path,
        run_dir=ns.run_dir,
        dry_run=bool(ns.dry_run),
    )


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI main function.
    """

    args = parse_args(argv)

    artifacts = RunArtifacts(
        RunArtifactsConfig(
            base_run_dir=args.run_dir,
            history_dump_path=args.output_path,
        )
    )
    artifacts.configure_logging()

    loop = GameLoop(
        walkthrough_path=args.walkthrough_path,
        artifacts=artifacts,
        dry_run=args.dry_run,
    )

    try:
        loop.run()
    except EOFError:
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        artifacts.close()
    return 0

