"""
AdventureGPT CLI entrypoint.

Usage:
    python -m adventuregpt [--walkthrough_path PATH] [--output_path PATH] [--run_dir DIR]

This module centralizes argument parsing and delegates execution to the game loop.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

from .game_loop import GameLoop
from .run_artifacts import RunArtifacts, RunArtifactsConfig
from .llm_client import CappedLLMClient, CachedLLMClient, OpenAIResponsesClient, OpenAIResponsesConfig


@dataclass(frozen=True)
class CLIArgs:
    """
    Parsed CLI arguments for AdventureGPT.
    """

    walkthrough_path: Optional[str]
    output_path: Optional[str]
    run_dir: Optional[str]
    dry_run: bool
    model: str
    temperature: float
    max_output_tokens: int
    planner_model: Optional[str]


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
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI model name. If omitted, uses ADVENTUREGPT_MODEL or a sane default."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature. If omitted, uses ADVENTUREGPT_TEMPERATURE or 0.0."
        ),
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=None,
        help=(
            "Global cap for max output tokens per LLM call. "
            "If omitted, uses ADVENTUREGPT_MAX_OUTPUT_TOKENS or 2000."
        ),
    )
    parser.add_argument(
        "--planner_model",
        default=None,
        help="Optional separate model for the win-planner (defaults to --model).",
    )
    ns = parser.parse_args(argv)

    model = ns.model or os.environ.get("ADVENTUREGPT_MODEL") or "gpt-4o-mini"
    temperature = (
        ns.temperature
        if ns.temperature is not None
        else float(os.environ.get("ADVENTUREGPT_TEMPERATURE", "0.0"))
    )
    max_output_tokens = (
        ns.max_output_tokens
        if ns.max_output_tokens is not None
        else int(os.environ.get("ADVENTUREGPT_MAX_OUTPUT_TOKENS", "2000"))
    )

    return CLIArgs(
        walkthrough_path=ns.walkthrough_path,
        output_path=ns.output_path,
        run_dir=ns.run_dir,
        dry_run=bool(ns.dry_run),
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        planner_model=ns.planner_model or os.environ.get("ADVENTUREGPT_PLANNER_MODEL"),
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
    artifacts.set_llm_config(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        dry_run=args.dry_run,
    )

    llm = None
    planner_llm = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it in your environment to run AdventureGPT."
            )
        base_llm = OpenAIResponsesClient(
            api_key=api_key,
            config=OpenAIResponsesConfig(model=args.model, temperature=args.temperature),
            on_usage=artifacts.add_usage,
        )
        llm = CachedLLMClient(CappedLLMClient(base_llm, max_output_tokens_cap=args.max_output_tokens))

        planner_model = args.planner_model or args.model
        if planner_model == args.model:
            planner_llm = llm
        else:
            planner_base = OpenAIResponsesClient(
                api_key=api_key,
                config=OpenAIResponsesConfig(model=planner_model, temperature=args.temperature),
                on_usage=artifacts.add_usage,
            )
            planner_llm = CachedLLMClient(CappedLLMClient(planner_base, max_output_tokens_cap=args.max_output_tokens))

    loop = GameLoop(
        walkthrough_path=args.walkthrough_path,
        artifacts=artifacts,
        dry_run=args.dry_run,
        llm=llm,
        planner_llm=planner_llm,
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

