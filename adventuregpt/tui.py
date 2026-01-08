"""
Curses-based TUI for AdventureGPT.

Usage:
    python -m adventuregpt.tui [--dry_run] [--model MODEL] ...

This is a Phase 4 feature. It uses the same GameLoop but renders:
- Game output (scrolling)
- Current objective + task list
- State + map summaries

Note: curses is stdlib, so this should run without extra deps.
"""

from __future__ import annotations

import argparse
import os
import curses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .game_loop import GameLoop
from .llm_client import CappedLLMClient, CachedLLMClient, OpenAIResponsesClient, OpenAIResponsesConfig
from .run_artifacts import RunArtifacts, RunArtifactsConfig
from .ui_hooks import UIHooks


@dataclass(frozen=True)
class TUIArgs:
    dry_run: bool
    run_dir: Optional[str]
    model: str
    temperature: float
    max_output_tokens: int


def parse_args(argv: Optional[List[str]] = None) -> TUIArgs:
    p = argparse.ArgumentParser(prog="adventuregpt-tui", description="AdventureGPT curses UI")
    p.add_argument("--dry_run", action="store_true", help="Run without calling OpenAI.")
    p.add_argument("--run_dir", default=None, help="Run directory for artifacts.")
    p.add_argument("--model", default=None, help="Model (default: env or gpt-4o-mini).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature (default: env or 0.0).")
    p.add_argument("--max_output_tokens", type=int, default=None, help="Max output tokens per call.")
    ns = p.parse_args(argv)

    model = ns.model or os.environ.get("ADVENTUREGPT_MODEL") or "gpt-4o-mini"
    temperature = ns.temperature if ns.temperature is not None else float(os.environ.get("ADVENTUREGPT_TEMPERATURE", "0.0"))
    max_output_tokens = ns.max_output_tokens if ns.max_output_tokens is not None else int(os.environ.get("ADVENTUREGPT_MAX_OUTPUT_TOKENS", "2000"))

    return TUIArgs(
        dry_run=bool(ns.dry_run),
        run_dir=ns.run_dir,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


class CursesUI(UIHooks):
    def __init__(self, stdscr: "curses._CursesWindow"):
        self.stdscr = stdscr
        self.game_lines: List[str] = []
        self.task_list: str = ""
        self.status: str = ""
        self.last_command: str = ""
        self.objective: str = ""
        self.state_summary: str = ""
        self.map_summary: str = ""

    def on_task_list(self, text: str) -> None:
        self.task_list = text or ""
        self.render()

    def on_output(self, text: str) -> None:
        # Append output to buffer, split into lines.
        for part in (text or "").splitlines(True):
            if part.endswith("\n"):
                self.game_lines.append(part.rstrip("\n"))
            else:
                if self.game_lines:
                    self.game_lines[-1] = self.game_lines[-1] + part
                else:
                    self.game_lines.append(part)
        self.game_lines = self.game_lines[-2000:]  # cap memory
        self.render()

    def on_command(self, command: str) -> None:
        self.last_command = command
        self.game_lines.append(f"> {command}")
        self.render()

    def on_status(self, text: str) -> None:
        self.status = text
        self.render()

    def on_info(self, *, objective: str, state_summary: str, map_summary: str) -> None:
        self.objective = objective or ""
        self.state_summary = state_summary or ""
        self.map_summary = map_summary or ""
        self.render()

    def render(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        left_w = max(20, int(w * 0.65))
        right_w = w - left_w
        top_h = h - 3

        # Left pane: game output
        left = self.stdscr.derwin(top_h, left_w, 0, 0)
        left.box()
        left.addnstr(0, 2, " Game ", left_w - 4)
        visible = self.game_lines[-(top_h - 2) :]
        for i, line in enumerate(visible):
            left.addnstr(1 + i, 1, line, left_w - 2)

        # Right pane split: tasks + info
        right = self.stdscr.derwin(top_h, right_w, 0, left_w)
        right.box()
        right.addnstr(0, 2, " Tasks / Info ", right_w - 4)
        split_y = max(6, int(top_h * 0.55))

        # Tasks section
        right.addnstr(1, 1, f"Objective: {self.objective}", right_w - 2)
        task_lines = (self.task_list or "").splitlines()
        for i, line in enumerate(task_lines[: split_y - 3]):
            right.addnstr(2 + i, 1, line, right_w - 2)

        # Info section
        info_start = split_y
        right.addnstr(info_start, 1, f"Last cmd: {self.last_command}", right_w - 2)
        info_lines = []
        info_lines.extend((self.state_summary or "").splitlines())
        info_lines.append("")
        info_lines.extend((self.map_summary or "").splitlines())
        for i, line in enumerate(info_lines[: top_h - info_start - 2]):
            right.addnstr(info_start + 1 + i, 1, line, right_w - 2)

        # Bottom status bar
        bottom = self.stdscr.derwin(3, w, top_h, 0)
        bottom.box()
        bottom.addnstr(1, 1, self.status or "", w - 2)

        self.stdscr.noutrefresh()
        curses.doupdate()


def _run(stdscr: "curses._CursesWindow", args: TUIArgs) -> int:
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    ui = CursesUI(stdscr)

    artifacts = RunArtifacts(RunArtifactsConfig(base_run_dir=args.run_dir))
    artifacts.configure_logging()
    artifacts.set_llm_config(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        dry_run=args.dry_run,
    )

    llm = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for non-dry-run TUI).")
        base = OpenAIResponsesClient(
            api_key=api_key,
            config=OpenAIResponsesConfig(model=args.model, temperature=args.temperature),
            on_usage=artifacts.add_usage,
        )
        llm = CachedLLMClient(CappedLLMClient(base, max_output_tokens_cap=args.max_output_tokens))

    loop = GameLoop(
        walkthrough_path=None,
        artifacts=artifacts,
        dry_run=args.dry_run,
        llm=llm,
        ui=ui,
    )

    try:
        loop.run()
        ui.on_status(f"Run complete. Artifacts in: {artifacts.run_dir}")
        stdscr.getch()
        return 0
    finally:
        artifacts.close()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return curses.wrapper(lambda stdscr: _run(stdscr, args))


if __name__ == "__main__":
    raise SystemExit(main())

