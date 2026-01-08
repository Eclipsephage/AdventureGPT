# AdventureGPT Roadmap (Upgrade + Feature Path)

This document is the canonical upgrade and feature-addition path for this repo.
It prioritizes **stability, modularity, and measurable progress** toward the long-term goal:
**reliably winning Colossal Cave Adventure** with autonomous agents.

## Guiding principles

- **Stabilize before expanding**: add tests/artifacts first so changes are safe.
- **Keep modules small**: prefer splitting by responsibility; avoid files > ~500 LOC.
- **Separate concerns**: game loop, task planning, command selection, memory/state, UI.
- **Make it observable**: every run should produce logs and replayable artifacts.
- **Make it measurable**: add benchmarks (win rate, steps, cost) to prevent regressions.

## Current architecture (today)

- **Runtime entrypoint**: `adventuregpt/__main__.py` runs the `adventure` engine loop.
- **LLM agents**: `adventuregpt/agent.py` provides:
  - task creation (`gametask_creation_agent`, `walkthrough_gametask_creation_agent`)
  - task prioritization (`prioritization_agent`)
  - command selection (`player_agent`)
  - completion check (`task_completion_agent`)
- **State**: mostly raw chat history; limited structured state.
- **Dependencies**: pinned to legacy `openai==0.27.7`.

## Phase 0 — Stabilize + make it safe to change (1–2 days)

### Goals

- Prevent common runtime failures.
- Establish a maintainable module layout.
- Make runs reproducible and debuggable.

### Work items

- **CLI robustness**
  - Validate CLI args and defaults (e.g., avoid `None` output path).
  - Handle empty task queue safely (`popleft()` can throw today).
- **Correctness fixes**
  - Fix typos/bugs that can break execution (e.g., token chunking increment typo).
  - Remove mutable default args (e.g., `initial_list: list = []`).
  - Make `__repr__` return a useful string (it currently returns `None`).
- **Refactor into modules (no behavior change)**
  - Suggested modules:
    - `adventuregpt/cli.py`
    - `adventuregpt/game_loop.py`
    - `adventuregpt/storage.py`
    - `adventuregpt/agents/{planner.py,prioritizer.py,player.py,completion.py}`
    - `adventuregpt/llm/{client.py,retry.py,prompts.py}`
- **Run artifacts**
  - Persist per-run outputs: `runs/<timestamp>/history.jsonl`, `commands.log`, `metrics.json`.
  - Add basic logging (levels, timestamps).
- **Minimal tests**
  - Unit tests for task parsing and task storage ordering/edge cases.

### Exit criteria

- A run completes without common crashes.
- Artifacts are written reliably for every run.
- Basic unit tests exist and pass locally/CI.

## Phase 1 — Modernize LLM integration (2–4 days)

### Goals

- Decouple “agents” from a single vendor SDK.
- Upgrade to a current OpenAI SDK cleanly.

### Work items

- **LLM interface**
  - Introduce a small internal interface (e.g., `LLMClient.chat(messages)->str`).
  - Keep prompts/response parsing outside the transport layer.
- **SDK upgrade**
  - Replace legacy `openai==0.27.7` usage with the current OpenAI Python SDK.
  - Normalize retry/backoff behavior in one place.
- **Configuration**
  - Central config for model/temperature/max tokens/retry settings (env + CLI flags).
  - Remove interactive API key prompting (fail fast with clear error).

### Exit criteria

- Agents work via the new abstraction.
- Changing model/temperature is a config change, not a code change.

## Phase 2 — Add structured state + memory (3–7 days)

### Goals

- Reduce “guessing” by giving the model structured state.
- Reduce prompt size and cost while improving success rate.

### Work items

- **State extractor**
  - Track: current location name (as text), visible exits, visible objects, inventory,
    last command + last response, and high-level “what changed”.
- **Prompt strategy**
  - Feed a compact state summary + a bounded history window.
  - Periodic summarization of long histories.
- **Command safety layer**
  - Enforce single command, 1–3 words (or normalize it into that format).
  - Detect loops and apply “escape hatches” (e.g., `help`, exploration moves).
- **Optional retrieval**
  - Start with a simple persistent store (SQLite) and add embeddings later if needed.

### Exit criteria

- Fewer invalid commands and fewer repetitive loops.
- Noticeably smaller prompts with same/better behavior.

## Phase 3 — Map/location agent + win-oriented planning (1–3 weeks)

### Goals

- Build an internal map of the world.
- Plan with pre-requisites and puzzle progress, not generic “explore” tasks.

### Work items

- **Map agent**
  - Maintain a graph of rooms and transitions; track “frontier” unexplored edges.
  - Provide map-based suggestions (“go back to X, then try Y”).
- **Win-oriented planner**
  - Replace generic task creation with state + map + inventory aware objectives.
  - Add a notion of “blocked tasks” and “unlock conditions”.
- **Internal tools**
  - Let the model query structured data (inventory, map, last N outputs) instead of
    re-parsing raw logs.

### Exit criteria

- The system produces a usable map and uses it to drive exploration.
- Tasks become sequential, actionable, and reflect puzzle progression.

## Phase 4 — TUI + replay (1–2 weeks)

### Goals

- Make development/debugging pleasant.
- Make runs shareable and “media friendly.”

### Work items

- **Curses/TUI**
  - Panes: game output, current objective, task queue, inventory/state, map view,
    and optionally a “debug” pane with the last prompt/response.
- **Replay mode**
  - Re-render a past run from `history.jsonl`.
  - Optional overlays: task changes, map evolution, cost/latency over time.

### Exit criteria

- You can replay any run deterministically from artifacts.
- The UI makes it easy to see what the agent believed and why.

## Phase 5 — Benchmarking + “win rate” as the north star (ongoing)

### Goals

- Make “Win the game” an engineering target with metrics.

### Work items

- **Eval harness**
  - Run N seeds, track: win rate, steps, wall-clock time, token/cost, loop count.
- **Regression tests**
  - Golden fixtures for known puzzles/rooms (state snapshot -> expected action class).
- **Cost controls**
  - Prompt budgeting, caching, and model tiering (cheap planner, stronger solver).

### Exit criteria

- A repeatable report (JSON/CSV) shows progress and catches regressions.

## Suggested near-term sequencing (recommended)

1. **Phase 0** (stability + tests + artifacts)
2. **Phase 1** (LLM abstraction + SDK upgrade)
3. **Phase 2** (structured state + memory)
4. **Phase 3** (map agent + win planner)
5. **Phase 4** (TUI + replay)
6. **Phase 5** (benchmarks forever)

