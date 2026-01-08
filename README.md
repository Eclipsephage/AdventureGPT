# AdventureGPT

An set of autonomous agents designed to play the 1977 game
ADVENTURE or [Colossal Cave Adventure](https://en.m.wikipedia.org/wiki/Colossal_Cave_Adventure). Currently utilizing OpenAI
python SDK,the eventual goal is to switch to LangChain.
The code base here is based off code from the following repos:

* [python-adventure](https://github.com/brandon-rhodes/python-adventure)
* [BabyAGI](https://github.com/yoheinakajima/babyagi)

That said, code from other repos has been heavily modified and all modifications are licensed under the Apache 2.0 License.

Currently, the only requirement is an OpenAI API key and to have it set as the `OPENAI_API_KEY` environment variable.

## Running

Run the code by cloning the repository, navigate to the cloned repository, and run the following commands:

```bash
python -m pip install -r requirements.txt
python -m adventuregpt
```
Add a `--help` flag to see the command line arguments.

### Configuration

- **OPENAI_API_KEY**: required for non-`--dry_run` runs.
- **ADVENTUREGPT_MODEL** / `--model`: choose the model (default `gpt-4o-mini`).
- **ADVENTUREGPT_PLANNER_MODEL** / `--planner_model`: optional separate model for the win-planner (defaults to `--model`).
- **ADVENTUREGPT_TEMPERATURE** / `--temperature`: sampling temperature (default `0.0`).
- **ADVENTUREGPT_MAX_OUTPUT_TOKENS** / `--max_output_tokens`: per-call output cap (default `2000`).

### State, memory, and command safety

AdventureGPT now includes:

- **Heuristic state tracking** (`adventuregpt/state.py`): extracts a compact state summary (turn, last command, inventory hints, mentioned directions, recent output snippet).
- **Bounded prompt memory** (`adventuregpt/memory.py`): keeps a rolling window of messages and (when an LLM is available) periodically summarizes older history into a short memory summary.
- **Command normalization + loop breaking** (`adventuregpt/command_safety.py`): converts model output into a single short command and attempts to break out of simple repetition loops.
- **Map tracking** (`adventuregpt/map_agent.py`, `adventuregpt/map_graph.py`): builds a room/exit graph during play, tracks frontier exits, and writes `map.json` into each run directory.
- **Deterministic navigation for map objectives** (`adventuregpt/navigation_tasks.py`): if an objective matches patterns like `Explore east from <room>`, AdventureGPT will route to `<room>` using known map edges before exploring.

### Smoke test (no OpenAI calls)

```bash
python -m adventuregpt --dry_run
```

### TUI (curses)

```bash
python -m adventuregpt.tui
```

You can also combine with `--dry_run` to validate UI wiring without OpenAI.

### Replay a run

```bash
python -m adventuregpt.replay --run_dir runs/<timestamp>
```

Replay with pacing and map stats:

```bash
python -m adventuregpt.replay --run_dir runs/<timestamp> --speed 0.05 --overlay_map_stats
```

### Evaluation harness

Run multiple sessions and produce an aggregate report:

```bash
python -m adventuregpt.eval --runs 10 --dry_run
```

This writes `report.json` and `report.csv` under `./eval_runs/<timestamp>/`.

#### Eval matrix (models x temperatures)

```bash
python -m adventuregpt.eval --dry_run --runs 2 --max_steps 1 \
  --models gpt-4o-mini,gpt-4o-mini \
  --temperatures 0.0,0.2
```

This creates `matrix_report.json` at the root and per-combo `report.json`/`report.csv` under subdirectories.

#### Trend report over multiple eval runs

```bash
python -m adventuregpt.trend --eval_root eval_runs --out_dir eval_trends
```

#### Non-dry-run eval (real model calls)

```bash
export OPENAI_API_KEY="..."
python -m adventuregpt.eval --runs 3 --max_steps 200 --max_seconds 60 --model gpt-4o-mini
```

#### Optional cost estimation

If you want a rough USD estimate in the reports, provide token rates:

```bash
python -m adventuregpt.eval --runs 3 --max_steps 200 \
  --cost_per_1k_input_usd 0.15 \
  --cost_per_1k_output_usd 0.60
```

#### Win rate / score reporting

Eval summaries include `wins`, `win_rate`, and score fields when the game outputs a score line (heuristic detection).

## TODO

Here is a list of eventual goals for the project:

* Switch to LangChain
* Add map/location agent
* Win the game
* Add a curses style UI for displaying tasks and prompts while showing gameplay in its own pane
* Better utilization of context/memory in prompts, maybe storing results in a vector DB
* Add a "replay" feature to take a history dump from a run and replay the main game text (for media creation)

## Roadmap

See `ROADMAP.md` for a phased upgrade + feature-addition plan.

## Contributing

This project is a playground for me to learn more about prompt engineering and play with OpenAI's models. That said, I am interested in pushing this to the absolute limit of what is possible. If you want to contribute, make a fork and create pull requests. I will do my best to be a good steward of the project and comment on pull requests within a timely manner.
