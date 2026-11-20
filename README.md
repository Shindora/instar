<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/instar-logo-dark.svg">
    <img src="assets/logo/instar-logo.svg" alt="instar" width="320">
  </picture>
</p>

<p align="center">Pre-release testing for AI agents, run locally and in CI.</p>

instar answers four questions about an agent before you ship it: does it still do the job, how much
load does it take, what happens when things break, and is this version better than the last one.
A project writes one small agentfile that calls its agent function; instar runs it, measures it and
judges it, without changing agent code.

**Status:** early development. Working: the LLM proxy (OpenAI, Anthropic and Gemini shapes,
seeded fault injection, spend cap), the run store, `instar run` / `instar doctor` with process
and inline executors, load shapes (stages, timed holds, ramps, open-loop arrivals), the load curve
and breaking point, baselines, `--fail-if` gates and Markdown/JSON/JUnit reports. Next: A/B,
telemetry, the web UI.

```bash
OPENAI_API_KEY=... uv run instar doctor -f examples/quickstart/agentfile.py
OPENAI_API_KEY=... uv run instar run -f examples/quickstart/agentfile.py -u 4 --repeat 2
OPENAI_API_KEY=... uv run instar run -f examples/quickstart/agentfile.py --stages 1,4,8 --fail-if "passed < 0.8"
```

## Development

```bash
uv sync                    # install
uv run ruff check .        # lint
uv run ruff format --check .
uv run ty check            # types
uv run pytest -q           # tests, offline
```
