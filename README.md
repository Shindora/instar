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
and breaking point, baselines, `--fail-if` gates, Markdown/JSON/JUnit reports, and A/B runs
(`--ab AGENT_VERSION=v1,v2`) with a paired comparison over tasks. Next: telemetry, the web UI.

```bash
OPENAI_API_KEY=... uv run instar doctor -f examples/quickstart/agentfile.py
OPENAI_API_KEY=... uv run instar run -f examples/quickstart/agentfile.py -u 4 --repeat 2
OPENAI_API_KEY=... uv run instar run -f examples/quickstart/agentfile.py --stages 1,4,8 --fail-if "passed < 0.8"
```

Tokens are always counted by instar's proxy; `--token-cap N` stops agent model calls after N tokens.
Dollar cost needs a price table (USD per million tokens); without one, cost reads "n/a" and says why:

```toml
[tool.instar.prices."gpt-4o-mini"]
input = 0.15
output = 0.60
cached = 0.075
```

## In CI

```yaml
- run: uv run instar run --profile smoke
- if: always()
  uses: actions/upload-artifact@v4
  with:
    name: instar-report
    path: |
      runs/*/*/report.md
      runs/*/*/report.json
      runs/*/*/junit.xml
```

Upload those three files only, never the whole run directory: `logs/` holds what the agent
printed and can contain secrets. Exit codes: `0` passed, `1` a gate failed, `2` bad usage or
configuration, `3` the test could not run. When `GITHUB_STEP_SUMMARY` is set, the Markdown report
is also appended to the job summary.

## Development

```bash
uv sync                    # install
uv run ruff check .        # lint
uv run ruff format --check .
uv run ty check            # types
uv run pytest -q -n auto   # tests, offline
```
