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

**Status:** early development. The LLM proxy (OpenAI, Anthropic and Gemini shapes, seeded fault
injection, spend cap) and the run store work; the runner, analysis, CI gate and web UI are next.

## Development

```bash
uv sync                    # install
uv run ruff check .        # lint
uv run ruff format --check .
uv run ty check            # types
uv run pytest -q           # tests, offline
```
