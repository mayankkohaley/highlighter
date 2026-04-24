# AGENTS.md

Guidance for any agent (Claude, Codex, humans, etc.) working in this repo.

## Project

`highlighter` is an experiment. See [README.md](README.md) for the rationale
and pipeline sketch. Stages 0 (normalize), 1 (expand), 2 (chunk), and 3
(extract) are implemented. Stages 4 (consolidate) and 5 (synthesize) TBD.

## Setup

```
uv sync
```

## Dev loop

```
uv run pytest          # all tests must stay green
uv run ruff check .    # must stay clean
```

## Commit style

- **Small, focused commits.** One concern per commit. Implementation and the tests
  that drove it land together.
- **Conventional Commits** (`feat:`, `fix:`, `test:`, `refactor:`,
  `docs:`, `chore:`, `build:`, …).
- **Subject line under 80 characters.** Use the body for detail.
- Propose a commit breakdown before running `git commit` on non-trivial
  work; wait for the user's go-ahead.

## Scope discipline

Stage 0 is **markdown-only** and has **no cache layer** (scoped out by
the user). Don't pull features from later stages into earlier ones
unless a test explicitly demands it.

Stage 3 uses [pydantic-ai](https://ai.pydantic.dev) for model-agnostic
LLM calls. Tests use `FunctionModel` via `agent.override(...)` to stub
the LLM boundary — this is a legitimate mock per /tdd rules (external
API, system boundary, dependency-injected). Internal logic (verification,
citation mapping) is never stubbed.
