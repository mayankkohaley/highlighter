# AGENTS.md

Guidance for any agent (Claude, Codex, humans, etc.) working in this repo.

## Project

`highlighter` is an experiment. See [README.md](README.md) for the rationale
and pipeline sketch. Only Stage 0 (normalize) is implemented.

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

- **Small, focused commits.** One concern per commit. Impl and the tests
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
