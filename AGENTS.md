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

## Development discipline

- **TDD, strictly.** Write one failing test. Run pytest and see it red.
  Implement the minimum to go green. Run ruff. Move to the next test.
  Never implement behavior that no failing test demands.
- **No mocks.** Use real objects and real I/O scoped to `tmp_path` or
  constructor-injected dependencies. `capsys` and `monkeypatch.chdir` are
  fine — they redirect real I/O, they don't mock.
- **Pydantic for types**, not `@dataclass`.
- **No speculative abstractions.** Don't introduce features, fallbacks, or
  configurability that no test or user need has demanded.

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
