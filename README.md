# highlighter

> _Glow big or go home._

> **Status: experiment.** Interfaces, scope, and behavior will change
> without notice.

A researcher reading a dense academic paper — or a student working through
a textbook — reaches for a neon ink pen and marks the passages that answer
the question they're holding in their head. They don't paraphrase on the
fly; they highlight the verbatim text and come back to synthesize once the
full picture is on the page.

`highlighter` is the digital counterpart. It reads a document with a query
in mind, pulls out verbatim excerpts as it goes, and only then synthesizes
an answer grounded in those highlights. Every claim in the final answer
points back to a quoted span in the source — no paraphrasing, no
invention.

We're trading latency for higher accuracy, for now. The sweet spot is
*deep* questions over *single* documents where the answer has to be
assembled from many passages rather than retrieved as one. Markdown input
only.

## Usage

Ask a question, get cited excerpts:

```
uv run python -m highlighter <markdown-file> -q "your question"
```

Add `--synthesize` to also get a short grounded answer that cites the
excerpts by number.

| Arg / flag         | Required | Default | Notes                                     |
| ------------------ | -------- | ------- | ----------------------------------------- |
| `<markdown-file>`  | yes      | —       | Path to a markdown document.              |
| `-q`, `--question` | yes      | —       | The question to ask.                      |
| `--chunk-size`     | no       | `2000`  | Tokens per chunk.                         |
| `--chunk-overlap`  | no       | `200`   | Token overlap between consecutive chunks. |
| `--synthesize`     | no       | off     | Run the final LLM synthesis step.         |

Requires an API key for the configured provider (default:
`anthropic:claude-haiku-4-5-20251001`, so `ANTHROPIC_API_KEY` must be
set).

## Development

```
uv sync
uv run pytest
uv run ruff check .
```
