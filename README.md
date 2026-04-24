# highlighter

> **Status: experiment.** This is a sketch, not a tool. Interfaces, scope, and
> behavior will change without notice.

## Rationale

When a careful researcher reads a paper, they highlight the passages that
matter for the question they're holding in their head. They don't paraphrase,
they don't summarize on the fly — they mark the verbatim text and come back
to synthesize at the end.

`highlighter` is an experiment in doing the same thing with an LLM:
methodically reading a document chunk by chunk with a query in mind, pulling
out verbatim excerpts as it goes, and only synthesizing once it has the full
set of highlights. The pitch versus embeddings-based RAG is faithfulness and
verifiability — every claim in the final answer points back to a quoted span
in the source.

The intended sweet spot is *deep* questions over *single* documents, where
the question requires assembling evidence from many parts of the document
rather than retrieving one passage.

## Pipeline (planned)

0. **Normalize** — load the document, normalize whitespace, parse the
   heading hierarchy. *(implemented)*
1. **Query expansion** — turn the user's question into sub-questions
   and a relevance rubric. *(implemented)*
2. **Chunk** — split the normalized text into overlapping chunks that
   prefer paragraph/heading boundaries. *(implemented)*
3. **Per-chunk extraction** — for each chunk, ask an LLM to return verbatim
   spans relevant to the query. Verify each returned span actually appears
   in the chunk. *(implemented)*
4. **Consolidate** — dedupe overlapping excerpts, merge adjacent spans.
5. **Synthesize** — final LLM call answers the question using only the
   consolidated excerpts, with citations.

Markdown input only — PDF support will slot in behind the same interface
later.

## Command-line usage

### `highlighter` — ask a question, get cited excerpts

Runs the full pipeline (normalize → expand → chunk → extract) against a
markdown file and prints the question, LLM-generated sub-questions and
rubric, and each verified verbatim excerpt with its citation.

```
uv run python -m highlighter <markdown-file> -q "your question"
```

| Arg / flag          | Required | Default                               | Notes                                          |
| ------------------- | -------- | ------------------------------------- | ---------------------------------------------- |
| `<markdown-file>`   | yes      | —                                     | Positional. Path to a markdown document.       |
| `-q`, `--question`  | yes      | —                                     | The question to ask.                           |
| `--chunk-size`      | no       | `2000`                                | Tokens per chunk.                              |
| `--chunk-overlap`   | no       | `200`                                 | Token overlap between consecutive chunks.      |

Requires an API key for the configured provider (default:
`anthropic:claude-haiku-4-5-20251001`, so `ANTHROPIC_API_KEY` must be
set).

Example:

```
$ uv run python -m highlighter tmp/agentcore-get-started-cli.md \
    -q "What are the prerequisites for deploying an agent?"

Question: What are the prerequisites for deploying an agent?

Sub-questions:
  - Which AWS services are required?
  - What IAM permissions does the user need?
  - Which language runtimes must be installed?

Rubric: Useful excerpts name specific software, permissions, accounts, or versions required.

Excerpts (3):

[1] L14-L21  Get started with Amazon Bedrock AgentCore > Prerequisites
    sub-q: Which language runtimes must be installed?   confidence: 0.95
    > Node.js 20 or later. The AgentCore CLI is distributed as an npm package.
...
```

Exit codes: `0` on success; `2` on a usage error (missing required argument).

### `highlighter.normalize`

```
uv run python -m highlighter.normalize <markdown-file>
```

| Arg              | Required | Description                                   |
| ---------------- | -------- | --------------------------------------------- |
| `<markdown-file>` | yes      | Path to a markdown document to normalize.     |

No flags. Reads the file, normalizes whitespace and line endings,
parses the heading hierarchy, and prints:

- the source path
- the sha256 of the raw bytes
- the number of parsed sections
- an indented heading tree, one heading per line, with 1-indexed line
  ranges (`L{start}-L{end}`)

Example:

```
$ uv run python -m highlighter.normalize tmp/agentcore-get-started-cli.md
source:  tmp/agentcore-get-started-cli.md
sha256:  ceed3dcc0035ab6da706681e91d3b5201b74194366d7b7b114752f0f043a80b6
sections: 13

# Get started with Amazon Bedrock AgentCore  (L3-209)
  ## Prerequisites  (L14-21)
  ## Step 1: Install the AgentCore CLI  (L22-47)
    ### Opt into the preview channel  (L37-47)
  ...
```

Exit codes: `0` on success, `1` if no path argument is given.

## Stage 0 — normalize (programmatic)

```python
from highlighter.normalize import normalize

doc = normalize("path/to/doc.md")  # same behavior as the CLI, callable form
doc.text                          # normalized markdown (LF, trimmed)
doc.content_hash                  # sha256 of the original bytes
doc.sections                      # list[Section] with (level, title, line_start, line_end)
doc.section_path_for_line(42)     # heading hierarchy at a given 1-indexed line
```

## Stage 2 — chunk

Wraps [Chonkie](https://docs.chonkie.ai)'s `RecursiveChunker` (markdown
recipe) and `OverlapRefinery` to produce boundary-aware chunks with a
configurable token overlap (tokenizer: `cl100k_base`).

```python
from highlighter.normalize import normalize
from highlighter.chunk import chunk_document

doc = normalize("path/to/doc.md")
chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

for c in chunks:
    c.text            # chunk text, including prepended overlap from the prior chunk
    c.line_start      # 1-indexed start line in doc.text
    c.line_end        # 1-indexed end line, inclusive
    c.section_path    # heading hierarchy at line_start
```

## Stage 1 — expand

One LLM call that turns a raw user question into a structured `Query`
with 3–7 sub-questions and a one-to-two sentence relevance rubric. The
raw question is passed through verbatim; only the sub-questions and
rubric come from the LLM.

```python
from highlighter.expand import expand_query

query = expand_query("What are the prerequisites for deploying an AgentCore agent?")
query.question       # preserved verbatim
query.sub_questions  # list[str] filled by the LLM
query.rubric         # one-to-two sentence relevance anchor
```

The returned `Query` gets passed unchanged to every chunk reader in
Stage 3 — that's what keeps relevance judgments consistent across the
document.

## Stage 3 — extract

Asks a [pydantic-ai](https://ai.pydantic.dev) agent (default:
`anthropic:claude-haiku-4-5-20251001`) for verbatim spans from a single
chunk that address the query, then verifies each returned span actually
appears as a substring before attaching citation metadata. Verified
excerpts are the only spans that reach downstream stages.

```python
from highlighter.extract import build_extractor_agent, extract_excerpts
from highlighter.query import Query

query = Query(
    question="What are the prerequisites for deploying?",
    sub_questions=["Which AWS services are required?", "What IAM permissions?"],
    rubric="Useful excerpts name specific software, permissions, or accounts.",
)

for chunk in chunks:
    for excerpt in extract_excerpts(chunk, query):
        excerpt.text               # verbatim span from the chunk
        excerpt.which_subquestion  # which sub-question it addresses (or None)
        excerpt.confidence         # extractor-reported confidence
        excerpt.line_start         # citation: inherited from chunk
        excerpt.section_path
```

Swap providers via the `model` argument: `build_extractor_agent(model="openai:gpt-4o-mini")`,
`"ollama:llama3"`, etc. Requires the corresponding API key env var for
hosted providers (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## Development

```
uv sync
uv run pytest
uv run ruff check .
```

Markdown has no native page numbers, so the citation surrogate is the
heading path (e.g. `Step 1: Install the AgentCore CLI > Opt into the
preview channel`) plus a line number into the normalized text.
