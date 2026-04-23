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

## Stage 0 — normalize

```
python -m highlighter.normalize path/to/doc.md
```

Prints the content hash and the parsed heading tree with line ranges.

Programmatic:

```python
from highlighter.normalize import normalize

doc = normalize("path/to/doc.md")
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
