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
1. **Query expansion** — turn the user's question into sub-questions,
   entities to watch for, and a relevance rubric.
2. **Chunk** — split the normalized text into overlapping chunks that
   prefer paragraph/heading boundaries.
3. **Per-chunk extraction** — for each chunk, ask an LLM to return verbatim
   spans relevant to the query. Verify each returned span actually appears
   in the chunk.
4. **Consolidate** — dedupe overlapping excerpts, merge adjacent spans.
5. **Synthesize** — final LLM call answers the question using only the
   consolidated excerpts, with citations.

Only Stage 0 is implemented right now. Markdown input only — PDF support
will slot in behind the same interface later.

## Stage 0 usage

```
python -m highlighter.normalize path/to/doc.md
```

Prints the content hash and the parsed heading tree with line ranges.

## Development

```
uv sync
uv run pytest
uv run ruff check .
```

Markdown has no native page numbers, so the citation surrogate is the
heading path (e.g. `Step 1: Install the AgentCore CLI > Opt into the
preview channel`) plus a line number into the normalized text.
