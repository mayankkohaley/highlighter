"""Stage 0: load and normalize a markdown document."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel

_ATX_HEADING_RE = re.compile(r"^(#{1,6})(?!#)\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


class Section(BaseModel):
    level: int
    title: str
    line_start: int
    line_end: int


class NormalizedDoc(BaseModel):
    source_path: str
    content_hash: str
    text: str
    sections: list[Section] = []

    def section_path_for_line(self, line: int) -> list[str]:
        """Return the heading titles (outermost first) containing a 1-indexed line."""
        stack: list[Section] = []
        for s in self.sections:
            if s.line_start > line:
                break
            while stack and stack[-1].level >= s.level:
                stack.pop()
            if s.line_end >= line:
                stack.append(s)
        return [s.title for s in stack]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _parse_sections(text: str) -> list[Section]:
    lines = text.split("\n")
    sections: list[Section] = []
    in_fence = False
    fence_char = ""
    for i, line in enumerate(lines, start=1):
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
                fence_char = ""
            continue
        if in_fence:
            continue
        m = _ATX_HEADING_RE.match(line)
        if m:
            sections.append(Section(
                level=len(m.group(1)),
                title=m.group(2).strip(),
                line_start=i,
                line_end=i,  # filled in below
            ))

    # Each section runs until the next same-or-shallower heading, or EOF.
    # Trailing blank lines at EOF are excluded from the final section's range.
    eof_line = len(lines)
    while eof_line > 0 and lines[eof_line - 1] == "":
        eof_line -= 1
    for idx, sec in enumerate(sections):
        end = eof_line
        for j in range(idx + 1, len(sections)):
            if sections[j].level <= sec.level:
                end = sections[j].line_start - 1
                break
        sec.line_end = end
    return sections


def normalize(path: Path | str) -> NormalizedDoc:
    path = Path(path)
    raw = path.read_bytes()
    text = _normalize_text(raw.decode("utf-8"))
    return NormalizedDoc(
        source_path=str(path),
        content_hash=hashlib.sha256(raw).hexdigest(),
        text=text,
        sections=_parse_sections(text),
    )


def _main(argv: list[str]) -> int:
    import sys
    if len(argv) != 2:
        print("usage: python -m highlighter.normalize <markdown-file>", file=sys.stderr)
        return 1
    doc = normalize(argv[1])
    print(f"source:  {doc.source_path}")
    print(f"sha256:  {doc.content_hash}")
    print(f"sections: {len(doc.sections)}")
    print()
    for s in doc.sections:
        indent = "  " * (s.level - 1)
        print(f"{indent}{'#' * s.level} {s.title}  (L{s.line_start}-{s.line_end})")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
