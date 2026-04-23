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


class NormalizedDoc(BaseModel):
    source_path: str
    content_hash: str
    text: str
    sections: list[Section] = []


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _parse_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    in_fence = False
    fence_char = ""
    for i, line in enumerate(text.split("\n"), start=1):
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
            ))
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
