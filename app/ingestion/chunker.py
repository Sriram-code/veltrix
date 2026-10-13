"""Section-aware chunking. A chunk never crosses a heading, so each chunk has exactly
one locator (the heading, or 'p. N' when there is none)."""
from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers import Block
from app.retrieval.tokens import count_tokens

TARGET_TOKENS = 220
MAX_TOKENS = 300


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str
    text: str
    tokens: int


def _split_long(text: str, max_tokens: int) -> list[str]:
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if count_tokens(" ".join(cur)) >= max_tokens:
            out.append(" ".join(cur))
            cur = cur[-20:]  # small overlap
    if cur and (not out or len(cur) > 20):
        out.append(" ".join(cur))
    return out


def chunk_blocks(doc_id: str, blocks: list[Block], is_pdf: bool) -> list[Chunk]:
    groups: list[tuple[str, list[str]]] = []
    for b in blocks:
        if b.kind == "heading":
            continue
        loc = b.heading or (f"p. {b.page}" if is_pdf else "p. 1")
        if not groups or groups[-1][0] != loc:
            groups.append((loc, []))
        groups[-1][1].append(b.text)
    chunks: list[Chunk] = []
    for loc, texts in groups:
        cur: list[str] = []
        pieces: list[str] = []
        for t in texts:
            if count_tokens(t) > MAX_TOKENS:
                if cur:
                    pieces.append("\n".join(cur))
                    cur = []
                pieces.extend(_split_long(t, TARGET_TOKENS))
                continue
            if cur and count_tokens("\n".join(cur + [t])) > TARGET_TOKENS:
                pieces.append("\n".join(cur))
                cur = []
            cur.append(t)
        if cur:
            pieces.append("\n".join(cur))
        for p in pieces:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, loc, p, count_tokens(p)))
    return chunks
