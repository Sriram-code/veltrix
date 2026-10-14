"""Builds the corpus context for one LLM call.
Access control happens HERE, before scoring: chunks from documents the requester may not read
(role/access_level) or that are not for their region are never candidates, so they can never reach
the LLM. The 3,000-token corpus budget is also enforced here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.config import HARD_CORPUS_TOKEN_LIMIT, settings
from app.ingestion.pipeline import KnowledgeBase
from app.retrieval import precedence as P
from app import trace
from app.retrieval.tokens import count_tokens


@dataclass
class Source:
    sid: str
    doc_id: str
    locator: str
    text: str
    upcoming: bool


@dataclass
class Context:
    sources: list[Source]
    rendered: str
    corpus_tokens: int
    precedence_notes: str


def allowed_chunk_indices(kb: KnowledgeBase, role: str, region: str) -> set[int]:
    ok = {d for d, m in kb.docs.items() if P.readable_for(m, role, region)}
    return {i for i, c in enumerate(kb.chunks) if c.doc_id in ok}


def _render_one(s: Source, meta: P.DocMeta) -> str:
    status = f"UPCOMING (not yet in force; effective {meta.effective_date})" if s.upcoming else "IN FORCE"
    return (f'<source id="{s.sid}" doc_id="{s.doc_id}" type="{meta.doc_type}" regions="{",".join(meta.regions)}" '
            f'effective="{meta.effective_date}" status="{status}" section="{s.locator}">\n{s.text}\n</source>')


def _render(kb, chosen) -> str:
    return "\n".join(_render_one(s, kb.docs[s.doc_id]) for s in chosen)


def _ordered_hits(kb, hits, top_n=20):
    """Best chunk of each regional doc (from the top hits) first,
    then everything else in relevance order."""
    first = []
    seen = set()
    for i, _ in hits[:top_n]:
        meta = kb.docs[kb.chunks[i].doc_id]
        if P.scope(meta) and meta.doc_id not in seen:
            seen.add(meta.doc_id)
            first.append(i)
    rest = [i for i, _ in hits if i not in first]
    return first + rest


def build_context(kb, role, region, ref, query, budget=None,
                  per_doc_cap=3, max_upcoming=2, max_sources=12):
    budget = min(budget or settings.corpus_token_budget,
                 HARD_CORPUS_TOKEN_LIMIT)
    allowed = allowed_chunk_indices(kb, role, region)
    hits = kb.index.search(query, allowed, k=40) if kb.index else []
    chosen = []
    per_doc = {}
    n_up = 0
    used = 0
    for i in _ordered_hits(kb, hits):
        if len(chosen) >= max_sources:
            break
        c = kb.chunks[i]
        meta = kb.docs[c.doc_id]
        up = meta.effective_date > ref
        if per_doc.get(c.doc_id, 0) >= per_doc_cap:
            continue
        if up and n_up >= max_upcoming:
            continue
        s = Source(f"S{len(chosen) + 1}", c.doc_id, c.locator,
                   c.text, up)
        t = count_tokens(_render_one(s, meta))
        if used + t > budget:
            continue
        chosen.append(s)
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        n_up += up
        used += t

    rendered = _render(kb, chosen)
    total = count_tokens(rendered)
    while total > HARD_CORPUS_TOKEN_LIMIT and chosen:
        chosen.pop()
        rendered = _render(kb, chosen)
        total = count_tokens(rendered)

    notes = precedence_notes(kb, chosen, ref)
    trace.step(
        "retrieval",
        query=query[:200],
        reference_date=str(ref),
        search=kb.index.describe() if kb.index else "none",
        docs_allowed=len({kb.chunks[i].doc_id for i in allowed}),
        chunks_allowed=len(allowed),
        top_hits=[{"doc_id": kb.chunks[i].doc_id,
                   "locator": kb.chunks[i].locator,
                   "score": round(sc, 4)} for i, sc in hits[:10]],
        selected=[{"sid": s.sid, "doc_id": s.doc_id,
                   "locator": s.locator, "upcoming": s.upcoming,
                   "tokens": count_tokens(s.text)} for s in chosen],
        corpus_tokens=total,
        budget=budget,
        hard_limit=HARD_CORPUS_TOKEN_LIMIT,
        precedence=notes,
    )
    return Context(chosen, rendered, total, notes)


def precedence_notes(kb: KnowledgeBase, chosen: list[Source], ref: date) -> str:
    """Precedence ordering computed in code (metadata only, no corpus text)."""
    ids = list(dict.fromkeys(s.doc_id for s in chosen))
    cur = [kb.docs[d] for d in ids if kb.docs[d].effective_date <= ref]
    up = [kb.docs[d] for d in ids if kb.docs[d].effective_date > ref]
    lines = ["Documents in force, strongest first (same rank number = tie):"]
    rank, prev = 0, None
    for m in sorted(cur, key=P.precedence_key, reverse=True):
        k = P.precedence_key(m)
        if k != prev:
            rank, prev = rank + 1, k
        lines.append(f"  rank {rank}: {m.doc_id} (authority={'high' if P.authority(m) else 'low'}, "
                     f"scope={'regional' if P.scope(m) else 'global'}, effective={m.effective_date})")
    for u in up:
        beats = [c.doc_id for c in cur if P.compare(u, c) > 0]
        lines.append(f"  UPCOMING {u.doc_id} effective {u.effective_date}: would outrank {beats or 'none'} "
                     f"on its effective date")
    return "\n".join(lines)
