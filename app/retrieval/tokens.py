"""Token counting with tiktoken cl100k_base (as required). The Docker image pre-downloads
the encoding. If unavailable (offline dev box), a pessimistic estimate keeps us under budget."""
from __future__ import annotations

import logging
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # pragma: no cover
        logging.getLogger("veltrix").warning("tiktoken_unavailable_using_fallback",
                                             extra={"fields": {"err": str(e)[:80]}})
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(len(text) // 2, int(len(text.split()) * 2)) + 1
