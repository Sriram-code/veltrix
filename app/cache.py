"""Answer cache for document-only (rag) answers, tolerant of typos
and rewording.

How a lookup works:
1. Partition: role, region, today, reference date, holiday flag,
   corpus fingerprint and prompt/model version must match EXACTLY.
   So an HR or manager answer can never reach an employee, and a
   different region, date or corpus never shares answers.
2. Inside the partition, the question is compared with stored ones:
   - exact match after normalising (case, punctuation, spaces), or
   - spelling similarity (difflib) >= CACHE_FUZZY_MIN, or
   - meaning similarity (local embeddings) >= CACHE_SEMANTIC_MIN.
3. Safety guards: numbers and negation words must be identical, and
   key words (months, weekdays, leave types, tense, places, units)
   must match, allowing clear typos. "June" never matches "July",
   "maternity" never matches "paternity", "5 days" never "20 days".

Never cached: tool / hybrid answers (personal leave data), clarify,
confirmations, no_answer and LLM failures. In memory only.
"""
from __future__ import annotations

import difflib
import hashlib
import os
import re
import threading
import time
from collections import OrderedDict

ENABLED = os.getenv("CACHE_ENABLED", "1") != "0"
TTL_S = int(os.getenv("CACHE_TTL_S", "3600"))
MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "500"))
FUZZY_MIN = float(os.getenv("CACHE_FUZZY_MIN", "0.88"))
SEMANTIC_MIN = float(os.getenv("CACHE_SEMANTIC_MIN", "0.92"))
PREPLAN_SEMANTIC_MIN = float(os.getenv("CACHE_PREPLAN_SEMANTIC_MIN", "0.95"))

_NEG = {"not", "no", "never", "cannot", "cant", "dont", "doesnt",
        "didnt", "wont", "isnt", "arent", "without"}

# Words that change the meaning even when the rest of the question is
# the same. Both questions must contain the same ones (typos allowed).
_KEY_WORDS = {
    # months and weekdays
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "today", "tomorrow", "yesterday",
    "week", "month", "year",
    # leave types
    "annual", "sick", "maternity", "paternity", "parental", "unpaid",
    "bereavement", "wellbeing", "learning", "jury", "restricted",
    # time direction / tense
    "did", "was", "were", "last", "previous", "ago", "next",
    "before", "after", "until", "since", "upcoming", "future",
    # places
    "india", "us", "usa", "germany", "bengaluru", "austin", "berlin",
    # amounts and units
    "day", "night", "hour", "hours", "half", "minimum",
    "maximum", "max", "min", "per",
}


def norm(text) -> str:
    text = str(text or "").lower().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _guards(text: str):
    """Numbers and negation words: must match exactly."""
    words = norm(text).split()
    numbers = tuple(sorted(w for w in words if w.isdigit()))
    negations = tuple(sorted(w for w in words if w in _NEG))
    return numbers, negations


def _closest_key(tok):
    """The key word this token is clearly a typo of, or None."""
    if len(tok) < 3 or tok.isdigit():
        return None
    scored = sorted(((difflib.SequenceMatcher(None, tok, k).ratio(), k)
                     for k in _KEY_WORDS), reverse=True)
    best, second = scored[0], scored[1]
    if (best[0] >= 0.75 and best[0] > second[0]
            and abs(len(tok) - len(best[1])) <= 1
            and tok[0] == best[1][0]):
        return best[1]
    return None


def keys_compatible(a: str, b: str) -> bool:
    """Same key words in both questions, allowing clear typos."""
    wa, wb = norm(a).split(), norm(b).split()
    ka = {w for w in wa if w in _KEY_WORDS}
    kb = {w for w in wb if w in _KEY_WORDS}
    oa = [w for w in wa if w not in _KEY_WORDS]
    ob = [w for w in wb if w not in _KEY_WORDS]
    for missing, others in ((ka - kb, ob), (kb - ka, oa)):
        for k in missing:
            if not any(_closest_key(t) == k for t in others):
                return False
    return True


def make_partition(*, role, region, today, ref_date, holiday,
                   corpus_hash, version) -> str:
    parts = [str(role), str(region), str(today), str(ref_date),
             "H" if holiday else "-", str(corpus_hash), str(version)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


class AnswerCache:
    def __init__(self, ttl_s=None, max_items=None, enabled=None,
                 embed_fn=None, fuzzy_min=None, semantic_min=None):
        self.ttl_s = TTL_S if ttl_s is None else ttl_s
        self.max_items = MAX_ITEMS if max_items is None else max_items
        self.enabled = ENABLED if enabled is None else enabled
        self.fuzzy_min = FUZZY_MIN if fuzzy_min is None else fuzzy_min
        self.semantic_min = (SEMANTIC_MIN if semantic_min is None
                             else semantic_min)
        self.embed_fn = embed_fn  # text -> list[float], optional
        self._data: OrderedDict = OrderedDict()  # (part, norm) -> entry
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _embed(self, text):
        if self.embed_fn is None:
            return None
        try:
            return list(self.embed_fn(text))
        except Exception:
            return None

    def lookup(self, partition: str, message: str, exact_only: bool = False,
               min_meaning=None, count_miss: bool = True):
        """Returns (value, info) or (None, info).

        exact_only  : only an exact (normalised) match counts.
        min_meaning : stricter meaning-match threshold (used before the planner).
        count_miss  : False for the pre-planner check so a miss is not counted twice.
        """
        if not self.enabled:
            return None, {"match": "disabled"}
        q = norm(message)
        guards = _guards(message)
        now = time.time()
        sem_min = (self.semantic_min if min_meaning is None
                   else max(min_meaning, self.semantic_min))
        with self._lock:
            for k in [k for k, e in self._data.items() if e["exp"] <= now]:
                del self._data[k]
            exact = self._data.get((partition, q))
            if exact is not None:
                self._data.move_to_end((partition, q))
                self.hits += 1
                return exact["value"], {"match": "exact", "score": 1.0}
            if exact_only:
                return None, {"match": "none"}
            cands = [(k, e) for k, e in self._data.items()
                     if k[0] == partition and e["guards"] == guards
                     and keys_compatible(message, e["question"])]
        if not cands:
            if count_miss:
                with self._lock:
                    self.misses += 1
            return None, {"match": "none"}

        best = None  # (score, kind, key, entry)
        for k, e in cands:
            r = difflib.SequenceMatcher(None, q, k[1]).ratio()
            if r >= self.fuzzy_min and (best is None or r > best[0]):
                best = (r, "spelling", k, e)
        if best is None and self.embed_fn is not None:
            qv = self._embed(q)
            if qv is not None:
                for k, e in cands:
                    if e["vec"] is None:
                        continue
                    s = _cosine(qv, e["vec"])
                    if s >= sem_min and (best is None or s > best[0]):
                        best = (s, "meaning", k, e)
        with self._lock:
            if best is None:
                if count_miss:
                    self.misses += 1
                return None, {"match": "none"}
            self.hits += 1
            if best[2] in self._data:
                self._data.move_to_end(best[2])
            return best[3]["value"], {
                "match": best[1], "score": round(best[0], 3),
                "cached_question": best[3]["question"]}

    def store(self, partition: str, message: str, value) -> None:
        if not self.enabled:
            return
        q = norm(message)
        entry = {"exp": time.time() + self.ttl_s, "value": value,
                 "guards": _guards(message), "question": message[:200],
                 "vec": self._embed(q)}
        with self._lock:
            self._data[(partition, q)] = entry
            self._data.move_to_end((partition, q))
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "items": len(self._data),
                    "hits": self.hits, "misses": self.misses}