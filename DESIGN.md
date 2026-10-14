# DESIGN

## Ingestion: tables and the scanned document
`app/ingestion/pipeline.py` reads `MANIFEST_PATH` and parses every file in `CORPUS_DIR` at startup, so a replaced
`./data` folder is picked up automatically. `parsers.py` handles each format: **PDF** (PyMuPDF; larger or bold short
lines become headings), **DOCX** (python-docx heading styles and tables in document order), **HTML** (BeautifulSoup,
h1–h6 and `<table>`) and **Markdown** (`#` headings, pipe tables). **Scanned pages** (under 25 characters of text)
are rendered at 300 dpi and read with **Tesseract OCR**; detection is automatic, so no file is special-cased, and a
character-quality filter drops OCR noise.

**Tables** are stored one row per line as `Header: value; Header: value`, so each row stays meaningful on its own.
For calendars, `holidays.py` extracts dates into the new table `ingested_holidays` (ISO, D Month, Month D, D.M.Y,
D/M/Y, M/D for US, German month names), skips "effective/updated" lines and flags optional holidays. No corpus fact
is written into code.

## Chunking and retrieval
- **Section-aware chunks.** A chunk never crosses a heading, so each has one locator. About 220 tokens (max 300),
  so 10–12 passages from different documents fit the budget and precedence can compare v1, v2, the FAQ and the
  addendum side by side.
- **Hybrid search.** BM25 plus dense embeddings (`bge-small-en-v1.5` via fastembed, ONNX on CPU, no API key),
  merged with Reciprocal Rank Fusion. The dense vectors live in **Qdrant, embedded in-process** (no extra
  container; `QDRANT_URL` switches to a Qdrant server). Search is restricted to the requester's allowed chunk ids
  with a Qdrant filter. If Qdrant is unavailable, exact NumPy search is used instead.
- **Selection.** The best chunk of each regional document in the top 20 goes first (added after regional addenda
  ranked behind FAQ and wiki chunks). Then at most 3 chunks per document, 2 upcoming documents and 12 passages.
- **Budget enforced in code.** `retriever.build_context` counts the rendered context with tiktoken `cl100k_base`
  against 2,800 and re-checks the hard limit of 3,000. `LLMClient.complete_json` refuses any call over 3,000 before
  contacting any provider, and logs `corpus_tokens` for every call.
- **Answer cache.** Document-only answers are cached in memory (1 hour, 500 items), partitioned by role, region,
  today, reference date, holiday flag, corpus fingerprint and prompt/model version, never by employee ID. Inside a
  partition, a question matches exactly, by spelling (difflib ≥ 0.88) or by meaning (embeddings ≥ 0.92), but only
  if numbers, negations and key words (months, weekdays, leave types, tense, places, units) agree. A hit skips
  retrieval and the answer call. Tool and hybrid answers are never cached (personal leave data).

### Retrieval evaluation (reranker tested and turned off)
`eval/eval_set.json` has 35 labelled cases, 26 of which test retrieval: precedence by region, past dates, the memo,
holidays, and restricted documents for each role. `scripts/run_eval.py --retrieval` measures this without the LLM;
`--e2e` measures route, citations, facts, leaks, latency and tokens against the running service.

| Setup (Qdrant + BM25 + RRF) | Recall | MRR | Leaks | Avg corpus tokens | Avg retrieval time |
|---|---|---|---|---|---|
| With cross-encoder reranker (ms-marco-MiniLM-L-6-v2, top 30) | 1.0 | 0.713 | 0 | 1,565 | 1,579 ms |
| **Without reranker (default)** | **1.0** | **0.733** | **0** | **1,473** | **23 ms** |

The reranker improved one case (manager handbook, rank 2 → 1) and lowered five (for example sick leave in 2025,
5 → 7). It is trained on web search queries, not HR policies, and scores passages one at a time, so it cannot know
that an addendum outranks a newer FAQ. Since recall is already 1.0, all 12 passages reach the LLM and precedence is
computed in code, so it adds 1.5 s per question for no gain. It is **off by default** (`RERANKER=none`) and kept as
an option. With 26 cases the MRR difference is small; the latency cost is clear.

## LLM providers
`app/llm/client.py` builds a provider chain: the configured paid provider, then other paid keys, then Groq (free
tier, open-weight models), then local Ollama. Failures fall through to the next provider. All calls use JSON mode,
temperature 0 and a fixed seed where supported; the same budget guard and logs apply to every provider.

## Section 5 rules
`app/retrieval/precedence.py` has pure, unit-tested functions: `is_eligible` (access, region, effective date),
`precedence_key` (authority, scope, effective date), `winners` (all tied documents, i.e. the conflict case) and
`upcoming_that_would_win`. The planner extracts the reference date; documents not yet in force are labelled
UPCOMING. The answer prompt gets the ranking computed in code, so the LLM only has to recognise which passages
describe the same rule. Holiday answers also receive the extracted holiday records as structured data.

## Where access control is enforced
- **Identity**: `main.chat` (401) and `SessionStore.get_or_create` (403). Role and region always come from the DB.
- **Documents**: `retriever.allowed_chunk_indices` uses `precedence.readable_for`; only those chunk ids are searched
  (Qdrant filter, or NumPy rows), so restricted text never reaches a prompt or a citation.
  `orchestrator._answer_from_docs` cites only source ids that were actually sent.
- **Leave data**: `leave_tools.can_view_employee`, `get_team_leave` (managers only), `validate_apply`,
  `apply_leave`, `validate_cancel`, `cancel_leave` (own requests only), all using `ToolContext.requester`.
- **Name lookup**: `orchestrator._resolve_employee` searches only employees the requester may see; otherwise the
  identical refusal.
- **Cache**: the role and region partition means an HR or manager answer can never be served to an employee.
- **Manipulation**: a regex check before any LLM call, plus the planner category. Writes always need a
  server-stored proposal plus a confirmation.

## Keeping personal data out of logs and prompts
`db.Employee` never loads `phone`. Logs hold only IDs, routes, tool statuses and token counts, never message text,
and `JsonFormatter` also redacts emails and phone numbers. The planner prompt carries the requester's ID, role and
region, not their name. Tool data is scrubbed by `privacy.scrub`; names appear only in team-leave results. Answers
and tool arguments have emails masked and phone numbers removed.

## Failure modes found
- **Fixed:** injected text in documents (passages wrapped in `<source>` tags and treated as data; tools run only
  from planner output and server-side proposals); table rows losing headers; "Effective 1 January" parsed as a
  holiday; optional holidays reducing working days; relative dates resolved wrongly (date cheat sheet, "next week"
  fixed in code); ISO dates masked as phone numbers; holiday names losing weekday words ("Good Friday" → "Good");
  OCR noise; source ids leaking into answers ("(S3)", now removed in code).
- **Fixed: the LLM merged separate sentences into one rule** ("jury service follows local law" + "up to 20 unpaid
  days"). Prompt rules 9–12: each sentence is its own rule, "follows local law" means a partial answer plus HR.
- **Fixed: the retired default Groq model** returned 404; default now `openai/gpt-oss-120b`, errors logged.
- **Fixed: the typo-tolerant cache matched different questions** ("holidays in June" vs "July", "maternity" vs
  "paternity", similarity 0.94–0.96). Key words must now agree.
- **Tested and dropped:** a cross-encoder reranker (see the evaluation above).
- **Open:** "same rule" detection is done by the LLM; OCR quality depends on the scan; answer wording varies a
  little between runs on the free model.

## What breaks first at 10,000 documents, and fixes
1. **Startup ingestion and OCR** would exceed 10 minutes. Fix: incremental ingestion keyed on file hash, embeddings
   persisted in Qdrant server mode, and a worker pool for OCR.
2. **In-memory BM25 and embedded Qdrant** grow with the corpus. Fix: Qdrant server with payload filters
   (access_level, regions, effective_date) and its sparse vectors or full-text index in place of in-memory BM25.
3. **Many versions per topic crowd the budget.** Fix: group policy families at ingestion and send only the winner,
   tied rivals and one upcoming version, or extract structured facts so code resolves precedence.
4. **In-memory sessions and cache.** Fix: Valkey (open-source Redis fork) with the same keys, and several workers.