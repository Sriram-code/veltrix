# DESIGN

## Ingestion: tables and the scanned document
`app/ingestion/pipeline.py` reads `MANIFEST_PATH` and parses every file in `CORPUS_DIR` on each startup, so a
replaced `./data` folder is picked up automatically. `parsers.py` handles each format:
- **PDF**: PyMuPDF text with font sizes. Larger or bold short lines are treated as headings.
- **Scanned pages**: any page with fewer than 25 characters of text is rendered at 300 dpi and read with
  **Tesseract OCR**. Scans are detected automatically, so no file is special-cased.
- **DOCX**: python-docx, with heading styles and tables read in document order.
- **HTML**: BeautifulSoup, using h1–h6 headings and `<table>` elements.
- **Markdown**: `#` headings and pipe tables.

**Tables** are stored one row per line as `Header: value; Header: value`, so each row stays meaningful on its own.
For calendar documents, `holidays.py` extracts dates during ingestion into the new table `ingested_holidays`. It
handles ISO, D Month, Month D, D.M.Y and D/M/Y formats (M/D for US), plus German month names. It skips
"effective/updated" lines and flags optional holidays. No corpus fact is written into code.

## Chunking and retrieval
- **Chunks are section-aware.** A chunk never crosses a heading, so each chunk has one locator. Chunks are about
  220 tokens (300 at most).
- **Small chunks fit the budget.** About 10 passages from different documents fit within 3,000 tokens, which
  precedence needs: v1, v2, the FAQ and the addendum must be visible together.
- **Search is hybrid.** It combines BM25 with local open-source dense embeddings (`bge-small-en-v1.5` via
  fastembed, ONNX on CPU, no API key), merged with Reciprocal Rank Fusion. Results are capped at 3 chunks per
  document and 2 upcoming documents.
- **The budget is enforced in code.** `retriever.build_context` counts the rendered context with tiktoken
  `cl100k_base` against 2,800 and re-checks the hard limit of 3,000. `LLMClient.complete_json` refuses any call over
  3,000 before contacting *any* provider, and logs `corpus_tokens` for every call.
- **Regional documents go first.** The best chunk from each
  regional document in the top 20 hits is placed first, and at
  most 12 passages are sent. This was added after testing showed
  regional addenda ranked low behind FAQ and wiki chunks.

## LLM providers
`app/llm/client.py` resolves a provider chain: the configured paid provider, then other paid keys, then Groq
(free tier, open-weight models), then a local Ollama model. Failures fall through to the next provider. All calls
use JSON mode and temperature 0, plus a fixed seed where the provider supports one. The same budget guard and logs
apply to every provider.

## Section 5 rules
`app/retrieval/precedence.py` contains pure, unit-tested functions:
- `is_eligible`: access, region and effective date.
- `precedence_key`: (authority, scope, effective date).
- `winners`: returns all tied documents, which is the conflict case.
- `upcoming_that_would_win`: checks whether an upcoming document would outrank the current ones on its effective
  date.

The planner extracts the reference date. Documents that are not yet in force are labelled UPCOMING. The answer
prompt gets a ranking computed in code, so the LLM only has to recognise which passages describe the same rule.
Holiday answers also receive the extracted holiday records as structured data.

## Where access control is enforced
- **Identity**: `main.chat` (401) and `SessionStore.get_or_create` (403). Role and region always come from the
  database.
- **Documents**: `retriever.allowed_chunk_indices` uses `precedence.readable_for`, and `HybridIndex.search` scores
  only those chunks, so restricted text never reaches a prompt. `orchestrator._answer_from_docs` cites only source
  IDs that were actually sent.
- **Leave data**: `leave_tools.can_view_employee`, `get_team_leave` (managers only), and `validate_apply`,
  `apply_leave`, `validate_cancel`, `cancel_leave` (own requests only). All use `ToolContext.requester`.
- **Name lookup**: `orchestrator._resolve_employee` searches only employees the requester may see, and gives the
  identical refusal otherwise.
- **Manipulation**: caught by a regex check before any LLM call, and by the planner category. Writes always need a
  server-stored proposal plus a confirmation.

## Keeping personal data out of logs and prompts
- **Data loading**: `db.Employee` never loads `phone`.
- **Logs**: only IDs, routes, tool statuses and token counts, never message text. `JsonFormatter` also redacts
  emails and phone numbers.
- **Planner prompt**: carries the requester's ID, role and region, not their name.
- **Tool data in prompts**: scrubbed by `privacy.scrub`. Names appear only in team-leave results.
- **Answers and tool arguments**: emails are masked and phone numbers removed.

## Failure modes found
- **Fixed: injected text in documents.** Passages are wrapped in `<source>` tags and treated as data, and tools run
  only from planner output and server-side proposals.
- **Fixed: table rows lost their headers** when split into chunks.
- **Fixed: "Effective 1 January" was parsed as a holiday.**
- **Fixed: optional holidays reduced working days.**
- **Fixed: relative dates were resolved wrongly.** The planner now gets a computed date cheat sheet, and "next week"
  is overridden in code.
- **Fixed: ISO dates were masked as phone numbers.**
- **Fixed: the LLM merged separate sentences into one rule.**
  It combined "jury service follows local law" with the
  unrelated "up to 20 unpaid days" rule. Fixed with answer
  prompt rules 9–11 (each sentence is its own rule; "follows
  local law" means a partial answer plus a pointer to HR).
- **Fixed: holiday names lost weekday words** ("Good Friday"
  became "Good"). The name cleaner now removes only cells that
  contain nothing but a weekday.
- **Fixed: OCR noise** (signature and logo marks) is dropped by
  a character-quality filter.
- **Fixed: the default Groq model was retired**, which returned a
  404. The default is now `openai/gpt-oss-120b`, and provider
  errors are logged with their reason.
- **Open: "same rule" detection is done by the LLM.** OCR quality depends on the scan. Small local models are less
  reliable.
- **Fixed: source IDs leaked into answers** ("(S3)"). They are
  now removed in code, not left to the prompt.

## What breaks first at 10,000 documents, and fixes
1. **Startup ingestion and OCR** would exceed 10 minutes. Fix: incremental ingestion keyed on file hash, with a
   persistent index and worker pool.
2. **Brute-force vectors and in-memory BM25.** Fix: Qdrant or pgvector with payload filters (access_level, regions,
   effective_date), so filtering happens inside the vector store.
3. **Many versions per topic crowd the budget.** Fix: group policy families at ingestion and send only the winner,
   tied rivals and one upcoming version, or extract structured facts so code resolves precedence.
4. **In-memory sessions.** Fix: Redis and multiple workers.
