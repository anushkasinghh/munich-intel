# Architectural Decisions

Tradeoffs made during development. Revisit these when upgrading past the MVP.

---

## Embeddings: sentence-transformers over FlagEmbedding

**Chosen:** `sentence-transformers` (wraps BGE-M3)
**Rejected:** `FlagEmbedding` (official BGE-M3 package)

**Why:** FlagEmbedding handles sparse vectors needed for hybrid search (Phase 2), but has heavier dependencies and a more complex API. sentence-transformers wraps BGE-M3 fine for dense-only Phase 1.

**When to revisit:** Phase 2 hybrid search upgrade. Swap happens in one file (the embeddings module).

---

## Data sources: config file (YAML) over hardcoded URLs

**Chosen:** `companies.yaml` — all scrape targets live in config, not code
**Rejected:** hardcoding URLs in Python

**Why:** Adding a company means editing YAML, not touching Python. 8 companies × 2 URLs = 16 pages for Phase 1 — fast enough to ingest in one sitting, varied enough to surface real parsing edge cases.

**When to revisit:** If sources grow beyond ~50 companies, consider a database or CMS instead of a flat file.

---

## Scraper: save raw pages to disk before chunking

**Chosen:** scrape once → save JSON to `data/raw/` → chunk/embed from disk
**Rejected:** scrape → chunk → embed in one pass without saving

**Why:** Scraping is slow and rate-limited. Chunking and embedding are fast and free to re-run. Saving raw pages means you can tune your chunker or embedder without re-hitting websites. The separation also makes debugging easier — you can inspect what was actually scraped.

**When to revisit:** If the data/raw directory grows too large (>10GB), consider streaming directly into the pipeline instead.

---

## Scraper: tenacity retries on HTTP fetches

**Chosen:** 3 retries with exponential backoff (2s → 10s)
**Rejected:** single attempt, fail immediately

**Why:** Startup websites go down constantly. Without retries, a transient 503 kills the entire ingest run. Exponential backoff avoids hammering a struggling server.

**When to revisit:** Never — this is always the right call for external HTTP calls.

---

## Chunker: hand-rolled over LangChain's RecursiveCharacterTextSplitter

**Chosen:** custom `chunk_text()` — ~20 lines, paragraph-first splitting
**Rejected:** `langchain_text_splitters.RecursiveCharacterTextSplitter`

**Why:** LangChain's splitter is battle-tested but adds a heavy dependency for something you can write in 20 lines. Phase 1 priority is understanding what chunking actually does. If edge cases appear in Phase 2, swap in LangChain's version — the interface is almost identical.

**When to revisit:** Phase 2, if you see chunks that split mid-sentence badly or lose context across boundaries.

---

## Chunker: word count as proxy for token count

**Chosen:** count words (`.split()`) to approximate 512-token chunks
**Rejected:** use a real tokenizer (e.g. tiktoken or the model's own tokenizer)

**Why:** BGE-M3 uses subword tokenization — "Batterieanalyse" is one word but several tokens. Word count is an approximation (~1 token ≈ 0.75–1.3 words). Good enough for Phase 1. For Phase 3 eval, replace with the model's actual tokenizer.

**When to revisit:** Phase 3, when tuning chunk size against retrieval quality metrics.

---

## Embedder: wrapped in its own file, not called directly

**Chosen:** `embedder.py` with `load_model()` + `embed()` as the only public API
**Rejected:** calling `SentenceTransformer` directly in indexer/retriever

**Why:** When Phase 2 swaps sentence-transformers for FlagEmbedding (sparse vectors), only this file changes. The rest of the codebase is insulated.

**When to revisit:** Phase 2 swap — should be a clean drop-in.

---

## Embedder: lru_cache for model loading

**Chosen:** `@functools.lru_cache(maxsize=1)` on `load_model()`
**Rejected:** reloading the model on every request, or a global mutable `_model = None`

**Why:** BGE-M3 is ~570MB. Reloading per request would be catastrophic. `lru_cache` ensures one load, cached for the process lifetime. Cleaner than a mutable module-level variable.

**When to revisit:** Never for single-process. If you move to multi-worker uvicorn, each worker loads its own copy — that's expected behavior.

---

## Embedder: normalize_embeddings=True

**Chosen:** unit-length vectors via `normalize_embeddings=True`
**Rejected:** raw vectors

**Why:** Normalizing to unit length makes cosine similarity equal to dot product. Qdrant can use dot product, which is faster. Without normalization, cosine similarity requires a division per comparison — minor at small scale, meaningful at tens of thousands of vectors.

**When to revisit:** Never — always normalize unless you have a specific reason not to.

---

## Indexer: deterministic UUIDs for point IDs

**Chosen:** `uuid5(NAMESPACE_URL, f"{slug}_{url}_{chunk_index}")` — same input always produces the same ID
**Rejected:** random UUIDs (`uuid4()`)

**Why:** Re-indexing the same page upserts (overwrites) existing vectors instead of creating duplicates. Without this, every ingest run multiplies your data. Phase 1 update strategy is simply: re-run ingest.

**When to revisit:** Phase 2, if you need to track version history of a page rather than overwrite it.

---

## Retriever: top_k=5 default over higher values

**Chosen:** 5 chunks retrieved per query
**Rejected:** 10+ chunks

**Why:** 5 chunks × 512 tokens = ~2560 tokens of context sent to Llama-3.1-8B. The model supports 128k context so headroom exists, but each extra chunk adds LLM latency with diminishing returns. 5 well-ranked chunks covers most factual company questions.

**When to revisit:** Phase 2 — add a `min_score` threshold to filter low-confidence results, and tune `top_k` against eval metrics.

---

## Retriever: returns plain dicts over a typed model

**Chosen:** `list[dict]` with keys chunk_text, company_name, url, score
**Rejected:** a `RetrievedChunk` dataclass or Pydantic model

**Why:** Simplest thing that works for Phase 1. Pipeline only has one consumer.

**When to revisit:** If more than one thing consumes retrieval results, or if missing-key errors become a debugging pain point. A `RetrievedChunk` dataclass is a one-line fix.

---

## Indexer: chunk_text stored in Qdrant payload

**Chosen:** store the raw chunk text alongside the vector in Qdrant's payload
**Rejected:** store only a reference (e.g. a file path or database ID) and do a secondary lookup

**Why:** Qdrant is designed for this — payload retrieval is free. A secondary lookup would add a round-trip to disk or another database on every query. For Phase 1 scale (hundreds of chunks), the storage cost is negligible.

**When to revisit:** If chunk text grows very large or you need to version/update text independently of vectors.

---

## API: lifespan context manager over @app.on_event

**Chosen:** `@asynccontextmanager async def lifespan(app)` — model and Qdrant client loaded on startup, stored in `app.state`
**Rejected:** `@app.on_event("startup")` / `@app.on_event("shutdown")` decorators

**Why:** `on_event` is deprecated in FastAPI 0.93+. `lifespan` is the modern pattern — startup and shutdown live together in one function, making the pairing impossible to miss. `app.state` is FastAPI's intended slot for shared per-process objects like heavy models and connection pools.

**When to revisit:** Never — lifespan is the right pattern going forward.

---

## API: POST /ingest is synchronous (blocking)

**Chosen:** regular `def ingest_endpoint()` — scrape + embed + index happens in the request thread
**Rejected:** async with Celery background task

**Why:** Scraping 7 companies takes 30–60 seconds. That's acceptable in Phase 1 where ingest is an operator-only action run once. Adding Celery means a broker, a worker process, task state management, and a status-polling endpoint. That's 4× the complexity for a dev tool.

**When to revisit:** Phase 2, when ingest needs to run on a schedule or be triggered externally without a blocking HTTP call.

---

## API: company_slug filter via Qdrant payload filter

**Chosen:** `Filter(must=[FieldCondition(key="company_slug", match=MatchValue(value=slug))])` passed down through pipeline → retriever → Qdrant search
**Rejected:** filtering in Python after retrieving all results

**Why:** Qdrant applies the filter at query time — only matching vectors are scored and returned. Python-side filtering would require over-fetching (get 50, filter to 5) and waste embedding computation. The filter field (`company_slug`) is already in every point's payload from the indexer.

**When to revisit:** Phase 2, if you add category-level filtering or multi-company queries.

---

## Pipeline: thin orchestration layer between API and RAG components

**Chosen:** `pipeline.py` with a single `answer()` function — the only entry point the API layer uses
**Rejected:** calling `retrieve()` and `generate()` directly from `api/main.py`

**Why:** When Phase 2 adds re-ranking between retrieval and generation, the change happens here. `api/main.py` stays unchanged. The API layer is insulated from RAG logic. 15 lines is the right size — complexity lives in the components it calls.

**When to revisit:** Phase 2 — add re-ranking (e.g. a cross-encoder pass) between `retrieve()` and `generate()` calls here.

---

## Generator: numbered context sections over a flat dump

**Chosen:** format each retrieved chunk as `[N] CompanyName (url)\n{text}` separated by `---`
**Rejected:** concatenating all chunks as one blob of text

**Why:** Numbered sections let the LLM say "According to [1] NavVis..." — grounding answers in specific sources without requiring any programmatic citation extraction in Phase 1. The pattern is already in place for Phase 2 to parse out citations.

**When to revisit:** Phase 2 — add citation extraction to pull `[N]` references from the response and attach source URLs to the API response.

---

## Generator: stream=False for Phase 1

**Chosen:** `ollama.chat(..., stream=False)` — block until the full response is returned
**Rejected:** streaming response tokens

**Why:** Streaming requires the FastAPI endpoint to return a `StreamingResponse` instead of JSON, which complicates the client contract and error handling. Get the full pipeline working first, then streaming is a 3-line change.

**When to revisit:** Phase 2 — add a `stream: bool` param to the API and switch the endpoint to `StreamingResponse`.

---

## API: retriever uses query_points() not search()

**Chosen:** `client.query_points(query=vector, ...)` with results accessed via `.points`
**Rejected:** `client.search(query_vector=vector, ...)` — removed in qdrant-client 1.13+

**Why:** Breaking API change in qdrant-client. `search()` no longer exists as of 1.13; `query_points()` is the unified replacement that also supports sparse vectors (needed for Phase 2 hybrid search).

**When to revisit:** Never — query_points() is the stable forward path.

---

## Config: Pydantic BaseSettings over module-level constants

**Chosen:** `settings.py` with `BaseSettings`, loaded from `.env`
**Rejected:** constants at the top of each file

**Why:** Environment variables are the standard interface for deployment config. Pydantic adds type validation and IDE autocomplete on top for free.

**When to revisit:** If the app grows to multiple services, split into domain-specific settings classes rather than one god-settings object.

---

## Deployment: HuggingFace Spaces + Qdrant Cloud

**Chosen:** HuggingFace Spaces (Docker, free CPU tier) for the app + Qdrant Cloud (free tier) for the vector DB
**Rejected:** Vercel, Render, single-VPS docker-compose

**Why:** Vercel is serverless — BGE-M3 takes ~15 seconds to load, which would be paid on every cold request. Render's free tier sleeps after inactivity and doesn't support two containers. A single VPS running both services is the simplest setup but costs money. Splitting across HF Spaces (app) and Qdrant Cloud (vectors) gives a permanently-on, zero-cost deployment: HF free tier has 2 vCPU and 16GB RAM (enough for BGE-M3), and Qdrant Cloud's free tier keeps vectors persistent across app restarts.

**When to revisit:** If the Space restarts too frequently or the 1GB Qdrant free tier fills up. Migrate to a Hetzner VPS (~€4/month) running docker-compose with both services.

---

## Docker: CPU-only PyTorch

**Chosen:** `torch` pinned to `https://download.pytorch.org/whl/cpu` via `[tool.uv.sources]` in `pyproject.toml`
**Rejected:** default PyTorch from PyPI (installs full CUDA stack)

**Why:** Default `uv install torch` pulls CUDA, cuDNN, and 15+ nvidia-* packages — 4GB of GPU libraries a CPU server never uses. Switching to the CPU wheel shrinks the image from 5.3GB to 1.3GB and cuts HF Spaces build time by ~6 minutes. BGE-M3 on CPU takes ~100ms per query, which is fine for this use case.

**When to revisit:** If you move to a GPU server. Remove the `[tool.uv.sources]` override and re-lock.

---

## Model versioning: revision pin + collection name as schema version

**Chosen:** `EMBEDDING_MODEL_REVISION` env var (HuggingFace commit hash) + collection name encodes model version (`munich_intel`, `munich_intel_v2`, …)
**Rejected:** no versioning (silently breaks when model weights update)

**Why:** HuggingFace model repos are mutable — weights can change under the same name. If the embedding model changes between indexing and querying, vectors become incompatible and retrieval silently degrades. Pinning to a commit hash freezes the weights. The Qdrant collection name acts as the schema version: when you change the embedding model, create a new collection rather than re-using the old one so old and new vectors coexist during migration.

**When to revisit:** Always pin before production. Get the hash from https://huggingface.co/BAAI/bge-m3/commits/main.

---

## Testing: invariant tests over strategy-specific tests

**Chosen:** tests that verify contracts — no chunk exceeds `chunk_size`, all input words appear in output, vectors are unit-normalized, output count matches input count
**Rejected:** tests tied to specific counts or exact overlap word positions (e.g. `assert len(chunks) == 2`)

**Why:** Strategy-specific tests break every time you change the implementation, even correctly. Invariant tests survive a complete rewrite — they check *what must always be true* regardless of how chunking or embedding is done. A test that asserts `len(chunks) == 2` for 600 words breaks the moment you tune `chunk_size`. A test that asserts no chunk exceeds `chunk_size` never breaks unless the function is actually wrong.

**When to revisit:** Never for the invariants themselves. Add strategy-specific tests only when you need to lock in exact behavior for a known-stable implementation.

---

## Embedder tests: integration test over mock

**Chosen:** real BGE-M3 model loaded in `tests/test_embedder.py`, marked `@pytest.mark.integration`, run manually before model swaps
**Rejected:** mock model with `MagicMock` + `fake_encode`

**Why:** A mock that already normalizes vectors makes `test_vectors_are_unit_normalized` pass even if `normalize_embeddings=True` is removed from `embed()` — it tests the mock, not the code. The only test that actually means something for an embedding wrapper is semantic: similar texts should score higher than dissimilar ones. That requires the real model. Since CI only runs ruff (not pytest), the 570MB download never hits CI. Run manually with `uv run pytest tests/test_embedder.py -v -m integration` before any model swap.

**When to revisit:** If CI ever adds a pytest step, exclude integration tests with `-m "not integration"` rather than letting them run on every push.

---

## Scraper: isolate site/careers/news failures per-source, not per-company

**Chosen:** each source (site urls, careers, news) wrapped in its own try/except inside `scrape_company()`, logged and skipped independently
**Rejected:** one try/except around the whole company in `ingest.py`

**Why:** Site/careers/news are meant to be independent — a bot-blocked domain shouldn't cost you its news coverage too — but the code only isolated them in config, not in execution. One source raising an exception silently killed every source scraped after it in the same function call. Found via Twaice: a Cloudflare-blocked `/careers` fetch was silently zeroing out its news as well, even though news comes from an entirely separate, unblocked source (Google News RSS).

**When to revisit:** Never for this pattern — always isolate independent I/O sources. Revisit if a fourth source type is added; make sure it gets its own try/except too.

---

## Entity extraction: parse NewsMention deterministically, reserve the LLM for FundingRound

**Chosen:** `extract_news_mentions()` splits `scraper._clean_rss`'s fixed Title/Link/Published/Source blocks with plain string parsing — no LLM call. `extract_funding_rounds()` is the only news-page function that calls an LLM, to classify *which* articles describe a new funding round.
**Rejected:** one combined LLM call per news page returning both NewsMention and FundingRound rows

**Why:** The RSS feed already arrives fully structured — that's why `_clean_rss` keeps it as one block per article instead of collapsing it to plain text. There's nothing for an LLM to infer for basic mention data (title/link/date/source), so spending a call on it adds cost, latency, and a fabrication surface for zero benefit. Funding classification is the one genuinely subjective task (does this headline describe a NEW raise, and what round/amount/investors?) — the only place an LLM earns its keep on a news page.

**When to revisit:** If Google News RSS ever changes its feed format, `_parse_news_blocks()` needs to change with it — same coupling `_clean_rss` already has to that format.

---

## Extractor: minimum-content and video-embed guards around the LLM call

**Chosen:** `MIN_CAREERS_WORD_COUNT` skips the LLM call entirely on JS-only shell pages; a `_VIDEO_HOSTS` check strips postings whose only link is a YouTube/Vimeo embed after the LLM call
**Rejected:** trusting the system prompt alone ("skip navigation links, benefits copy, testimonials...")

**Why:** Observed both failure modes in real scraped data. Augmented Industries' careers page is a 13-word JS shell ("enable JavaScript to run this app") — given that, the LLM hallucinated three fake postings with `example.com` URLs instead of reporting nothing found. Isar Aerospace's careers page has a "meet your future colleagues" employee-testimonial section with video embeds; the LLM classified the video captions as open roles despite an explicit prompt instruction not to. Prompt wording alone isn't reliable enough on a small/fast model (`llama-3.1-8b-instant`) — cheap deterministic guards catch what the prompt doesn't.

**When to revisit:** If job-posting precision still isn't good enough after these guards — e.g. ClearOps's FAQ and interview-process copy is still occasionally misclassified because its real job links live behind a JS-rendered Personio subdomain the static scraper can't see at all. The next fix there is ATS-aware scraping (hit Personio/Greenhouse's public listing APIs directly) rather than more prompt tuning.

---

## Extractor: retry LLM calls on transient Groq errors

**Chosen:** `tenacity` retry (3 attempts, exponential backoff) around `_chat_json()` for `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`
**Rejected:** single attempt — lose the page's extraction on any transient failure

**Why:** Same rationale as the existing scraper HTTP retry decision above. Groq's free tier rate-limits and occasionally 5xxs. Without a retry, one blip permanently drops a company's job/funding data until the next full ingest run (which can hit the same blip again).

**When to revisit:** Never for the retry itself. Groq's 413 "request too large" (tokens-per-minute limit) errors are deliberately NOT retried — retrying doesn't help when the payload itself exceeds the limit. Seen on companies with heavy news coverage (VoiceLine, Isar Aerospace); the real fix is truncating or chunking news text before the funding-extraction call, not yet implemented.

---

## Pipeline tests: mock retrieve() and generate(), not the clients

**Chosen:** `patch("munich_intel.pipeline.retrieve", return_value=[...])` and `patch("munich_intel.pipeline.generate", ...)` — replace the two functions the pipeline calls, not Qdrant or Groq directly
**Rejected:** spinning up a real Qdrant instance or making real Groq API calls in tests

**Why:** Pipeline tests check the wiring contract — that `answer()` returns `{"answer": str, "sources": list}`, that `company_name` is renamed to `company` before being returned, that empty retrieval returns the fallback string. None of that requires a real database or LLM. Patching at the function boundary (retrieve/generate) rather than the service boundary (Qdrant/Groq) keeps tests fast and focused. The most valuable test is `test_answer_sources_are_correctly_formatted` — it catches the `company_name → company` key rename breaking, which would silently break the frontend's source display without this test.

**When to revisit:** Phase 2, when re-ranking is added between retrieve() and generate() in pipeline.py — add a test that verifies re-ranked order is passed to generate(), not the original retrieval order.

---

## Graph: build Investor nodes ad hoc from FundingRound.investor_names, don't persist them

**Chosen:** `graph.py`'s `build_graph()` dedups `FundingRound.investor_names` strings (normalized: stripped, lowercased) into `Investor` nodes at graph-build time
**Rejected:** a persisted `data/entities/{slug}_investors.json` file, written by the extractor like the other three entity types

**Why:** `entities.py` already defines an `Investor` pydantic model (graph-ready by design, per its own docstring), but nothing populates it — `FundingRound.investor_names` is just a list of strings. Persisting Investor as its own extraction step would mean another LLM call (or a second pass over the same funding-round text) before the graph even exists. Building the node in-memory from data already on hand costs nothing and is trivially re-derivable, so there's no need to persist it — this can change later without touching existing entity files.

**When to revisit:** If investor identity needs to survive independently of any one funding round (e.g. an investor detail page, or co-investment clique analysis — parked in VISION.md), promote this to a real extraction step with its own persisted file. Also worth revisiting because investor_names is empty on all but one of the 25 real funding rounds extracted so far — the funding-round LLM prompt rarely fills it in, so today's investor graph is close to empty regardless of where the nodes come from.

---

## JobPosting gets a `scraped_at` first-seen date + append-on-save, instead of a full snapshot mechanism

**Chosen:** `JobPosting.scraped_at` (required, stamped from `ScrapedPage.scraped_at` at extraction time) plus `extractor._save_jobs()`, which merges new postings into the existing `{slug}_jobs.json` by URL rather than overwriting it — an existing row's URL wins, so its original `scraped_at` and fields survive a re-scrape, and only genuinely new URLs get appended
**Rejected:** VISION.md step 5's "append a timestamped run" — a full snapshot mechanism that copies the whole graph/entity set on every run

**Why:** VISION.md's build order put a snapshot mechanism (step 5) before the eval harness (step 6), reasoning that the momentum arc needs to compare state over time. But `FundingRound.announced_on` and `NewsMention.published_on` are already real, reliable dates (funding is a dated event; news RSS carries publish dates) — a single graph snapshot already answers beats 1 and 3 of the arc. The only place a time signal was actually missing is `JobPosting`: checked the real extracted data and only 1 of 77 postings has a real `posted_on` — career pages essentially never state one. So the only entity that needed a "when did we observe this" mechanism was `JobPosting`, and a per-row first-seen date achieves that without duplicating the whole graph on every run. `NewsMention`/`FundingRound` still overwrite their files per run, unchanged — that's intentional, not an oversight, since they don't need it.

**When to revisit:** If beat 4 ("does activity fade after a spike?") ends up needing more than job-posting counts over time — e.g. tracking whether a `NewsMention` or `FundingRound` set itself changes shape across runs — extend the same append-and-dedup pattern to those save paths rather than reaching for a full timestamped-snapshot mechanism. Also revisit if a listing can legitimately change URL (re-posted under a new link) — today that reads as a brand-new posting with a fresh `scraped_at`, which is the correct behavior for "when did we first see this specific listing" but would double-count in a "how many total roles" metric.

---
