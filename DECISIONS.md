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
