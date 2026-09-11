# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Orientation

Munich Intel is a V1 RAG chatbot over scraped Munich startup pages, now growing a V2
entity-extraction pipeline on top: raw scraped text -> structured entities
(`Company`, `FundingRound`, `JobPosting`, `NewsMention`) -> a graph -> eval questions
about whether Munich AI/deep-tech momentum is real. See [VISION.md](VISION.md) for V2
scope and build order, [DECISIONS.md](DECISIONS.md) for architecture rationale and
tradeoffs, [README.md](README.md) for setup and project structure.

## Current V2 state (as of 2026-09-11)

- Scraping (site/careers/news): done. `scrape_company()` isolates each source — one
  failing (e.g. a bot-blocked domain) no longer kills the others.
- `JobPosting` extraction: done, LLM-based, with guards against thin/JS-shell pages
  and video-embed false positives. Precision is still imperfect on some page layouts
  (see below). Each posting now carries a required `scraped_at` date, stamped from
  the page's scrape time — the fallback signal for "is this listing new" since
  `posted_on` is populated in only 1 of 77 real postings (career pages rarely state
  one). `extractor._save_jobs()` merges new postings into `data/entities/{slug}_jobs.json`
  by URL instead of overwriting it, so a re-scrape appends newly-seen postings and
  preserves each existing one's original `scraped_at` (see DECISIONS.md).
- `NewsMention` extraction: done, deterministic — no LLM call, just parses the
  RSS blocks `scraper._clean_rss` already produces.
- `FundingRound` extraction: done, LLM-based — the one genuinely inferential task on
  a news page (deciding which articles describe a new raise).
- Graph build: done. `graph.py`'s `build_graph()` turns `companies.yaml` +
  `data/entities/*.json` into an `nx.DiGraph` — Company/FundingRound/JobPosting/
  NewsMention/Investor nodes, typed edges (`RAISED`, `INVESTED_IN`, `POSTED`,
  `MENTIONED_IN`). `Investor` nodes are built ad hoc from `FundingRound.investor_names`
  (deduped by normalized name) — nothing persists Investor as its own entity file yet.
  `scripts/build_graph.py` prints a sanity-check summary;
  `scripts/visualize_graph.py` renders a pannable/zoomable SVG to `static/graph.html`
  (not yet wired into `api/main.py` — that's VISION step 8, after the eval harness).
  On the real data: 1208 nodes, 1192 edges, but only 1 investor node — funding-round
  extraction rarely captures `investor_names` in practice, worth revisiting.
- VISION.md step 5 ("snapshot mechanism") is covered for the one entity that
  actually needed it — see the `JobPosting.scraped_at` note above and DECISIONS.md
  for why a full timestamped-graph-snapshot mechanism wasn't necessary.
  `NewsMention`/`FundingRound` still overwrite per run, by design (they already
  carry reliable dates).
- Eval harness (VISION.md step 6): a thin first slice is done — jobs only, 5
  companies, no LLM calls, no network. See [EVAL_DESIGN.md](EVAL_DESIGN.md) for the
  scope argument and the 2026-09-01 baseline. `src/munich_intel/eval/metrics.py` is
  pure scoring (URL-keyed P/R/F1, field accuracy), `eval/datasets.py` loads gold and
  predictions through `entities.JobPosting`, `scripts/eval_jobs.py` prints the tables.
  Gold lives in `data/eval/gold_jobs/{slug}.json`, causes in `data/eval/error_tags.yaml`.
  Headline: posting-level precision 0.47 — but bimodal. On ATS careers pages
  (Greenhouse/Personio) the extractor makes zero confirmed errors; on non-ATS pages
  23 of 23 postings are fabrications. That is the case for ATS-aware scraping.
  The 4-beat momentum question arc is still not evaluated — that needs the news and
  funding entities, and a judge, both deliberately out of this slice's scope.
- Retrieval eval (done 2026-09-11, branch `eval-retrieval`): 29 hand-labelled
  questions in `data/eval/questions.yaml`, scored by `scripts/eval_retrieval.py`
  (recall@k, MRR, company diversity, threshold gap; no LLM calls). See
  [EVAL_DESIGN.md](EVAL_DESIGN.md) "Retrieval eval" for the baseline and a delta
  table per change. Headline, at the same ~2,000-token budget: recall 0.35 (k=2)
  → 0.72 (k=10). Shipped: per-page capping in `retriever.cap_per_key`,
  `retrieval_top_k=5`, 200-word chunks, and `chunker.chunk_rss` (one chunk per news
  article with the base64 redirect stripped — those were 64% of news text).
  Measured and NOT shipped: per-company capping (lost recall — cap was on the wrong
  unit) and the cross-encoder reranker in `reranker.py` (best MRR, 33 s/question on
  CPU). Both stay reproducible via eval flags.
- Retrieval index (as of 2026-09-11): Qdrant Cloud collection `munich_intel` holds
  1,110 chunks over 74 URLs and 21 companies — 902 news (one per article, ~39
  tokens), 208 site/careers (200 words, ~600 tokens). bge-m3, 1024-dim, cosine.
  Payload now includes `source_type`, with keyword payload indexes on `source_type`
  and `company_slug` (Qdrant 400s on an unindexed filter). `scripts/reingest.py`
  syncs the index to `data/raw/` in both directions; dry run by default.
  **`.env` overrides `config.py` defaults** (Pydantic BaseSettings) — a chunk_size
  change in code silently did nothing until `.env` was updated too.
- Four news feeds were scraping the wrong entity (viktor → a footballer, allo → a
  biotech, ocell → a climbing crag, quantum-systems → quantum physics; 13% of the
  old index). Fixed with an opt-in `news_query` per company in `companies.yaml`.
  Ocell's feed is still thin (3/9 on-topic) — genuinely little press, not a query
  bug.
- `companies.yaml` has 21 companies. VISION.md's own build order says finish steps
  1–6 (extraction -> graph -> eval) on this set before scaling company count further.

## Known issues to be aware of

- ~~Embedding pipeline is broken in this environment.~~ **Fixed.** The cause was
  `EMBEDDING_MODEL_REVISION=` (blank, not unset) in `.env`, which made
  `sentence-transformers` resolve a revision over the network on every
  `load_model()` call. The key is gone from `.env`, `settings.embedding_model_revision`
  resolves to `None`, the model loads from the local cache and returns 1024-dim
  vectors. Verified 2026-09-07.
- **Job-posting extraction has a real precision ceiling** that prompt tuning and
  guards (`MIN_CAREERS_WORD_COUNT`, `_VIDEO_HOSTS`) only partly close. ClearOps's
  actual job listings live behind a JS-rendered Personio subdomain the static
  scraper never reaches — its scraped careers page text is entirely FAQ/marketing
  copy with no real per-job links, so the LLM has nothing solid to work from. The
  next real fix is ATS-aware scraping (hit Personio/Greenhouse's public listing
  APIs directly for companies that use them) rather than more prompt tuning.
  The eval now quantifies this — see the baseline in EVAL_DESIGN.md.
- **`extractor._save_jobs` never retires a closed listing.** Merging by URL preserves
  `scraped_at` (which is why it merges), but a posting that disappears from the
  careers page stays in `data/entities/` forever, so the job count only ever rises.
  The eval catches three of these. Momentum questions read that count, so this needs
  a `last_seen_at` field before the graph is trusted for trend claims.
- **`data/raw/` keeps only the newest scrape per URL** (files are hash-named by URL,
  so a re-scrape overwrites). That makes some eval misses unattributable: there is no
  way to check what a careers page said at extraction time.
- **Two URL normalizers, on purpose.** `eval.metrics.normalize_url` (jobs eval)
  drops the query string; `eval.retrieval_metrics.normalize_page_url` keeps it.
  Every Google News feed is `news.google.com/rss/search?q=<company>` and differs
  only in the query, so the jobs normalizer collapses all 21 into one key. A test
  pins that they disagree. Do not "simplify" them back together.
- **No similarity-score cutoff can detect an unanswerable question.** Measured with
  two negative controls: under the bi-encoder they outscore several real questions
  (gap −0.073); under the reranker one real question (`hardware-sensor-companies`,
  phrased at a category level no chunk uses) sits below them (gap −0.156). A
  "say I don't know" feature cannot be built on score alone.
- **Groq free-tier TPM limit (6000 tokens/min)** can 413 `extract_funding_rounds`
  for companies with heavy news coverage (seen on VoiceLine, Isar Aerospace — their
  Google News RSS feed alone exceeds the per-request budget). Not retried on
  purpose — retrying doesn't help when the payload itself is too large. Needs
  truncating/chunking the news text before that call; not yet implemented.
