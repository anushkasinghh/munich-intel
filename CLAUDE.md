# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Orientation

Munich Intel is a V1 RAG chatbot over scraped Munich startup pages, now growing a V2
entity-extraction pipeline on top: raw scraped text -> structured entities
(`Company`, `FundingRound`, `JobPosting`, `NewsMention`) -> a graph -> eval questions
about whether Munich AI/deep-tech momentum is real. See [VISION.md](VISION.md) for V2
scope and build order, [DECISIONS.md](DECISIONS.md) for architecture rationale and
tradeoffs, [README.md](README.md) for setup and project structure.

## Current V2 state (as of 2026-08-16)

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
- Not started: eval harness for the 4-beat momentum question arc in VISION.md
  (step 6) — the next step.
- `companies.yaml` has 21 companies. VISION.md's own build order says finish steps
  1–6 (extraction -> graph -> eval) on this set before scaling company count further.

## Known issues to be aware of

- **Embedding pipeline is broken in this environment.** `EMBEDDING_MODEL_REVISION=`
  (blank, not unset) in `.env` makes `sentence-transformers` try to resolve a
  revision over the network on every `load_model()` call, which fails here even
  though the model itself is already cached locally (`OSError: We couldn't connect
  to huggingface.co...`). Every ingest run has been silently producing 0 indexed
  chunks as a result. Unrelated to entity extraction, not yet fixed.
- **Job-posting extraction has a real precision ceiling** that prompt tuning and
  guards (`MIN_CAREERS_WORD_COUNT`, `_VIDEO_HOSTS`) only partly close. ClearOps's
  actual job listings live behind a JS-rendered Personio subdomain the static
  scraper never reaches — its scraped careers page text is entirely FAQ/marketing
  copy with no real per-job links, so the LLM has nothing solid to work from. The
  next real fix is ATS-aware scraping (hit Personio/Greenhouse's public listing
  APIs directly for companies that use them) rather than more prompt tuning.
- **Groq free-tier TPM limit (6000 tokens/min)** can 413 `extract_funding_rounds`
  for companies with heavy news coverage (seen on VoiceLine, Isar Aerospace — their
  Google News RSS feed alone exceeds the per-request budget). Not retried on
  purpose — retrying doesn't help when the payload itself is too large. Needs
  truncating/chunking the news text before that call; not yet implemented.
