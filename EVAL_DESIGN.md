# Eval Design

Eval is not a phase after building — it's the instrument that proves a fix landed.
Measure one known-broken component, fix it, re-measure, report the delta.

Order: diagnose the index; hand-label jobs for 5 companies; baseline precision/recall
with an error taxonomy; ATS-aware scraping; re-measure. Then investor extraction
(entity resolution), 20 non-deep-tech control companies, beats 1 and 3.

Why thin: the graph is 19 disconnected components, 92% news mentions, 53 job postings,
1 investor. Fix the data before scaling the measurement.

**Future work** — once data is real and traffic exists: full 21-company gold set, gold
QA bank, calibrated LLM judge, retrieval metrics, ingest manifests, health gates, traces.

---

## Build spec — thin eval (jobs only)

Scope guards: **jobs only**, 5 companies, no LLM calls, no network, no judge, no
thresholds file. Everything below is pure functions plus one CLI.

```
src/munich_intel/eval/
  __init__.py
  metrics.py      # pure, zero I/O — unit-testable against worked examples
  datasets.py     # load gold files through entities.JobPosting
data/eval/
  gold_jobs/{slug}.json     # same schema as data/entities/{slug}_jobs.json
  error_tags.yaml           # {url: tag} for false positives, filled in by hand
scripts/eval_jobs.py        # python scripts/eval_jobs.py [--slugs a,b,c]
tests/test_metrics.py
```

**`metrics.py`**

```python
@dataclass(frozen=True)
class PRF1:
    precision: float; recall: float; f1: float
    tp: int; fp: int; fn: int          # counts, so results aggregate across companies

def normalize_url(url: str) -> str: ...
    # lowercase host, strip query/fragment/trailing slash — job URLs differ only by
    # tracking params across scrapes

def prf1(predicted: set[str], gold: set[str]) -> PRF1: ...
    # zero-division -> 0.0, never NaN; empty gold and empty predicted -> f1 1.0

def match(predicted: list[JobPosting], gold: list[JobPosting]
          ) -> tuple[list[tuple[JobPosting, JobPosting]], list[JobPosting], list[JobPosting]]: ...
    # (matched pairs, false positives, false negatives), keyed on normalize_url(url)

def field_accuracy(matched: list[tuple[JobPosting, JobPosting]], field: str) -> float: ...
    # exact match on matched pairs only; skips pairs where gold's field is None
```

**`datasets.py`** — `load_gold_jobs(slug, gold_dir=Path("data/eval/gold_jobs")) -> list[JobPosting]`
and `load_predicted_jobs(slug, entities_dir=Path("data/entities")) -> list[JobPosting]`.
Both parse through `entities.JobPosting`, so a malformed gold file fails loudly.
Missing file returns `[]` (a company with no postings is a valid zero, not an error).

**`scripts/eval_jobs.py`** — `rich` output in the style of `scripts/build_graph.py`:
per-company P/R/F1 table, an aggregate row from summed tp/fp/fn (not averaged rates),
per-field accuracy for `title`/`location`, and a ranked count of error tags. Prints
every FP and FN URL so the tags in `error_tags.yaml` can be filled in.

**`tests/test_metrics.py`** — hand-computed examples: known tp/fp/fn → known P/R/F1;
URL variants that must normalize equal; both empty-set edge cases; `field_accuracy`
ignoring `None` gold fields. No fixtures from real data.

**Seed the gold set with:** `clearops`, `isar-aerospace`, `augmented-industries`
(known-bad — JS shells, video embeds, ATS behind JS) plus two that look healthy.
Label by reading `data/raw/` for each careers page, not by editing the extractor output.

**Error tags** are derived from what the false positives actually are, not decreed in
advance. Start with an empty vocabulary and add tags as you label.

**Definition of done:** a baseline precision/recall number per company, and a ranked
list of causes. That ranking is the extraction backlog — and the before/after delta
once ATS-aware scraping lands.

---

## Baseline — 2026-09-01

`python scripts/eval_jobs.py`, 5 companies, gold labelled by hand from the careers
`page_text` in `data/raw/` (scraped 2026-08-21) against `data/entities/` (extracted
2026-08-12).

| Company | Gold | Pred | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| augmented-industries | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| clearops | 0 | 15 | 0 | 2 | 0 | 0.00 | 0.00 | 0.00 |
| isar-aerospace | 0 | 8 | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| marvel-fusion | 14 | 12 | 10 | 2 | 4 | 0.83 | 0.71 | 0.77 |
| merantix | 14 | 14 | 13 | 1 | 1 | 0.93 | 0.93 | 0.93 |
| **all (pooled)** | **28** | **29** | **23** | **6** | **5** | **0.79** | **0.82** | **0.81** |

Field accuracy on the 23 correctly-found postings: `title` 0.91, `location` 1.00.
Both title errors are a dropped leading qualifier — `"Manager Technical Recruting"`
for `"(Senior) Manager Technical Recruting"`, `"Praktikum: Marketing (m/w/d)"` for
`"Praktikum: Founders Associate – Marketing (m/w/d)"`.

**Read the pooled row with care.** Matching is keyed on normalized URL, so ClearOps's
14 invented postings collapse to 2 FP keys and Isar's 8 to 1. Counted per posting
rather than per URL, precision is 23/49 = **0.47**, and that is the number that
reflects what the graph actually contains.

### Ranked causes

| Tag | URL keys | Postings |
|---|---|---|
| page-heading-as-job | 1 | 14 |
| video-caption-as-job | 1 | 8 |
| listing-closed-since-extraction | 3 | 3 |
| snapshot-drift-unverifiable | 3 | 3 |
| listing-added-after-extraction | 2 | 2 |
| ats-board-link-as-job | 1 | 1 |

### What the baseline says

1. **The extractor's quality is bimodal, and the split is the page type, not the
   prompt.** On the two companies whose careers URL is a real ATS board (Greenhouse,
   Personio), every single FP and FN is date skew between the 2026-08-12 extraction
   and the 2026-08-21 gold page — zero confirmed extraction errors, 0.91 title
   accuracy. On the two non-ATS pages, 23 of 23 postings are fabrications: page
   headings, hiring-process steps, YouTube testimonial captions, location anchors,
   and one title (`"Senior ML Engineer"`) that appears nowhere in the page text.
   No prompt tuning closes that gap, because those pages contain no jobs to extract.
   ATS-aware scraping is the fix, and it is the only fix that matters.
2. **`_save_jobs`'s merge-on-save leaks closed listings.** Three FPs are postings that
   were real in August and are gone now. Preserving `scraped_at` requires not
   overwriting the file, but nothing currently marks a listing as no longer seen, so
   the job count only ever rises — which quietly corrupts the momentum question the
   graph exists to answer. Needs a `last_seen_at` alongside `scraped_at`.
3. **`data/raw/` keeps only the newest scrape per URL, so three FNs cannot be
   attributed** — "added after extraction" and "the extractor missed it" are
   indistinguishable without the older page text. Archiving raw pages per scrape date
   would make the eval able to answer its own question.
4. **The empty-page guards work.** `augmented-industries` is a 13-word JS shell and
   the extractor correctly returned nothing — the one company scoring a clean 1.00.

### Re-measuring after ATS-aware scraping

Re-run extraction and `python scripts/eval_jobs.py`. The target is ClearOps and Isar
moving off 0.00 without marvel-fusion or merantix regressing. Re-label gold from the
same-day `data/raw/` scrape so the date-skew tags drop out of the taxonomy — until
then, the three skew tags (8 of 11 errors) are noise the extractor is not responsible
for.

---

# Retrieval eval

Scores which *pages* come back for a question and in what order. Entirely separate
from the jobs eval above, which scores entity extraction — the two share only the
discipline, not a line of scoring code (see `normalize_page_url` below for why they
cannot even share a URL normalizer).

`python scripts/eval_retrieval.py`, 29 hand-labelled questions over the live Qdrant
collection. No LLM calls, so it is safe to run in a loop while tuning.

## What is measured, and why each metric is here

| Metric | Question it answers | Why not just recall |
|---|---|---|
| `recall@k` | of the pages that genuinely answer this, how many made the top k | the headline |
| `MRR` | how far down the first good answer sits | separates "missed by one rank" from "absent" |
| `C@k` (company diversity) | how many distinct companies the reader actually saw | recall can be decent while every chunk is one company |
| `threshold gap` | can an unanswerable question be told apart from an answerable one | catches a change that lifts recall by returning more of everything |

`C@k` is reported **only for `multi-company` and `comparison`**. For a
single-company question the ideal diversity is 1 — "What is Konux?" wants Konux —
so averaging it across kinds gives a number that rises both when multi-company
results improve and when single-company results degrade.

Aggregation questions and negative controls are excluded from the headline row.
Neither is answerable by retrieval at any k, so folding them in would make every
future change look smaller than it is.

## Baseline — 2026-09-09

Config: `retrieval_top_k=2`, chunk 512 words / 50 overlap, bge-m3 1024-dim cosine,
no filtering. Index: 158 chunks / 74 URLs / 21 companies.

| Kind | R@2 | R@5 | R@10 | MRR | C@2 | C@5 | C@10 | Qs |
|---|---|---|---|---|---|---|---|---|
| single-company | 0.60 | 1.00 | 1.00 | 0.81 | — | — | — | 16 |
| multi-company | 0.09 | 0.14 | 0.27 | 0.47 | 1.4 | 2.8 | 4.6 | 7 |
| comparison | 0.31 | 0.54 | 0.62 | 1.00 | 1.0 | 1.2 | 2.2 | 4 |
| aggregation | 0.00 | 0.00 | 0.22 | 0.08 | — | — | — | 2 |
| **retrievable (headline)** | **0.35** | **0.58** | **0.65** | **0.77** | **1.2** | **2.1** | **3.6** | **25** |
| all questions | 0.30 | 0.51 | 0.59 | 0.72 | 1.2 | 2.1 | 3.6 | 29 |

Negative controls: worst answerable top-score **0.522**, best unanswerable
top-score **0.553**, **threshold gap −0.031**.

### What the baseline says

1. **Single-company retrieval is already solved by raising k, and nothing else.**
   R@2 0.60 → R@5 **1.00**. Every one of the 16 single-company answers is present
   by rank 5; `top_k=2` is the only thing hiding them. This is the cheapest win
   available and it needs no new machinery.
2. **Multi-company retrieval is NOT fixed by raising k — this is the real finding.**
   R@2 0.09 → R@10 0.27. Ten chunks still surface barely a quarter of the relevant
   pages. Compare that with single-company hitting 1.00 by k=5: the failure is not
   depth, it is that one company's chunks occupy the slots. `climate-energy-companies`
   is the extreme case — 6 distinct companies in the top 10 yet only 1 of 7 labelled
   pages found, meaning retrieval surfaces the right *companies* but the wrong
   *pages*. Per-company capping (3b) targets exactly this; a bigger k alone will not.
3. **Every comparison question returns exactly one company at k=2** (C@2 = 1.0
   across all four). The generator is being asked to compare two things while
   being shown one of them. This is failure mode 2, and C@k is what makes it visible
   — recall alone reads a middling 0.31.
4. **No relevance threshold can exist.** The threshold gap is **negative**: the worst
   answerable question (0.522) scores *below* the best question the corpus cannot
   answer at all (0.553). "Which Munich startups are building self-driving cars?" —
   nothing in the corpus does — outranks several real questions. Any score cutoff
   would discard genuine answers before it discarded junk, so filtering by score is
   off the table and a reranker (3d) is the only lever that could move this.
5. **Aggregation is at the floor and should stay there.** R@2 0.00, MRR 0.08.
   That is the intended result: it sizes the gap the graph work fills.

### Two bugs the eval found before it produced a single score

Both were found while building the instrument, which is the argument for building
it first.

**Four contaminated news feeds (fixed, commit `d4000ab`).** `scraper.news_url` used
the quoted company name as the whole query for every company. Measured on-topic
headline rates: `viktor` 2/100 (Arsenal's Viktor Gyökeres and a golfer), `allo`
2/100 (Allogene Therapeutics), `ocell` 1/21 (a Spanish climbing crag),
`quantum-systems` 11/102 (quantum-physics coverage). That was 22 of 173 chunks —
**13% of the index** — of pure noise. Fixed with an opt-in per-company `news_query`
in `companies.yaml`, re-scraped and re-ingested to 158 chunks. Fixed *before* the
baseline deliberately: a baseline is long-lived, and one taken on a 13%-junk corpus
would have been void the moment the feeds were fixed.

**The jobs eval's URL normalizer cannot be reused here.** `metrics.normalize_url`
strips the query string, which is right for a job listing (`?language=en`,
`?utm_source=...` do not change which job is meant). Applied to page URLs it is
catastrophic: every Google News feed is `news.google.com/rss/search?q=<company>`,
differing *only* in the query, so **all 21 collapse to one key** — 74 distinct pages
became 54. The first baseline run was measurably inflated by it (headline R@2 0.38
vs the true 0.35, aggregation MRR 0.50 vs 0.08) because any news-labelled question
matched any company's feed. `retrieval_metrics.normalize_page_url` keeps the query
and normalizes only host case, trailing slash and fragment. Both normalizers still
key on URL rather than Qdrant point id, which is the property that lets a gold label
survive the `chunk_size` change in 3c.

### Labelling method

Labelled by reading the scraped `page_text` in `data/raw/` — never by running
retrieval and keeping what came back, which would score the system against itself
and always look perfect. 69 labels over 27 questions, every one verified to resolve
to a real page. Source mix is site 38 / careers 15 / news 16, and that balance is
deliberate: an earlier draft of the question set had ~20 site, 4 news and **zero**
careers, which would have scored "drop news, prefer site pages" as a clear win while
breaking every hiring and recent-events answer in the app.

Two questions were reworded during labelling because their pages could not answer
them: OroraTech's careers page lists no vacancies and NavVis's lists benefits rather
than openings, so "what roles are they hiring for" was unanswerable from the index —
a scraping gap, not a retrieval one.

## 3a — `source_type` in the payload (enabling work)

`indexer.ingest()` now writes `source_type` (site | news | careers). `ScrapedPage`
has carried it since the V2 scraper split, but it never reached Qdrant, so nothing
downstream could tell a company's own page from a Google News feed — and news is 39%
of the index.

Re-ingested all 74 pages with `python scripts/reingest.py --force --apply`.
158/158 points carry it: site 69, news 61, careers 28.

| | R@2 | R@5 | R@10 | MRR | C@2 | C@10 |
|---|---|---|---|---|---|---|
| baseline | 0.35 | 0.58 | 0.65 | 0.77 | 1.2 | 3.6 |
| 3a | 0.35 | 0.58 | 0.65 | 0.77 | 1.2 | 3.6 |
| **delta** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |

Identical in every cell, which is the intended result: 3a changes what the payload
*contains*, not what the retriever *does*. It is also a free determinism check —
every page was re-embedded from scratch and every score reproduced exactly.

**What it caught.** Writing a key into the payload is not enough to filter on it.
Qdrant returns `400 Bad Request: Index required but not found for "source_type"`
until a keyword payload index exists, so a source-aware query would have failed at
runtime while the data looked perfectly correct in every payload dump.
`setup_collection` now creates keyword indexes for `source_type` and `company_slug`,
idempotently and on existing collections, since this one predates both keys.
Verified by filtering: site 69 / news 61 / careers 28, and `company_slug='konux'`
returns 6.

This is the whole argument for the phase's "confirm it is queryable" step being
separate from "confirm it is present".

## 3b — per-key capping, and a negative result worth more than the change

The brief specified: retrieve `3*k` candidates, then keep at most 2 chunks per
`company_slug`. Implemented as `retriever.cap_per_key` — a pure function on an
already-ranked list, unit-tested without Qdrant (`tests/test_retriever_capping.py`).

Measured at identical k, all three variants on the same index:

| variant | R@2 | R@5 | R@10 | MRR | C@2 | C@5 | C@10 |
|---|---|---|---|---|---|---|---|
| baseline, no cap | 0.35 | 0.58 | 0.65 | 0.77 | 1.2 | 2.1 | 3.6 |
| **cap 2 per company** (as specified) | 0.35 | **0.40** | **0.42** | 0.73 | 1.2 | **3.3** | **6.6** |
| **cap 2 per page** (shipped) | 0.35 | **0.62** | **0.75** | 0.78 | 1.2 | 2.6 | 4.7 |

### Per-company capping made things worse — this is the finding

It did exactly what it promised on diversity: C@10 nearly doubled, 3.6 → **6.6**.
And it **lost recall everywhere**, including on the multi-company questions it was
designed to fix (R@10 0.27 → 0.18). Single-company recall collapsed 1.00 → **0.60**.

The cause is a unit mismatch. **Recall counts distinct pages; the cap counted
companies.** Sixteen of the 29 questions are single-company, and most are answered by
two or three pages *of the same company* — `konux.com` and `konux.com/company`,
`ororatech.com` and `/about-us`. A cap of 2 per company means Konux can contribute at
most two chunks in total, and if both come from `konux.com` then `konux.com/company`
is unreachable at any k. The cap was starving the exact pages recall was counting.

Diversity went up and answers got worse, which is precisely why both metrics are
reported side by side. A diversity-only report would have called this a success.

### Keying the cap on `url` instead

Same mechanism, same code path, one field changed — and it fixes the monopoly it was
aimed at *without* the side effect, because the original failure was repeated chunks
of one **page**, not one company:

- headline R@10 **0.65 → 0.75**
- comparison R@10 **0.62 → 1.00** — every comparison question now finds every
  labelled page by k=10, where the baseline found under two thirds
- multi-company R@10 0.27 → **0.32**
- single-company **unchanged at 1.00** — no regression
- C@10 3.6 → 4.7 — diversity still improves, just less than the per-company cap
  bought at the cost of correctness

Shipped as the default (`cap_key="url"`). `cap_per_company` is kept as a named
wrapper so the negative result stays reproducible with
`--per-company 2 --cap-key company_slug` rather than becoming a discarded idea
nobody can check.

### `retrieval_top_k` 2 → 5

Justified by the same table: R@2 0.35 vs R@5 0.62, and single-company recall reaching
1.00 at k=5. Not raised to 10, which would add another +0.13, because of the token
budget.

Measured on the live index: chunks average 389 words (~525 tokens) against a 512-word
cap, so k=5 costs roughly **2,600 tokens** of context and k=10 roughly **5,300**,
against Groq's free-tier ~6,000 TPM. k=5 leaves room for the system prompt (~150
tokens) and the answer; k=10 does not, once more than one query lands in a minute.

Worth correcting the old note in `config.py` that said k=5 "asked for ~12800 tokens":
that cannot be one request at this chunk size. TPM is per *minute* and cumulative
across requests, so it was measuring several queries, not one oversized one. k=10
waits for 3c to shrink the chunks.
