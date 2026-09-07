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
