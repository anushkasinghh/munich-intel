"""Scoring primitives for the jobs eval — pure functions, zero I/O.

Everything here takes values in and returns values out, so the whole module is
unit-testable against hand-computed examples (tests/test_metrics.py) without
touching disk, the network, or an LLM.

The unit of comparison is the *normalized job URL*, not the title. A job posting
is identified by where it points: two scrapes of the same listing produce the same
URL but may word the title differently, and the extractor's characteristic failure
mode is inventing postings that point at a page heading rather than a real job.
Keying on URL measures exactly that.

One consequence worth knowing: postings that share a URL collapse to one key. On a
page with no per-job links (isaraerospace.com/career) the extractor emits eight
postings all pointing at the same careers URL, and they score as a single false
positive. `duplicate_url_groups` exists to make that collapse visible rather than
silent.
"""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from munich_intel.entities import JobPosting


@dataclass(frozen=True)
class PRF1:
    """Precision/recall/F1 plus the raw counts they were computed from.

    The counts are the point. Rates cannot be averaged across companies without
    lying — a company with 1 posting would weigh as much as one with 40. Carrying
    tp/fp/fn lets the aggregate row be recomputed from summed counts instead
    (a "micro average"), which weighs every posting equally.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def normalize_url(url: str) -> str:
    """Canonical form of a job URL, for comparing gold against predicted.

    Lowercases scheme and host, drops the query string and fragment, and strips a
    trailing slash from the path. The same listing shows up across scrapes with
    different tracking params (`?language=en`, `?utm_source=...`) and sometimes an
    anchor (`#kiruna`); none of those change which job is being pointed at.

    Deliberately does NOT strip `www.` or unify http/https — those distinguish
    genuinely different hosts often enough that silently merging them would hide
    scraper bugs rather than normalize noise.
    """
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def prf1_from_counts(tp: int, fp: int, fn: int) -> PRF1:
    """Precision, recall and F1 from raw counts.

    precision = of what we extracted, how much was real  (tp / (tp + fp))
    recall    = of what was really there, how much we found  (tp / (tp + fn))
    f1        = their harmonic mean — punishes a lopsided score, so an extractor
                that emits one perfect posting and misses thirty cannot look good.

    Zero denominators return 0.0 rather than NaN, so a bad company still produces a
    number the aggregate can absorb. The single exception is all-zero counts:
    predicting no jobs for a company that genuinely has none is a correct answer,
    and scores 1.0 rather than 0.0.

    Separate from `prf1` so the aggregate row can be computed from summed counts.
    Per-company URL sets cannot simply be unioned to get that — pooling the counts
    is the operation we actually mean.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return PRF1(1.0, 1.0, 1.0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision, recall, f1, tp, fp, fn)


def prf1(predicted: set[str], gold: set[str]) -> PRF1:
    """Precision, recall and F1 for two sets of normalized URLs."""
    return prf1_from_counts(
        tp=len(predicted & gold), fp=len(predicted - gold), fn=len(gold - predicted)
    )


def match(
    predicted: list[JobPosting], gold: list[JobPosting]
) -> tuple[list[tuple[JobPosting, JobPosting]], list[JobPosting], list[JobPosting]]:
    """Align predicted postings against gold ones by normalized URL.

    Returns (matched pairs, false positives, false negatives). Matched pairs are
    what `field_accuracy` then grades on title/location — a posting the extractor
    found but mistitled is a different (milder) problem than one it invented, and
    the two should not be summed into one number.

    Where several postings share a URL key the first is kept and the rest dropped;
    `duplicate_url_groups` reports them so the drop is never invisible.
    """
    gold_by_url: dict[str, JobPosting] = {}
    for posting in gold:
        gold_by_url.setdefault(normalize_url(str(posting.url)), posting)

    predicted_by_url: dict[str, JobPosting] = {}
    for posting in predicted:
        predicted_by_url.setdefault(normalize_url(str(posting.url)), posting)

    matched = [
        (predicted_by_url[url], gold_by_url[url])
        for url in predicted_by_url
        if url in gold_by_url
    ]
    false_positives = [p for url, p in predicted_by_url.items() if url not in gold_by_url]
    false_negatives = [g for url, g in gold_by_url.items() if url not in predicted_by_url]
    return matched, false_positives, false_negatives


def field_accuracy(matched: list[tuple[JobPosting, JobPosting]], field: str) -> float:
    """Exact-match accuracy for one field, over matched pairs only.

    Pairs whose gold value is None are skipped: "the labeller did not record a
    location" is not evidence that the extractor got the location wrong. Only
    correctly-found postings are graded, so this never double-counts a miss that
    precision/recall already charged for.

    Returns 0.0 when nothing is comparable — pair with `field_comparable_count` to
    tell "graded 0 of 4" apart from "nothing to grade".
    """
    comparable = [(p, g) for p, g in matched if getattr(g, field) is not None]
    if not comparable:
        return 0.0
    correct = sum(1 for p, g in comparable if getattr(p, field) == getattr(g, field))
    return correct / len(comparable)


def field_comparable_count(matched: list[tuple[JobPosting, JobPosting]], field: str) -> int:
    """How many matched pairs `field_accuracy` actually graded (gold value not None)."""
    return sum(1 for _, g in matched if getattr(g, field) is not None)


def duplicate_url_groups(postings: list[JobPosting]) -> dict[str, list[JobPosting]]:
    """Postings sharing a normalized URL, keyed by that URL — only groups of 2+.

    Not a metric: a diagnostic. A large group means the extractor emitted several
    "jobs" that all point at the same page, which is the signature of a careers page
    with no real per-job links (see CLAUDE.md on ClearOps/Isar). Those collapse to a
    single key during matching, so without this they would score as one FP and the
    scale of the problem would not show up anywhere.
    """
    groups: dict[str, list[JobPosting]] = {}
    for posting in postings:
        groups.setdefault(normalize_url(str(posting.url)), []).append(posting)
    return {url: group for url, group in groups.items() if len(group) > 1}
