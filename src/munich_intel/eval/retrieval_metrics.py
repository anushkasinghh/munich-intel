"""Scoring primitives for the retrieval eval — pure functions, zero I/O.

Same discipline as `metrics.py` (the jobs eval): ordered list in, number out, no
disk, no network, no Qdrant. Everything here is testable against hand-computed
examples in tests/test_retrieval_metrics.py.

The unit of comparison is the *normalized page URL*, reusing `metrics.normalize_url`
so both evals agree on what "the same page" means. This is load-bearing, not tidiness:
Qdrant point IDs are `uuid5(slug, url, chunk_index)`, so any change to `chunk_size`
renumbers every chunk and invalidates every ID. Phase 3c changes `chunk_size` on
purpose. URLs survive that; IDs do not. Never key a gold label on an ID or an index.

One consequence: retrieval returns *chunks*, and several chunks can come from one
page. `k` therefore counts chunks (that is what the generator's token budget spends),
while recall counts distinct pages. Five chunks of one page score as one relevant
page found, which is exactly the "one company monopolises the results" failure this
eval exists to measure.
"""

from dataclasses import dataclass

from munich_intel.eval.metrics import normalize_url


@dataclass(frozen=True)
class RecallCounts:
    """Recall at k plus the raw counts it came from.

    Carrying `found`/`total` is what makes an honest overall row possible. Averaging
    per-question recall rates would weigh a question with one relevant page the same
    as one with eight; pooling the counts weighs every relevant page equally. Same
    argument as `PRF1`'s counts in the jobs eval.
    """

    recall: float
    found: int
    total: int
    k: int


def _top_k_urls(retrieved_urls: list[str], k: int) -> list[str]:
    """The first k retrieved URLs, normalized, order preserved, duplicates kept.

    Duplicates are kept because they are the finding: three chunks of one page
    occupying three of five slots is the crowding-out problem, and collapsing them
    here would hide it. Callers that want distinct pages take a set afterwards.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return [normalize_url(url) for url in retrieved_urls[:k]]


def _check_relevant(relevant_urls: set[str]) -> set[str]:
    """Normalize the gold set, refusing an empty one.

    An unlabelled question is a bug, not a zero. Scoring it as 0.0 would quietly
    drag every aggregate down and look like a retrieval regression; raising makes
    the labelling gap impossible to miss. `qa_dataset` filters unlabelled questions
    out *before* they reach here, so this fires only on a real mistake.
    """
    if not relevant_urls:
        raise ValueError(
            "relevant_urls is empty — an unlabelled question cannot be scored. "
            "Label it in data/eval/questions.yaml or exclude it from the run."
        )
    return {normalize_url(url) for url in relevant_urls}


def recall_at_k(retrieved_urls: list[str], relevant_urls: set[str], k: int) -> float:
    """Fraction of the relevant pages that appear in the top k retrieved chunks.

    Note the ceiling: a question with 4 relevant pages can score at most 0.5 at k=2,
    because two chunks cannot name four pages. That is not a flaw in the metric, it
    is the case for raising k — read recall@2 next to recall@10 rather than alone.
    """
    relevant = _check_relevant(relevant_urls)
    found = len(set(_top_k_urls(retrieved_urls, k)) & relevant)
    return found / len(relevant)


def recall_counts_at_k(retrieved_urls: list[str], relevant_urls: set[str], k: int) -> RecallCounts:
    """`recall_at_k` with the counts kept, so pooled rows can be recomputed from sums."""
    relevant = _check_relevant(relevant_urls)
    found = len(set(_top_k_urls(retrieved_urls, k)) & relevant)
    total = len(relevant)
    return RecallCounts(recall=found / total, found=found, total=total, k=k)


def mrr(retrieved_urls: list[str], relevant_urls: set[str]) -> float:
    """Reciprocal rank of the *first* relevant page: 1/rank, or 0.0 if none appears.

    Strictly this is one question's reciprocal rank; MRR is the mean of these across
    questions, which `scripts/eval_retrieval.py` takes. Named `mrr` because that is
    what the metric is called in the eval's output and in every paper.

    Ranks are 1-based over the full retrieved list, not truncated to k — the point is
    to see how far down the first good answer sits, including when it sits below the
    cutoff. Failure mode 3 (konux.com/company at rank 3 with k=2) is exactly a
    reciprocal rank of 0.33 next to a recall@2 of 0.
    """
    relevant = _check_relevant(relevant_urls)
    for rank, url in enumerate(retrieved_urls, start=1):
        if normalize_url(url) in relevant:
            return 1 / rank
    return 0.0


def company_diversity(retrieved: list[dict], k: int) -> int:
    """How many distinct companies the top k chunks come from.

    The metric that proves failure mode 1 fixed. Recall alone cannot: a query that
    returns six marvel-fusion chunks and one proxima-fusion chunk can score decent
    recall while still being unable to answer "which startups work on fusion" — the
    generator only ever sees one company. Diversity counts what recall averages away.

    Keyed on `company_slug` rather than `company_name`, since the slug is the stable
    identifier — display names drift between scrapes, slugs are what companies.yaml
    and every entity file agree on.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    slugs = set()
    for hit in retrieved[:k]:
        slug = hit.get("company_slug")
        if not slug:
            raise ValueError(f"retrieved hit has no company_slug: {hit!r}")
        slugs.add(slug)
    return len(slugs)


def hit_rate(retrieved_urls: list[str], relevant_urls: set[str], k: int) -> bool:
    """Did at least one relevant page make the top k?

    The weakest useful question — "could the generator possibly have answered this?"
    Worth reporting next to recall because they diverge in the way that matters:
    a comparison question with one of two companies retrieved is a hit but only half
    the recall, and it will produce a confidently one-sided answer.
    """
    relevant = _check_relevant(relevant_urls)
    return bool(set(_top_k_urls(retrieved_urls, k)) & relevant)


def top_score(retrieved: list[dict]) -> float:
    """The best similarity score in a result set, or 0.0 if nothing came back.

    Not a quality metric on its own — cosine scores in this index are compressed
    into roughly 0.41-0.65 for everything, which is the point. It is only meaningful
    compared across questions, via `threshold_gap`.
    """
    return max((hit["score"] for hit in retrieved), default=0.0)


def threshold_gap(answerable_top_scores: list[float], unanswerable_top_scores: list[float]) -> float:
    """How much room exists for a relevance cutoff: worst answerable minus best unanswerable.

    A positive gap means some threshold sits between them, so retrieval could learn
    to say "I don't know" — anything scoring below it is a question the corpus
    cannot answer. Zero or negative means no such cutoff exists at any value: the
    best match for a question about self-driving cars scores as high as the best
    match for a question the corpus genuinely answers, so filtering by score would
    discard real answers before it discarded junk. That is failure mode 4 as a
    single number, and it is the number a reranker (Phase 3d) would have to move.

    Returns 0.0 when either side is empty — with nothing to compare, no gap has been
    demonstrated, and claiming one would be the wrong direction to fail in.
    """
    if not answerable_top_scores or not unanswerable_top_scores:
        return 0.0
    return min(answerable_top_scores) - max(unanswerable_top_scores)


def pooled_recall(counts: list[RecallCounts]) -> RecallCounts:
    """Sum found/total across questions and recompute the rate — never average rates.

    An empty list pools to 0.0 rather than raising: a `kind` with no labelled
    questions yet is a legitimate empty row in the table, unlike an unlabelled
    question that claims to be scored.
    """
    if not counts:
        return RecallCounts(recall=0.0, found=0, total=0, k=0)
    k_values = {c.k for c in counts}
    if len(k_values) > 1:
        raise ValueError(f"cannot pool recall at different k: {sorted(k_values)}")
    found = sum(c.found for c in counts)
    total = sum(c.total for c in counts)
    return RecallCounts(
        recall=found / total if total else 0.0, found=found, total=total, k=k_values.pop()
    )
