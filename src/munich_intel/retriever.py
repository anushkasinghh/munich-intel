from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer

from munich_intel.embedder import embed


def cap_per_key(hits: list[dict], k: int, per_key: int = 2, key: str = "url") -> list[dict]:
    """Keep at most `per_key` chunks sharing the same `key`, then fill up to k.

    Pure post-processing on an already-ranked list — no Qdrant, no embedding — so it
    is unit-testable on hand-written dicts.

    The problem it solves, measured: "Which Munich startups work on nuclear fusion?"
    returns marvel-fusion at ranks 1-6 with proxima-fusion first appearing at rank 7.
    Raising k does not fix it — multi-company recall went 0.09 at k=2 to only 0.27 at
    k=10, while single-company recall hit 1.00 by k=5. Depth was never the binding
    constraint; repeated chunks of one page occupying every slot was.

    WHICH FIELD THE CAP COUNTS IS THE WHOLE POINT, and the brief's first instinct
    (cap per company) measured WORSE than no capping at all: headline recall@10 fell
    0.65 -> 0.42. Recall counts distinct pages, and most answers here live in two or
    three pages of ONE company, so a per-company cap of 2 makes the second page
    unreachable. Keying on `url` instead fixes the monopoly it was aimed at without
    that side effect: recall@10 0.65 -> 0.75, comparison recall@10 0.62 -> 1.00.
    Hence the default `key="url"`. Full tables in EVAL_DESIGN.md 3b.

    Two passes, both in score order:
      1. take hits whose company is still under its cap, until k are held
      2. if that starved (fewer than k), backfill from what was skipped

    The backfill is deliberate and matters. Without it a question whose answer
    genuinely lives in one place — "What is Konux?", 16 of 29 questions here — would
    come back with 2 chunks when the caller asked for 8, throwing away context budget
    that was already paid for. The cap is meant to stop one page crowding OUT others,
    not to stop it filling space nobody else wants.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if per_key < 1:
        raise ValueError(f"per_key must be >= 1, got {per_key}")

    kept: list[dict] = []
    skipped: list[dict] = []
    seen: dict[str, int] = {}
    for hit in hits:
        if len(kept) >= k:
            break
        value = hit[key]
        if seen.get(value, 0) < per_key:
            seen[value] = seen.get(value, 0) + 1
            kept.append(hit)
        else:
            skipped.append(hit)

    if len(kept) < k:
        kept.extend(skipped[: k - len(kept)])
    return kept


def cap_per_company(hits: list[dict], k: int, per_company: int = 2) -> list[dict]:
    """`cap_per_key` keyed on company_slug. Kept as its own name because it is the
    variant the brief specified, and because the eval records it as a measured
    negative result rather than a discarded idea."""
    return cap_per_key(hits, k, per_key=per_company, key="company_slug")


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
    query_filter: Filter | None = None,
    per_company: int | None = 2,
    overfetch: int = 3,
    cap_key: str = "url",
) -> list[dict]:
    """Top-k chunks for a query, with at most `per_company` from any one company.

    Over-fetches `overfetch * top_k` candidates and caps them down, because the
    capping can only promote a second company if a second company was fetched in the
    first place — at top_k=2 the fusion query's first proxima-fusion chunk sits at
    rank 7, so nothing below the cutoff can be rescued without asking for more.

    `per_company=None` disables capping and restores the plain nearest-neighbour
    behaviour, which is what the eval uses to reproduce the pre-3b baseline.
    """
    query_vector = embed([query], model)[0]

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k * overfetch if per_company else top_k,
        with_payload=True,
        query_filter=query_filter,
    )

    hits = [
        {
            "chunk_text": hit.payload["chunk_text"],
            "company_name": hit.payload["company_name"],
            # The stable identifier: display names drift between scrapes, slugs are
            # what companies.yaml and every entity file agree on. The retrieval eval
            # counts distinct companies per result set with it.
            "company_slug": hit.payload["company_slug"],
            "url": hit.payload["url"],
            "score": hit.score,
        }
        for hit in results.points
    ]
    if per_company is None:
        return hits[:top_k]
    return cap_per_key(hits, top_k, per_key=per_company, key=cap_key)
