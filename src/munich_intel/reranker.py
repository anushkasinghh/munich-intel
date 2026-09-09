"""Cross-encoder reranking (Phase 3d).

The retriever is a *bi-encoder*: question and chunk are embedded separately and
compared by cosine. That is what makes it fast — every chunk's vector is computed
once at ingest — and also what limits it, because the chunk's vector is built
without ever seeing the question.

A *cross-encoder* reads the question and the chunk together in one forward pass and
scores the pair. Far more accurate, far too slow to run over a whole collection, so
the standard shape is: retrieve a generous candidate set with the bi-encoder, then
rerank that shortlist with the cross-encoder.

The specific reason it earns its place here: the baseline's cosine scores sit in a
compressed 0.41-0.65 band for everything, answerable or not, and the eval's negative
controls measured a *negative* threshold gap — a question about self-driving cars,
which no company in the corpus does, outscored several real questions. No cutoff can
separate them. The cross-encoder does, decisively: on a spot check it scored
"what is konux" at 0.9993 against Konux's description and 0.000016 against an Arsenal
transfer story. That is the number 3d is trying to move.
"""

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chunks are short after 3c (news ~39 tokens, site ~325), so 512 covers a
# question-plus-chunk pair without truncation.
_MAX_LENGTH = 512


@lru_cache(maxsize=1)
def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    """Load and cache the cross-encoder. ~2.2GB, so never load it per query."""
    return CrossEncoder(model_name, max_length=_MAX_LENGTH)


def rerank_by_scores(hits: list[dict], scores: list[float], top_k: int) -> list[dict]:
    """Reorder hits by given scores, best first, and keep the top k.

    Split out from `rerank` so the ordering logic is unit-testable without loading
    a 2GB model — the same reason `cap_per_key` is separate from `retrieve`.

    Each hit's `score` is REPLACED by its reranker score rather than kept alongside,
    because everything downstream (the eval's threshold gap, any future cutoff)
    should read the score that actually decided the ordering. Carrying both invites
    comparing a cosine against a cross-encoder probability, which are not on the
    same scale and are not comparable.

    Sorting is stable, so ties keep their original retrieval order.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if len(hits) != len(scores):
        raise ValueError(f"got {len(hits)} hits but {len(scores)} scores")

    rescored = [{**hit, "score": float(score)} for hit, score in zip(hits, scores)]
    rescored.sort(key=lambda h: h["score"], reverse=True)
    return rescored[:top_k]


def rerank(query: str, hits: list[dict], top_k: int, model: CrossEncoder | None = None) -> list[dict]:
    """Score every (query, chunk) pair with the cross-encoder and keep the best k."""
    if not hits:
        return []
    model = model or load_reranker()
    scores = model.predict([(query, hit["chunk_text"]) for hit in hits])
    return rerank_by_scores(hits, list(scores), top_k)
