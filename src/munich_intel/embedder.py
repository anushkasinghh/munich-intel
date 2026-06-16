import functools

from sentence_transformers import SentenceTransformer

from munich_intel.config import settings

# TODO(phase2): swap SentenceTransformer for FlagEmbedding to add sparse vectors for hybrid search.
# Only this file changes — indexer, retriever, and pipeline are unaffected. See DECISIONS.md.


@functools.lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model, revision=settings.embedding_model_revision)


def embed(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
