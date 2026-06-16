import logging
from collections.abc import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer

from munich_intel.config import Settings
from munich_intel.generator import generate, generate_stream
from munich_intel.retriever import retrieve

logger = logging.getLogger(__name__)


def answer(
    query: str,
    model: SentenceTransformer,
    qdrant_client: QdrantClient,
    settings: Settings,
    query_filter: Filter | None = None,
) -> dict:
    chunks = retrieve(query, model, qdrant_client, settings.collection_name, settings.retrieval_top_k, query_filter)
    if not chunks:
        logger.warning("No chunks retrieved for query: %r", query)
        return {"answer": "No relevant information found.", "sources": []}
    response = generate(query, chunks)
    sources = [{"company": c["company_name"], "url": c["url"]} for c in chunks]
    return {"answer": response, "sources": sources}


def answer_stream(
    query: str,
    model: SentenceTransformer,
    qdrant_client: QdrantClient,
    settings: Settings,
    query_filter: Filter | None = None,
) -> tuple[Iterator[str], list[dict]]:
    chunks = retrieve(query, model, qdrant_client, settings.collection_name, settings.retrieval_top_k, query_filter)
    if not chunks:
        def _empty():
            yield "No relevant information found."
        return _empty(), []
    sources = [{"company": c["company_name"], "url": c["url"]} for c in chunks]
    return generate_stream(query, chunks), sources
