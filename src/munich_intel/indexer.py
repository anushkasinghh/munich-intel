import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from munich_intel.chunker import chunk_text
from munich_intel.config import settings
from munich_intel.embedder import embed, load_model
from munich_intel.scraper import ScrapedPage

logger = logging.getLogger(__name__)


# Payload keys that need a Qdrant index before they can be used in a filter.
# Writing a key into the payload is NOT enough: filtering on an unindexed keyword
# returns 400 "Index required but not found", so a source-aware query would fail at
# runtime while the data looked perfectly correct in the payload.
_FILTERABLE_KEYS = ("source_type", "company_slug")


def setup_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # Idempotent, and applied to existing collections too — the collection predates
    # these keys, so creating it is not the only moment they can be needed.
    for key in _FILTERABLE_KEYS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug("payload index for %r already present", key, exc_info=True)


def _point_id(company_slug: str, url: str, chunk_index: int) -> str:
    # Deterministic: re-indexing the same page overwrites rather than duplicates.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company_slug}_{url}_{chunk_index}"))


def ingest(page: ScrapedPage, client: QdrantClient, collection_name: str) -> int:
    chunks = chunk_text(page.page_text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    vectors = embed(chunks, load_model())

    points = [
        PointStruct(
            id=_point_id(page.company_slug, page.url, i),
            vector=vector,
            payload={
                "company_name": page.company_name,
                "company_slug": page.company_slug,
                "url": page.url,
                "chunk_text": chunk,
                "chunk_index": i,
                "scraped_at": page.scraped_at,
                "category": page.category,
                # site | news | careers. ScrapedPage has carried this since the V2
                # scraper split, but it was never written to the payload, so nothing
                # downstream could tell a company's own page from a Google News feed
                # — and news is 39% of the index. Enabling work: writing it changes
                # no behaviour on its own, it just makes source-aware retrieval
                # possible to build and, more importantly, possible to measure.
                "source_type": page.source_type,
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)
