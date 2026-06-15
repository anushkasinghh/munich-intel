import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from munich_intel.chunker import chunk_text
from munich_intel.config import settings
from munich_intel.embedder import embed, load_model
from munich_intel.scraper import ScrapedPage


def setup_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


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
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)
