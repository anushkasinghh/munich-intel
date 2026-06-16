from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from munich_intel.config import settings
from munich_intel.embedder import load_model
from munich_intel.indexer import ingest, setup_collection
from munich_intel.pipeline import answer, answer_stream
from munich_intel.scraper import scrape_company


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    if settings.qdrant_url:
        app.state.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        app.state.qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    setup_collection(app.state.qdrant, settings.collection_name)
    yield
    app.state.qdrant.close()


app = FastAPI(title="Munich Intel", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def ui():
    return Path("index.html").read_text()


# --- request / response models ---

class IngestRequest(BaseModel):
    company_slug: str | None = None


class IngestResponse(BaseModel):
    pages_scraped: int
    chunks_indexed: int
    skipped: list[str]


class QueryRequest(BaseModel):
    question: str
    company_slug: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


# --- endpoints ---

@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    config = yaml.safe_load(Path("companies.yaml").read_text())
    companies = config["companies"]

    if req.company_slug:
        companies = [c for c in companies if c["slug"] == req.company_slug]
        if not companies:
            raise HTTPException(status_code=404, detail=f"Company '{req.company_slug}' not found in companies.yaml")

    pages_scraped = 0
    chunks_indexed = 0
    skipped: list[str] = []

    for company in companies:
        if company.get("skip"):
            skipped.append(company["slug"])
            continue
        pages = scrape_company(company)
        pages_scraped += len(pages)
        for page in pages:
            chunks_indexed += ingest(page, app.state.qdrant, settings.collection_name)

    return IngestResponse(pages_scraped=pages_scraped, chunks_indexed=chunks_indexed, skipped=skipped)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    query_filter = None
    if req.company_slug:
        query_filter = Filter(
            must=[FieldCondition(key="company_slug", match=MatchValue(value=req.company_slug))]
        )

    result = answer(req.question, app.state.model, app.state.qdrant, settings, query_filter)
    return QueryResponse(**result)


@app.post("/query/stream")
def query_stream_endpoint(req: QueryRequest):
    query_filter = None
    if req.company_slug:
        query_filter = Filter(
            must=[FieldCondition(key="company_slug", match=MatchValue(value=req.company_slug))]
        )

    token_stream, sources = answer_stream(req.question, app.state.model, app.state.qdrant, settings, query_filter)

    def event_stream():
        for token in token_stream:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    try:
        info = app.state.qdrant.get_collection(settings.collection_name)
        points_count = info.points_count
    except Exception:
        points_count = 0
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "points_count": points_count,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision or "latest",
        "llm_provider": settings.llm_provider,
        "groq_model": settings.groq_model if settings.llm_provider == "groq" else None,
    }
