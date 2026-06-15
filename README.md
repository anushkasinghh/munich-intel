# Munich Intel

A local AI research assistant for Munich — scrapes, embeds, and queries local knowledge using Ollama + Qdrant. Runs entirely on your machine, no external APIs.

## Stack

- **Ollama** — local LLM inference (llama3.1:8b)
- **Qdrant** — vector database for semantic search
- **sentence-transformers** — BGE-M3 dense embeddings (Phase 1; see DECISIONS.md)
- **FastAPI** — API layer (2 endpoints in Phase 1)
- **uv** — dependency management and lockfile

## Project Structure

```
munich-intel/
├── .github/
│   └── workflows/ci.yml        ← lint check on push
├── data/
│   └── raw/                    ← scraped JSON files, gitignored
├── src/
│   └── munich_intel/
│       ├── config.py           ← all settings, loaded from .env
│       ├── scraper.py          ← fetches and cleans HTML
│       ├── chunker.py          ← splits text into indexable pieces
│       ├── embedder.py         ← wraps BGE-M3
│       ├── indexer.py          ← writes to Qdrant
│       ├── retriever.py        ← queries Qdrant
│       ├── generator.py        ← calls Ollama
│       └── pipeline.py         ← retriever + generator = RAG answer
├── api/
│   └── main.py                 ← FastAPI, two endpoints only
├── scripts/
│   └── ingest.py               ← CLI: python scripts/ingest.py --company twaice
├── tests/
│   └── test_pipeline.py
├── companies.yaml              ← data source list, not code
├── docker-compose.yml          ← Qdrant only
├── pyproject.toml
└── .gitignore
```

## Design Decisions

**Data sources in YAML, not code** — adding a company means editing `companies.yaml`, not Python.

**`src/` layout** — prevents Python from accidentally importing local files instead of installed packages. Standard for anything you might eventually package or deploy.

**`api/` separate from `src/`** — the FastAPI layer is a delivery mechanism, not business logic. Keeping it outside `src/` makes that boundary explicit.

**Config via environment variables** — `src/munich_intel/config.py` uses Pydantic BaseSettings. Change any value via `.env` or environment variable, never by editing code. See `DECISIONS.md` for the full rationale.

**Phase 1: dense-only search** — sentence-transformers wraps BGE-M3 for dense vectors. Hybrid search (dense + sparse) comes in Phase 2 with FlagEmbedding.

## Setup

### Prerequisites
- Docker
- Ollama with `llama3.1:8b` pulled (`ollama pull llama3.1:8b`)
- Note: `BAAI/bge-m3` (~2GB) downloads from HuggingFace on first run

### Install

```bash
uv sync
uv sync --extra dev
```

### Configure

```bash
cp .env.example .env
# edit .env if needed — defaults work for local dev
```

### Run Qdrant

```bash
docker compose up -d
```

### Ingest a company

```bash
python scripts/ingest.py --company twaice
```

### Start the API

```bash
uvicorn api.main:app --reload
```

## Phase 1 (MVP)
Dense vector search over scraped Munich startup content. Two API endpoints: `POST /query` and `GET /health`.

## Phase 2 (Post-MVP)
Hybrid search (dense + sparse vectors) using FlagEmbedding. See [DECISIONS.md](DECISIONS.md).
