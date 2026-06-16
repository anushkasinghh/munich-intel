# Munich Intel

An AI research assistant for Munich startups — scrapes company pages, embeds them with BGE-M3, and answers questions using RAG.

**Live demo:** https://huggingface.co/spaces/anushkasinghh/munich-intel

## Stack

- **Groq** — LLM inference (llama-3.1-8b-instant, cloud API)
- **Qdrant Cloud** — hosted vector database (production) / local Docker (dev)
- **BGE-M3** via sentence-transformers — dense embeddings
- **FastAPI** — API + streaming chat UI
- **uv** — dependency management and lockfile
- **HuggingFace Spaces** — deployment (Docker, free CPU tier)

## Project Structure

```
munich-intel/
├── .github/
│   └── workflows/ci.yml        ← lint (ruff) + unit tests on every push
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
│       ├── generator.py        ← calls Groq API
│       └── pipeline.py         ← retriever + generator = RAG answer
├── api/
│   └── main.py                 ← FastAPI: /ingest, /query, /query/stream, /health
├── scripts/
│   └── ingest.py               ← CLI: python scripts/ingest.py --company twaice
├── tests/
│   ├── test_chunker.py         ← invariant unit tests (CI)
│   ├── test_embedder.py        ← integration tests, loads real BGE-M3 (manual only)
│   └── test_pipeline.py        ← wiring contract tests with mocks (CI)
├── companies.yaml              ← data source list, not code
├── DECISIONS.md                ← architecture decision log
├── docker-compose.yml          ← Qdrant only
├── pyproject.toml
└── .gitignore
```

## Design Decisions

**Data sources in YAML, not code** — adding a company means editing `companies.yaml`, not Python.

**`src/` layout** — prevents Python from accidentally importing local files instead of installed packages. Standard for anything you might eventually package or deploy.

**`api/` separate from `src/`** — the FastAPI layer is a delivery mechanism, not business logic. Keeping it outside `src/` makes that boundary explicit.

**Config via environment variables** — `src/munich_intel/config.py` uses Pydantic BaseSettings. Change any value via `.env` or environment variable, never by editing code.

**Phase 1: dense-only search** — sentence-transformers wraps BGE-M3 for dense vectors. Hybrid search (dense + sparse) comes in Phase 2 with FlagEmbedding.

See [DECISIONS.md](DECISIONS.md) for the full rationale on each choice.

## Setup

### Prerequisites
- Docker (for local Qdrant)
- A Groq API key — free at [console.groq.com](https://console.groq.com)
- Note: `BAAI/bge-m3` (~570MB) downloads from HuggingFace on first run

### Install

```bash
uv sync --extra dev
```

### Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY at minimum. Defaults work for local Qdrant dev.
```

### Run Qdrant (local dev)

```bash
docker compose up -d
```

### Start the API

```bash
uv run uvicorn api.main:app --reload
```

### Ingest companies

The `/ingest` endpoint requires the `X-Ingest-Token` header. Generate a secret once and set it as `INGEST_SECRET` in `.env` and in HF Space secrets:

```bash
# Generate a secret
python3 -c "import secrets; print(secrets.token_hex(32))"

# Trigger ingest (all companies)
curl -X POST http://localhost:8000/ingest \
  -H "X-Ingest-Token: your-secret" \
  -H "Content-Type: application/json" \
  -d '{}'

# Or a single company
curl -X POST http://localhost:8000/ingest \
  -H "X-Ingest-Token: your-secret" \
  -H "Content-Type: application/json" \
  -d '{"company_slug": "reverion"}'
```

## Running Tests

```bash
# Unit tests (fast, run in CI)
uv run pytest tests/test_chunker.py tests/test_pipeline.py -v

# Integration tests (loads real BGE-M3 model, ~15s — run manually before model swaps)
uv run pytest tests/test_embedder.py -v -m integration
```

## Phase 1 (MVP)
Dense vector search over scraped Munich startup content. Endpoints: `POST /query`, `POST /query/stream`, `POST /ingest` (auth required), `GET /health`.

## Phase 2 (Post-MVP)
Hybrid search (dense + sparse vectors) using FlagEmbedding. See [DECISIONS.md](DECISIONS.md).
