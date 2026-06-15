# data/

This directory is gitignored except for this file.

## data/raw/

Scraped pages written by `scraper.py`. One JSON file per page, named `{slug}_{url_hash}.json`.
Each file is a serialized `ScrapedPage` (company name, slug, URL, cleaned text, word count, scraped timestamp).

Run `python scripts/ingest.py` to populate this directory. Safe to delete and re-run — ingest is idempotent.

## data/qdrant_storage/

Qdrant's on-disk storage when running via Docker Compose (`docker-compose.yml` mounts this path).
Delete to wipe the vector index and start fresh.
