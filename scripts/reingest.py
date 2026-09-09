"""CLI entrypoint: python scripts/reingest.py [--apply] [--only SLUG,...] [--force]

Syncs the Qdrant collection to whatever is currently in data/raw/. Use it after a
re-scrape — especially one that CHANGES a page's URL, such as editing a company's
`news_query` in companies.yaml, which rewrites the Google News RSS URL and therefore
leaves the old chunks stranded in the index.

Why stranding happens: a point's id is uuid5(company_slug, url, chunk_index), so
re-ingesting the same URL overwrites cleanly, but a NEW url writes new points and
the old ones are never touched. Nothing else in the codebase deletes them, so
without this script the index quietly accumulates chunks of pages that no longer
exist — a "viktor" feed full of football reports outliving the query that fetched it.

Treats data/raw/ as the source of truth, in two directions:
  · a URL on disk but not in the index  -> ingest it
  · a URL in the index but not on disk  -> delete its points (stale)

DRY RUN BY DEFAULT. It prints the plan and changes nothing until you pass --apply,
because the delete half is destructive and a mistaken --only would silently wipe
chunks that are perfectly fine.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.config import settings
from munich_intel.indexer import ingest, setup_collection
from munich_intel.scraper import ScrapedPage

RAW_DIR = Path("data/raw")

console = Console()


def build_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def load_raw(only: set[str] | None) -> list[ScrapedPage]:
    pages = []
    for path in sorted(RAW_DIR.glob("*.json")):
        page = ScrapedPage(**json.loads(path.read_text()))
        if only is None or page.company_slug in only:
            pages.append(page)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    parser.add_argument("--only", help="comma-separated company slugs to limit the sync to")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed and re-ingest every page, not just the ones missing from the index. "
        "Needed when the PAYLOAD schema changes (adding source_type, say) or when "
        "chunk_size changes, since neither alters a page's URL and so neither is "
        "visible as a difference to the default sync.",
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    pages = load_raw(only)
    if not pages:
        console.print(f"[red]No pages in {RAW_DIR}" + (f" for {args.only}" if only else "") + "[/red]")
        return

    client = build_client()
    setup_collection(client, settings.collection_name)

    points, _ = client.scroll(
        settings.collection_name, limit=100_000, with_payload=True, with_vectors=False
    )
    indexed: dict[str, list] = defaultdict(list)
    for point in points:
        indexed[point.payload["url"]].append(point.id)

    on_disk = {page.url for page in pages}
    # Scope the stale check to the same slugs as --only, so a partial run never
    # proposes deleting a company it was not asked to look at.
    stale_ids = [
        pid
        for point in points
        if point.payload["url"] not in on_disk
        and (only is None or point.payload["company_slug"] in only)
        for pid in [point.id]
    ]
    to_ingest = [p for p in pages if args.force or p.url not in indexed]

    plan = Table(title=f"Sync plan for '{settings.collection_name}'" + (" [dim](dry run)[/dim]" if not args.apply else ""))
    plan.add_column("Action", style="cyan")
    plan.add_column("Count", justify="right")
    plan.add_column("Detail", overflow="fold")
    plan.add_row("index points now", str(len(points)), f"{len(indexed)} distinct URLs")
    plan.add_row("pages on disk", str(len(pages)), f"{RAW_DIR}" + (f", filtered to {args.only}" if only else ""))
    plan.add_row(
        "[green]ingest[/green]",
        str(len(to_ingest)),
        "every page (--force)" if args.force else "pages whose URL is not indexed",
    )
    plan.add_row("[red]delete[/red]", str(len(stale_ids)), "points whose URL is no longer in data/raw/")
    console.print(plan)

    if to_ingest:
        detail = Table(title="Pages to ingest")
        detail.add_column("Company", style="cyan")
        detail.add_column("Source", style="dim")
        detail.add_column("Words", justify="right")
        detail.add_column("URL", overflow="fold")
        for page in to_ingest:
            detail.add_row(page.company_slug, page.source_type, str(page.word_count), page.url)
        console.print(detail)

    if not args.apply:
        console.print("\n[yellow]Dry run — nothing written. Re-run with --apply to execute.[/yellow]")
        return

    if stale_ids:
        client.delete(collection_name=settings.collection_name, points_selector=stale_ids)
        console.print(f"[red]deleted[/red] {len(stale_ids)} stale points")

    total = 0
    for page in to_ingest:
        total += ingest(page, client, settings.collection_name)
    console.print(f"[green]ingested[/green] {total} chunks from {len(to_ingest)} pages")

    after = client.get_collection(settings.collection_name).points_count
    console.print(f"\ncollection '{settings.collection_name}' now holds [bold]{after}[/bold] points")


if __name__ == "__main__":
    main()
