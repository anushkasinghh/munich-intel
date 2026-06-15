"""CLI entrypoint: python scripts/ingest.py [--company SLUG]

Runs scrape → chunk → embed → index for one or all companies.
Does not require FastAPI to be running.
"""

import argparse
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.config import settings
from munich_intel.indexer import ingest, setup_collection
from munich_intel.scraper import scrape_company

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Munich startup data into Qdrant.")
    parser.add_argument("--company", metavar="SLUG", help="Ingest a single company by slug.")
    args = parser.parse_args()

    config = yaml.safe_load(Path("companies.yaml").read_text())
    companies = config["companies"]

    if args.company:
        companies = [c for c in companies if c["slug"] == args.company]
        if not companies:
            console.print(f"[red]No company with slug '{args.company}' found in companies.yaml.[/red]")
            sys.exit(1)

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    setup_collection(client, settings.collection_name)

    table = Table(title="Ingest Results", show_lines=True)
    table.add_column("Company", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")

    total_pages = 0
    total_chunks = 0

    for company in companies:
        if company.get("skip"):
            table.add_row(company["name"], "[yellow]skipped[/yellow]", "-", "-")
            continue

        console.print(f"Scraping [cyan]{company['name']}[/cyan]...")
        try:
            pages = scrape_company(company)
        except Exception as exc:
            table.add_row(company["name"], f"[red]scrape error: {exc}[/red]", "0", "0")
            continue

        chunks_for_company = 0
        for page in pages:
            try:
                n = ingest(page, client, settings.collection_name)
                chunks_for_company += n
            except Exception as exc:
                console.print(f"  [red]index error for {page.url}: {exc}[/red]")

        total_pages += len(pages)
        total_chunks += chunks_for_company
        table.add_row(company["name"], "[green]ok[/green]", str(len(pages)), str(chunks_for_company))

    console.print(table)
    console.print(
        f"\n[bold]Done.[/bold] {total_pages} page(s), {total_chunks} chunk(s) indexed "
        f"into [cyan]{settings.collection_name}[/cyan]."
    )


if __name__ == "__main__":
    main()
