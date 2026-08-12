"""CLI entrypoint: python scripts/ingest.py [--company SLUG]

Runs scrape → chunk → embed → index for one or all companies.
Does not require FastAPI to be running.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.config import settings
from munich_intel.extractor import extract_funding_rounds, extract_job_postings, extract_news_mentions
from munich_intel.indexer import ingest, setup_collection
from munich_intel.scraper import scrape_company

console = Console()


def main() -> None:
    # scrape_company() logs per-source failures (bot-blocked site, JS-only page, etc.)
    # as warnings instead of raising — without this they'd only surface via Python's
    # unformatted last-resort stderr handler.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
    table.add_column("Jobs", justify="right")
    table.add_column("News", justify="right")
    table.add_column("Funding", justify="right")

    total_pages = 0
    total_chunks = 0
    total_jobs = 0
    total_news = 0
    total_funding = 0

    for company in companies:
        # scrape_company() already skips bot-blocked site urls internally while still
        # fetching news for them — don't bypass it here or those companies get zero data.
        console.print(f"Scraping [cyan]{company['name']}[/cyan]...")
        try:
            pages = scrape_company(company)
        except Exception as exc:
            table.add_row(company["name"], f"[red]scrape error: {exc}[/red]", "0", "0", "0", "0", "0")
            continue

        chunks_for_company = 0
        jobs_for_company = 0
        news_for_company = 0
        funding_for_company = 0
        for page in pages:
            try:
                n = ingest(page, client, settings.collection_name)
                chunks_for_company += n
            except Exception as exc:
                console.print(f"  [red]index error for {page.url}: {exc}[/red]")

            if page.source_type == "careers":
                try:
                    jobs_for_company += len(extract_job_postings(page))
                except Exception as exc:
                    console.print(f"  [red]job extraction error for {page.url}: {exc}[/red]")

            if page.source_type == "news":
                try:
                    news_for_company += len(extract_news_mentions(page))
                except Exception as exc:
                    console.print(f"  [red]news extraction error for {page.url}: {exc}[/red]")
                try:
                    funding_for_company += len(extract_funding_rounds(page))
                except Exception as exc:
                    console.print(f"  [red]funding extraction error for {page.url}: {exc}[/red]")

        total_pages += len(pages)
        total_chunks += chunks_for_company
        total_jobs += jobs_for_company
        total_news += news_for_company
        total_funding += funding_for_company

        # scrape_company() now isolates site/careers/news, so a partial result (fewer
        # pages than configured sources) means one of them failed — check the log above
        # for which. Reflect that here instead of trusting the static `skip` flag.
        expected = (
            (0 if company.get("skip") else len(company.get("urls") or []))
            + (1 if company.get("careers_url") else 0)
            + 1  # news is always attempted
        )
        if not pages:
            status = "[red]scrape failed[/red]"
        elif len(pages) < expected:
            status = f"[yellow]partial ({len(pages)}/{expected}, see log)[/yellow]"
        else:
            status = "[green]ok[/green]"
        table.add_row(
            company["name"],
            status,
            str(len(pages)),
            str(chunks_for_company),
            str(jobs_for_company),
            str(news_for_company),
            str(funding_for_company),
        )

    console.print(table)
    console.print(
        f"\n[bold]Done.[/bold] {total_pages} page(s), {total_chunks} chunk(s) indexed, "
        f"{total_jobs} job posting(s), {total_news} news mention(s), {total_funding} funding round(s) "
        f"extracted into [cyan]{settings.collection_name}[/cyan] / [cyan]data/entities/[/cyan]."
    )


if __name__ == "__main__":
    main()
