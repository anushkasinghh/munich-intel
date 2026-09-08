"""CLI entrypoint: python scripts/list_pages.py [slug ...] [--grep TEXT]

Prints the exact URLs held in data/raw/, grouped by company. The labelling aid for
data/eval/questions.yaml: copy URLs from here rather than typing them, because
`normalize_url` does not strip "www." or unify http/https, so a plausible-looking
but wrong URL scores 0.0 while appearing correctly labelled.

    python scripts/list_pages.py                  # every company
    python scripts/list_pages.py konux navvis     # just these
    python scripts/list_pages.py --grep fusion    # companies whose text mentions it

--grep searches the scraped page text, which is how to find candidates for a
multi-company question without guessing from memory. It reports where the match
occurred, not whether the page actually answers anything — that judgement is the
labelling, and it is the part a machine must not do.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

RAW_DIR = Path("data/raw")

console = Console()


def load_pages() -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RAW_DIR.glob("*.json")):
        page = json.loads(path.read_text())
        pages[page["company_slug"]].append(page)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", help="company slugs (default: all)")
    parser.add_argument("--grep", help="only companies whose scraped text matches this regex")
    args = parser.parse_args()

    pages = load_pages()
    if not pages:
        console.print(f"[red]No scraped pages in {RAW_DIR}[/red]")
        return

    slugs = args.slugs or sorted(pages)
    missing = [slug for slug in slugs if slug not in pages]
    if missing:
        console.print(f"[yellow]No pages for: {', '.join(missing)}[/yellow]")

    table = Table(title=f"Pages in {RAW_DIR} — copy URLs from here, verbatim")
    table.add_column("Company", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Words", justify="right")
    if args.grep:
        table.add_column("Hits", justify="right")
    table.add_column("URL", overflow="fold")

    pattern = re.compile(args.grep, re.I) if args.grep else None
    shown = 0
    for slug in slugs:
        for page in sorted(pages.get(slug, []), key=lambda p: p["source_type"]):
            hits = len(pattern.findall(page["page_text"])) if pattern else 0
            if pattern and not hits:
                continue
            row = [slug, page["source_type"], str(page["word_count"])]
            if pattern:
                row.append(str(hits))
            row.append(page["url"])
            table.add_row(*row)
            shown += 1

    if not shown:
        console.print(f"[yellow]Nothing matched {args.grep!r}[/yellow]")
        return
    console.print(table)


if __name__ == "__main__":
    main()
