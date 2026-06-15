"""Quick diagnostic: scrape every URL in companies.yaml and report pass/fail."""

from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from munich_intel.scraper import scrape_page

console = Console()


def main() -> None:
    config = yaml.safe_load(Path("companies.yaml").read_text())
    companies = config["companies"]

    table = Table(title="Scraper Test", show_lines=True)
    table.add_column("Company", style="cyan")
    table.add_column("URL", style="white", max_width=50)
    table.add_column("Status", justify="center")
    table.add_column("Words", justify="right")
    table.add_column("Error", style="red", max_width=40)

    passed = failed = 0

    for company in companies:
        if company.get("skip"):
            table.add_row(company["name"], "—", "[yellow]skipped[/yellow]", "-", "skip: true in companies.yaml")
            continue
        for url in company["urls"]:
            try:
                page = scrape_page(url, company["name"], company["slug"], company.get("category", ""))
                table.add_row(company["name"], url, "[green]✓[/green]", str(page.word_count), "")
                passed += 1
            except Exception as e:
                table.add_row(company["name"], url, "[red]✗[/red]", "-", str(e)[:80])
                failed += 1

    console.print(table)
    console.print(f"\n[green]{passed} passed[/green]  [red]{failed} failed[/red]  out of {passed + failed} URLs")


if __name__ == "__main__":
    main()
