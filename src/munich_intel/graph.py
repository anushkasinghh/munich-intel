"""Build a networkx graph from saved entities (VISION.md step 4).

The extraction step (extractor.py) already writes graph-ready rows to
data/entities/*.json, keyed by company_slug (see entities.py's module docstring).
This module is the first place that actually turns those flat rows into a graph
object: a Company node per company.yaml entry, plus a node per FundingRound/
JobPosting/NewsMention/Investor row, connected by directed edges. Nothing here
scrapes, calls an LLM, or renders anything — that's scraper.py / extractor.py /
visualize_graph.py.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from munich_intel.entities import Company, FundingRound, JobPosting, NewsMention

COMPANIES_PATH = Path("companies.yaml")
ENTITIES_DIR = Path("data/entities")


def _company_node_id(slug: str) -> str:
    return f"company:{slug}"


def _investor_node_id(name: str) -> str:
    # Dedup by normalized name — different funding rounds spell the same investor
    # slightly differently (casing, whitespace) far more often than they name two
    # genuinely different investors that happen to collide once normalized.
    return f"investor:{name.strip().lower()}"


def _load_companies(companies_path: Path) -> list[Company]:
    raw = yaml.safe_load(companies_path.read_text())["companies"]
    companies = []
    for row in raw:
        # companies.yaml has no dedicated "homepage" field — fall back through what
        # it does have. Some skip:true entries even leave `urls` empty in yaml (a
        # comment where a list would be), which parses as None rather than [].
        homepage = (row.get("urls") or [None])[0] or row["careers_url"]
        companies.append(
            Company(
                slug=row["slug"],
                name=row["name"],
                hq=row["hq"],
                category=row["category"],
                homepage=homepage,
            )
        )
    return companies


def _load_rows(entities_dir: Path, slug: str, suffix: str, model: type) -> list[Any]:
    path = entities_dir / f"{slug}_{suffix}.json"
    if not path.exists():
        return []
    return [model(**row) for row in json.loads(path.read_text())]


def build_graph(companies_path: Path = COMPANIES_PATH, entities_dir: Path = ENTITIES_DIR) -> nx.DiGraph:
    graph = nx.DiGraph()

    for company in _load_companies(companies_path):
        slug = company.slug
        company_id = _company_node_id(slug)
        graph.add_node(
            company_id,
            kind="company",
            name=company.name,
            hq=company.hq,
            category=company.category,
            homepage=str(company.homepage),
        )

        for i, round_ in enumerate(_load_rows(entities_dir, slug, "funding", FundingRound)):
            funding_id = f"funding:{slug}:{i}"
            graph.add_node(
                funding_id,
                kind="funding",
                round_type=round_.round_type.value,
                announced_on=round_.announced_on.isoformat(),
                amount_eur=round_.amount_eur,
                source_url=str(round_.source_url),
            )
            graph.add_edge(company_id, funding_id, edge_type="RAISED")

            for name in round_.investor_names:
                if not name.strip():
                    continue
                investor_id = _investor_node_id(name)
                if investor_id not in graph:
                    graph.add_node(investor_id, kind="investor", name=name.strip())
                graph.add_edge(investor_id, funding_id, edge_type="INVESTED_IN")

        for job in _load_rows(entities_dir, slug, "jobs", JobPosting):
            job_id = f"job:{job.url}"
            graph.add_node(
                job_id,
                kind="job",
                title=job.title,
                posted_on=job.posted_on.isoformat() if job.posted_on else None,
                scraped_at=job.scraped_at.isoformat(),
                location=job.location,
            )
            graph.add_edge(company_id, job_id, edge_type="POSTED")

        for news in _load_rows(entities_dir, slug, "news", NewsMention):
            news_id = f"news:{news.url}"
            graph.add_node(
                news_id,
                kind="news",
                title=news.title,
                published_on=news.published_on.isoformat(),
                source=news.source,
            )
            graph.add_edge(company_id, news_id, edge_type="MENTIONED_IN")

    return graph
