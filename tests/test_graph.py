import json

import pytest
import yaml

from munich_intel.graph import build_graph


@pytest.fixture
def small_dataset(tmp_path):
    companies_path = tmp_path / "companies.yaml"
    companies_path.write_text(
        yaml.dump(
            {
                "companies": [
                    {
                        "name": "Acme",
                        "slug": "acme",
                        "urls": ["https://acme.example.com"],
                        "careers_url": "https://acme.example.com/careers",
                        "category": "ai-platform",
                        "hq": "Munich",
                    },
                    {
                        "name": "Blocked Co",
                        "slug": "blocked-co",
                        "urls": None,
                        "careers_url": "https://blocked.example.com/careers",
                        "category": "energy",
                        "hq": "Munich",
                        "skip": True,
                    },
                ]
            }
        )
    )

    entities_dir = tmp_path / "entities"
    entities_dir.mkdir()
    (entities_dir / "acme_funding.json").write_text(
        json.dumps(
            [
                {
                    "company_slug": "acme",
                    "round_type": "seed",
                    "announced_on": "2026-01-01",
                    "amount_eur": 1_000_000,
                    "investor_names": ["Acme Ventures", "acme ventures "],
                    "source_url": "https://news.example.com/acme-raises",
                }
            ]
        )
    )
    (entities_dir / "acme_jobs.json").write_text(
        json.dumps(
            [
                {
                    "company_slug": "acme",
                    "title": "ML Engineer",
                    "url": "https://acme.example.com/jobs/1",
                    "posted_on": None,
                    "location": "Munich",
                    "scraped_at": "2026-01-01",
                }
            ]
        )
    )
    (entities_dir / "blocked-co_news.json").write_text(
        json.dumps(
            [
                {
                    "company_slug": "blocked-co",
                    "title": "Blocked Co raises seed",
                    "url": "https://news.example.com/blocked-co",
                    "published_on": "2026-01-02",
                    "source": "TechCrunch",
                }
            ]
        )
    )

    return companies_path, entities_dir


def test_every_company_gets_a_node_even_without_entity_data(small_dataset):
    companies_path, entities_dir = small_dataset
    graph = build_graph(companies_path, entities_dir)

    assert graph.nodes["company:acme"]["kind"] == "company"
    assert graph.nodes["company:blocked-co"]["kind"] == "company"


def test_funding_round_connects_company_and_investors(small_dataset):
    companies_path, entities_dir = small_dataset
    graph = build_graph(companies_path, entities_dir)

    assert graph.has_edge("company:acme", "funding:acme:0")
    assert graph.edges["company:acme", "funding:acme:0"]["edge_type"] == "RAISED"
    assert graph.has_edge("investor:acme ventures", "funding:acme:0")


def test_differently_cased_investor_names_dedupe_to_one_node(small_dataset):
    companies_path, entities_dir = small_dataset
    graph = build_graph(companies_path, entities_dir)

    investor_nodes = [n for n, kind in graph.nodes(data="kind") if kind == "investor"]
    assert investor_nodes == ["investor:acme ventures"]


def test_job_and_news_nodes_link_back_to_their_company(small_dataset):
    companies_path, entities_dir = small_dataset
    graph = build_graph(companies_path, entities_dir)

    assert graph.has_edge("company:acme", "job:https://acme.example.com/jobs/1")
    assert graph.has_edge("company:blocked-co", "news:https://news.example.com/blocked-co")


def test_node_and_edge_counts(small_dataset):
    companies_path, entities_dir = small_dataset
    graph = build_graph(companies_path, entities_dir)

    # 2 companies + 1 funding round + 1 investor + 1 job + 1 news = 6
    assert graph.number_of_nodes() == 6
    # RAISED, INVESTED_IN, POSTED, MENTIONED_IN = 4
    assert graph.number_of_edges() == 4
