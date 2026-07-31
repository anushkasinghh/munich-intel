import pytest
from pydantic import ValidationError

from munich_intel.entities import Company, FundingRound, JobPosting, NewsMention, RoundType


def _company(**overrides):
    fields = {
        "slug": "acme",
        "name": "Acme",
        "hq": "Munich",
        "category": "ai-platform",
        "homepage": "https://acme.example.com",
    }
    fields.update(overrides)
    return Company(**fields)


def test_company_requires_a_homepage_url():
    with pytest.raises(ValidationError):
        _company(homepage="not-a-url")


def test_company_description_is_optional():
    assert _company().description is None


def test_funding_round_requires_a_valid_round_type():
    with pytest.raises(ValidationError):
        FundingRound(
            company_slug="acme",
            round_type="not-a-real-round",
            announced_on="2026-01-01",
            source_url="https://news.example.com/acme-raises",
        )


def test_funding_round_amount_is_optional():
    round_ = FundingRound(
        company_slug="acme",
        round_type=RoundType.SEED,
        announced_on="2026-01-01",
        source_url="https://news.example.com/acme-raises",
    )
    assert round_.amount_eur is None
    assert round_.investor_names == []


def test_job_posting_requires_company_slug_and_url():
    with pytest.raises(ValidationError):
        JobPosting(title="ML Engineer", url="https://acme.example.com/jobs/1")


def test_news_mention_requires_published_date():
    with pytest.raises(ValidationError):
        NewsMention(
            company_slug="acme",
            title="Acme raises seed",
            url="https://news.example.com/acme-raises",
            source="TechCrunch",
        )
