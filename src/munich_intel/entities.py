"""V2 entity-relationship schema (VISION.md).

The contract the extraction step (scraped text -> LLM call) must fill in.
Nothing here scrapes, calls an LLM, or builds a graph — that's steps 3-4.

Graph-ready by design: Company.slug is the node id. FundingRound, JobPosting,
and NewsMention reference it via company_slug rather than embedding a nested
Company, so they can become graph edges without restructuring later.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, HttpUrl


class Company(BaseModel):
    slug: str
    name: str
    hq: str
    category: str
    homepage: HttpUrl
    description: str | None = None


class RoundType(str, Enum):
    PRE_SEED = "pre-seed"
    SEED = "seed"
    SERIES_A = "series-a"
    SERIES_B = "series-b"
    SERIES_C = "series-c"
    SERIES_D_PLUS = "series-d+"
    GRANT = "grant"
    OTHER = "other"


class FundingRound(BaseModel):
    company_slug: str
    round_type: RoundType
    announced_on: date
    amount_eur: float | None = None
    investor_names: list[str] = []
    source_url: HttpUrl


class Investor(BaseModel):
    name: str
    kind: str | None = None  # e.g. "vc", "angel", "corporate", "public"


class JobPosting(BaseModel):
    company_slug: str
    title: str
    url: HttpUrl
    posted_on: date | None = None
    location: str | None = None


class NewsMention(BaseModel):
    company_slug: str
    title: str
    url: HttpUrl
    published_on: date
    source: str
