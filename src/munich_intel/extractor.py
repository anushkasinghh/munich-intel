"""LLM extraction: scraped page text -> entities (V2 step 3-4).

Career pages have no fixed layout, so an LLM reads the text and decides what's an
actual open posting. News pages are the opposite: scraper._clean_rss already splits
Google News RSS into one fixed Title/Link/Published/Source block per article, so
NewsMention rows are recovered by parsing that structure directly — no LLM, no
hallucination risk, no cost. The one genuinely inferential task left is deciding
*which* of those articles report a funding round, which is what the LLM is for here.
"""

import json
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import ollama
from groq import APIConnectionError, APITimeoutError, Groq, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from munich_intel.config import settings
from munich_intel.entities import FundingRound, JobPosting, NewsMention
from munich_intel.scraper import ScrapedPage

logger = logging.getLogger(__name__)

ENTITIES_DIR = Path("data/entities")

# Below this, a page is a JS-only shell ("enable JavaScript to run this app") rather
# than real content — observed the LLM hallucinate full fake postings (example.com
# URLs, invented cities) when given text that thin instead of reporting nothing found.
MIN_CAREERS_WORD_COUNT = 25

# A "posting" whose only link is a video embed is an employee-testimonial clip, not an
# application link — observed the LLM classify Isar Aerospace's "meet your future
# colleagues" video captions as open roles despite prompt instructions not to. Caught
# here deterministically rather than trusted to prompt wording alone.
_VIDEO_HOSTS = {"youtube.com", "www.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com", "vimeo.com", "www.vimeo.com"}

_SYSTEM_PROMPT = """\
You extract OPEN JOB POSTINGS from the text of a company's careers page. The text \
may contain inline links shown as "Link Text [https://url]" — use that URL for a \
posting's `url` field when one is present right next to it.

A careers page usually mixes several kinds of content. Only one kind counts as a posting:
- An actual open role with its own title (e.g. "Senior ML Engineer"), usually with its \
own application link.

None of the following count, even if they mention job-title-sounding words:
- Employee testimonial or "meet the team" sections — often paired with a person's first \
name and a video link (e.g. a youtube.com/vimeo.com embed) rather than an application link.
- Benefits, perks, or culture bullet lists (e.g. "Career Growth", "Mentoring Program", \
"Company Events", "Feedback Cycles").
- Generic "join us" / "life at X" copy, and navigation links.

Respond with a JSON object of the form {"postings": [...]}. Each item has:
- "title": the job title (string, required)
- "url": the direct application link, if one appears in the text (string or null)
- "posted_on": the date it was posted, as YYYY-MM-DD, if stated (string or null)
- "location": the job's location as a plain place name only — never a link (string or null)

If you're not confident an item is an actual open position rather than testimonial or \
benefits content, leave it out. If the page lists no open positions, return {"postings": []}.\
"""

_FUNDING_SYSTEM_PROMPT = """\
You read a company's recent news headlines from Google News. The text is a list of \
articles, each shown as a block:
Title: ...
Link: ...
Published: ...
Source: ...

Find only the articles that report the company announcing a NEW funding round (e.g. \
pre-seed, seed, Series A/B/C/D+, or a grant). Most headlines won't qualify — skip \
product launches, hires, partnerships, awards, and opinion pieces.

Respond with a JSON object of the form {"funding_rounds": [...]}. Each item has:
- "round_type": one of "pre-seed", "seed", "series-a", "series-b", "series-c", \
"series-d+", "grant", "other"
- "announced_on": the date, as YYYY-MM-DD, taken from that article's Published field
- "amount_eur": the amount in EUR as a plain number if stated (convert other \
currencies approximately), else null
- "investor_names": investors named in the headline, as a list (empty if none named)
- "source_url": that article's Link field, copied exactly

If no article describes a funding round, return {"funding_rounds": []}.\
"""


@lru_cache(maxsize=1)
def _groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# Groq's free tier rate-limits and occasionally 5xxs — retry those like the scraper
# retries network errors, rather than losing the whole page's extraction to one blip.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
)
def _chat_json(system_prompt: str, page_text: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": page_text},
    ]
    if settings.llm_provider == "groq":
        resp = _groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
    else:
        resp = ollama.chat(model=settings.ollama_model, messages=messages, format="json")
        content = resp["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Extraction returned invalid JSON, skipping")
        return {}


def _raw_postings(page_text: str) -> list[dict]:
    return _chat_json(_SYSTEM_PROMPT, page_text).get("postings", [])


def _raw_funding_rounds(page_text: str) -> list[dict]:
    return _chat_json(_FUNDING_SYSTEM_PROMPT, page_text).get("funding_rounds", [])


def _save(entities: list[BaseModel], company_slug: str, suffix: str) -> None:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    path = ENTITIES_DIR / f"{company_slug}_{suffix}.json"
    payload = [e.model_dump(mode="json") for e in entities]
    path.write_text(json.dumps(payload, indent=2))


def _save_jobs(postings: list[JobPosting], company_slug: str) -> None:
    """Merge into the existing file instead of overwriting it.

    Career pages rarely state a real `posted_on` (see JobPosting docstring in
    entities.py), so `scraped_at` — the date we first saw a listing — is the only
    signal momentum questions ("did postings rise after a funding round?") can use.
    That signal only survives if re-running the pipeline adds new listings rather
    than replacing the file each time, so existing rows win on URL collision.
    """
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    path = ENTITIES_DIR / f"{company_slug}_jobs.json"
    existing = [JobPosting(**row) for row in json.loads(path.read_text())] if path.exists() else []
    seen_urls = {str(p.url) for p in existing}
    merged = existing + [p for p in postings if str(p.url) not in seen_urls]
    path.write_text(json.dumps([e.model_dump(mode="json") for e in merged], indent=2))


def _parse_news_blocks(page_text: str) -> list[dict]:
    """Split scraper._clean_rss's "Key: value" blocks back into per-article dicts."""
    blocks = []
    for raw_block in page_text.split("\n---\n"):
        fields = {}
        for line in raw_block.splitlines():
            key, sep, value = line.partition(": ")
            if sep:
                fields[key] = value
        if fields:
            blocks.append(fields)
    return blocks


def extract_job_postings(page: ScrapedPage) -> list[JobPosting]:
    if page.source_type != "careers" or page.word_count < MIN_CAREERS_WORD_COUNT:
        return []

    postings = []
    for raw in _raw_postings(page.page_text):
        raw = dict(raw)
        raw["company_slug"] = page.company_slug
        raw["scraped_at"] = datetime.fromisoformat(page.scraped_at).date().isoformat()

        url = raw.get("url")
        if url and urlparse(url).hostname in _VIDEO_HOSTS:
            logger.warning("Skipping video-embed 'posting' (likely a testimonial) for %s: %r", page.company_slug, raw)
            continue
        # A posting the LLM couldn't find a discoverable link for still needs a real,
        # live `url` to satisfy the schema — fall back to the careers page itself.
        if not url:
            raw["url"] = page.url

        location = raw.get("location")
        if location and " [http" in location:
            # LLM occasionally leaves the "Text [url]" link marker in a non-url field
            # instead of stripping it — trim rather than reject the whole posting for it.
            raw["location"] = location.split(" [http", 1)[0].strip()

        try:
            postings.append(JobPosting(**raw))
        except ValidationError:
            logger.warning("Skipping malformed job posting for %s: %r", page.company_slug, raw)
            continue

    _save_jobs(postings, page.company_slug)
    return postings


def extract_news_mentions(page: ScrapedPage) -> list[NewsMention]:
    """Deterministic — no LLM call. See module docstring for why."""
    if page.source_type != "news" or not page.page_text.strip():
        return []

    mentions = []
    for fields in _parse_news_blocks(page.page_text):
        title, link, source = fields.get("Title", ""), fields.get("Link", ""), fields.get("Source", "")
        try:
            published_on = parsedate_to_datetime(fields.get("Published", "")).date()
        except ValueError:
            logger.warning("Skipping news block with unparseable date for %s: %r", page.company_slug, fields)
            continue
        try:
            mentions.append(
                NewsMention(
                    company_slug=page.company_slug,
                    title=title,
                    url=link,
                    published_on=published_on,
                    source=source,
                )
            )
        except ValidationError:
            logger.warning("Skipping malformed news mention for %s: %r", page.company_slug, fields)
            continue

    _save(mentions, page.company_slug, "news")
    return mentions


def extract_funding_rounds(page: ScrapedPage) -> list[FundingRound]:
    if page.source_type != "news" or not page.page_text.strip():
        return []

    rounds = []
    for raw in _raw_funding_rounds(page.page_text):
        raw = dict(raw)
        raw["company_slug"] = page.company_slug
        try:
            rounds.append(FundingRound(**raw))
        except ValidationError:
            logger.warning("Skipping malformed funding round for %s: %r", page.company_slug, raw)
            continue

    _save(rounds, page.company_slug, "funding")
    return rounds
