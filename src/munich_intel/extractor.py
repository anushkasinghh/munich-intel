"""LLM extraction: careers page text -> JobPosting entities (V2 step 3, job postings only).

Career pages have no fixed layout the way the news RSS feed does, so unlike
scraper._clean_rss this can't be split into records with regex — an LLM reads the
page text and decides what's an actual open posting.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import ollama
from groq import Groq
from pydantic import ValidationError

from munich_intel.config import settings
from munich_intel.entities import JobPosting
from munich_intel.scraper import ScrapedPage

logger = logging.getLogger(__name__)

ENTITIES_DIR = Path("data/entities")

_SYSTEM_PROMPT = """\
You extract open job postings from the text of a company's careers page. The text \
may contain inline links shown as "Link Text [https://url]" — use that URL for a \
posting's `url` field when one is present next to it.

Respond with a JSON object of the form {"postings": [...]}. Each item has:
- "title": the job title (string, required)
- "url": the direct link to the job posting, if one appears in the text (string or null)
- "posted_on": the date it was posted, as YYYY-MM-DD, if stated (string or null)
- "location": the job's location, if stated (string or null)

Only include actual open positions — skip navigation links, benefits copy, and \
generic "join us" text. If the page lists no open positions, return {"postings": []}.\
"""


@lru_cache(maxsize=1)
def _groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def _raw_postings(page_text: str) -> list[dict]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
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
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Job posting extraction returned invalid JSON, skipping")
        return []
    return data.get("postings", [])


def _save(postings: list[JobPosting], company_slug: str) -> None:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    path = ENTITIES_DIR / f"{company_slug}_jobs.json"
    payload = [p.model_dump(mode="json") for p in postings]
    path.write_text(json.dumps(payload, indent=2))


def extract_job_postings(page: ScrapedPage) -> list[JobPosting]:
    if page.source_type != "careers" or not page.page_text.strip():
        return []

    postings = []
    for raw in _raw_postings(page.page_text):
        raw = dict(raw)
        raw["company_slug"] = page.company_slug
        # A posting the LLM couldn't find a discoverable link for still needs a real,
        # live `url` to satisfy the schema — fall back to the careers page itself.
        if not raw.get("url"):
            raw["url"] = page.url
        try:
            postings.append(JobPosting(**raw))
        except ValidationError:
            logger.warning("Skipping malformed job posting for %s: %r", page.company_slug, raw)
            continue

    _save(postings, page.company_slug)
    return postings
