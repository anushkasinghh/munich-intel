import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DATA_DIR = Path("data/raw")

# 4xx errors are permanent — only retry on network/timeout/5xx failures.
_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # No Accept-Encoding — httpx sets this itself and only advertises encodings it can decompress.
    # Manually adding 'br' here causes servers to send Brotli, which httpx can't decode without
    # the optional brotli package, resulting in raw binary in page_text.
}


class ScrapedPage(BaseModel):
    company_name: str
    company_slug: str
    category: str
    url: str
    page_text: str
    scraped_at: str
    word_count: int
    # Distinguishes the company's own site from a news feed or careers page, so
    # extract_entities knows whether to look for Company/FundingRound, NewsMention, or
    # JobPosting facts.
    source_type: Literal["site", "news", "careers"] = "site"


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


def _clean_careers_html(html: str, base_url: str) -> str:
    # Same as _clean_html, but inlines each link's target next to its visible text
    # (e.g. "Senior ML Engineer [https://.../jobs/123]") instead of dropping hrefs via
    # get_text(). Career pages have no fixed layout, so the LLM extractor (not regex)
    # has to find postings itself — it can only recover a JobPosting.url if the link
    # survives cleaning.
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        a.replace_with(f"{a.get_text(strip=True)} [{href}]")
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


def news_url(company_name: str) -> str:
    """Google News RSS search for a company — free, no API key, one formula for every company."""
    query = quote_plus(f'"{company_name}"')
    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _clean_rss(xml: str) -> str:
    # One block per <item>, kept in a fixed Title/Link/Published/Source shape rather than
    # collapsed to plain text — extract_entities (V2 step 4) needs to split this back out
    # into individual NewsMention rows, one per article, not just a single text blob.
    soup = BeautifulSoup(xml, "xml")
    blocks = []
    for item in soup.find_all("item"):
        title = item.title.get_text(strip=True) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        pub_date = item.pubDate.get_text(strip=True) if item.pubDate else ""
        source = item.source.get_text(strip=True) if item.source else ""
        blocks.append(f"Title: {title}\nLink: {link}\nPublished: {pub_date}\nSource: {source}")
    return "\n---\n".join(blocks)


def _save(page: ScrapedPage) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(page.url.encode()).hexdigest()[:8]
    path = DATA_DIR / f"{page.company_slug}_{url_hash}.json"
    path.write_text(page.model_dump_json(indent=2))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
)
def scrape_page(url: str, company_name: str, company_slug: str, category: str = "") -> ScrapedPage:
    with httpx.Client(headers=_HEADERS, timeout=10, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    text = _clean_html(response.text)
    page = ScrapedPage(
        company_name=company_name,
        company_slug=company_slug,
        category=category,
        url=url,
        page_text=text,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        word_count=len(text.split()),
    )
    _save(page)
    return page


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
)
def scrape_news(company_name: str, company_slug: str, category: str = "") -> ScrapedPage:
    url = news_url(company_name)
    with httpx.Client(headers=_HEADERS, timeout=10, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    # A company with zero coverage still returns a valid feed with no <item> tags —
    # that's an empty page_text, not an error, so it's saved like any other result.
    text = _clean_rss(response.text)
    page = ScrapedPage(
        company_name=company_name,
        company_slug=company_slug,
        category=category,
        url=url,
        page_text=text,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        word_count=len(text.split()),
        source_type="news",
    )
    _save(page)
    return page


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
)
def scrape_careers(url: str, company_name: str, company_slug: str, category: str = "") -> ScrapedPage:
    with httpx.Client(headers=_HEADERS, timeout=10, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    text = _clean_careers_html(response.text, url)
    page = ScrapedPage(
        company_name=company_name,
        company_slug=company_slug,
        category=category,
        url=url,
        page_text=text,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        word_count=len(text.split()),
        source_type="careers",
    )
    _save(page)
    return page


def scrape_company(company_config: dict) -> list[ScrapedPage]:
    name = company_config["name"]
    slug = company_config["slug"]
    category = company_config.get("category", "")

    pages = []
    # `skip` marks the company's own domain as bot-blocked (Cloudflare/429/403) — it says
    # nothing about Google News, which is a separate, unblocked source. Skipping it too would
    # leave bot-blocked companies (Twaice, Helsing, Quantum Systems) with zero data at all.
    if not company_config.get("skip"):
        pages.extend(scrape_page(url, name, slug, category) for url in company_config["urls"])
    # careers_url is independent of `skip` too — it's often on a separate ATS domain
    # (Personio, Greenhouse, ...) that isn't bot-blocked even when the main site is.
    careers_url = company_config.get("careers_url")
    if careers_url:
        pages.append(scrape_careers(careers_url, name, slug, category))
    pages.append(scrape_news(name, slug, category))
    return pages
