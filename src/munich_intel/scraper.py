import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

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


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


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


def scrape_company(company_config: dict) -> list[ScrapedPage]:
    if company_config.get("skip"):
        return []
    return [
        scrape_page(url, company_config["name"], company_config["slug"], company_config.get("category", ""))
        for url in company_config["urls"]
    ]
