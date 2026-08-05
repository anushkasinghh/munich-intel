from unittest.mock import MagicMock, patch

from munich_intel.scraper import ScrapedPage, _clean_careers_html, scrape_careers, scrape_company

SAMPLE_HTML = """
<html><body>
<nav>Home | About</nav>
<h1>Open Positions</h1>
<a href="/jobs/123">Senior ML Engineer</a>
<a href="https://external.example.com/jobs/456">Backend Engineer</a>
<footer>© Example</footer>
</body></html>
"""


def test_clean_careers_html_inlines_link_targets():
    text = _clean_careers_html(SAMPLE_HTML, "https://example.com/careers")
    assert "Senior ML Engineer [https://example.com/jobs/123]" in text
    assert "Backend Engineer [https://external.example.com/jobs/456]" in text


def test_clean_careers_html_strips_nav_and_footer():
    text = _clean_careers_html(SAMPLE_HTML, "https://example.com/careers")
    assert "Home" not in text
    assert "© Example" not in text


def test_scrape_careers_tags_the_page_as_careers():
    mock_response = MagicMock(text=SAMPLE_HTML)
    mock_response.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response

    with (
        patch("munich_intel.scraper.httpx.Client", return_value=mock_client),
        patch("munich_intel.scraper._save"),
    ):
        page = scrape_careers("https://example.com/careers", "Example", "example", "ai")

    assert page.source_type == "careers"
    assert page.company_slug == "example"
    assert "Senior ML Engineer [https://example.com/jobs/123]" in page.page_text


def _page(**overrides) -> ScrapedPage:
    fields = {
        "company_name": "Reverion",
        "company_slug": "reverion",
        "category": "energy",
        "url": "https://reverion.com",
        "page_text": "text",
        "scraped_at": "2026-01-01T00:00:00Z",
        "word_count": 1,
    }
    fields.update(overrides)
    return ScrapedPage(**fields)


def test_scrape_company_appends_careers_page_when_configured():
    site_page = _page()
    careers_page = _page(url="https://reverion.com/careers", source_type="careers")
    news_page = _page(url="https://news.google.com/rss/search?q=Reverion", source_type="news")
    config = {
        "name": "Reverion",
        "slug": "reverion",
        "category": "energy",
        "urls": ["https://reverion.com"],
        "careers_url": "https://reverion.com/careers",
    }

    with (
        patch("munich_intel.scraper.scrape_page", return_value=site_page),
        patch("munich_intel.scraper.scrape_careers", return_value=careers_page) as mock_scrape_careers,
        patch("munich_intel.scraper.scrape_news", return_value=news_page),
    ):
        pages = scrape_company(config)

    mock_scrape_careers.assert_called_once_with("https://reverion.com/careers", "Reverion", "reverion", "energy")
    assert pages == [site_page, careers_page, news_page]


def test_scrape_company_skips_careers_when_not_configured():
    site_page = _page()
    news_page = _page(url="https://news.google.com/rss/search?q=Reverion", source_type="news")
    config = {"name": "Reverion", "slug": "reverion", "category": "energy", "urls": ["https://reverion.com"]}

    with (
        patch("munich_intel.scraper.scrape_page", return_value=site_page),
        patch("munich_intel.scraper.scrape_careers") as mock_scrape_careers,
        patch("munich_intel.scraper.scrape_news", return_value=news_page),
    ):
        pages = scrape_company(config)

    mock_scrape_careers.assert_not_called()
    assert pages == [site_page, news_page]


def test_scrape_company_scrapes_careers_even_when_site_is_skipped():
    # careers_url often lives on a separate ATS domain (Personio, Greenhouse, ...) that
    # isn't bot-blocked even when the company's own marketing site is.
    careers_page = _page(
        company_name="Twaice",
        company_slug="twaice",
        category="battery-analytics",
        url="https://twaice.jobs.personio.com",
        source_type="careers",
    )
    news_page = _page(
        company_name="Twaice",
        company_slug="twaice",
        category="battery-analytics",
        url="https://news.google.com/rss/search?q=Twaice",
        source_type="news",
    )
    config = {
        "name": "Twaice",
        "slug": "twaice",
        "category": "battery-analytics",
        "urls": [],
        "careers_url": "https://twaice.jobs.personio.com",
        "skip": True,
    }

    with (
        patch("munich_intel.scraper.scrape_page") as mock_scrape_page,
        patch("munich_intel.scraper.scrape_careers", return_value=careers_page) as mock_scrape_careers,
        patch("munich_intel.scraper.scrape_news", return_value=news_page),
    ):
        pages = scrape_company(config)

    mock_scrape_page.assert_not_called()
    mock_scrape_careers.assert_called_once_with(
        "https://twaice.jobs.personio.com", "Twaice", "twaice", "battery-analytics"
    )
    assert pages == [careers_page, news_page]
