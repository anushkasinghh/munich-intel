from unittest.mock import MagicMock, patch

from munich_intel.scraper import ScrapedPage, _clean_rss, news_url, scrape_company, scrape_news

SAMPLE_RSS = """<?xml version="1.0"?>
<rss><channel>
<item>
<title>Reverion raises Series B</title>
<link>https://news.example.com/reverion-series-b</link>
<pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
<source url="https://techcrunch.com">TechCrunch</source>
</item>
<item>
<title>Reverion opens Munich office</title>
<link>https://news.example.com/reverion-office</link>
<pubDate>Tue, 02 Jan 2026 00:00:00 GMT</pubDate>
<source url="https://example.com">Example News</source>
</item>
</channel></rss>"""


def test_news_url_quotes_the_company_name():
    url = news_url("1KOMMA5 Grad")
    assert url.startswith("https://news.google.com/rss/search?q=")
    assert "1KOMMA5" in url
    assert "%22" in url  # company name is quoted so multi-word names aren't matched as separate terms


def test_clean_rss_extracts_one_block_per_item():
    text = _clean_rss(SAMPLE_RSS)
    blocks = text.split("\n---\n")
    assert len(blocks) == 2
    assert "Title: Reverion raises Series B" in blocks[0]
    assert "Link: https://news.example.com/reverion-series-b" in blocks[0]
    assert "Source: TechCrunch" in blocks[0]
    assert "Title: Reverion opens Munich office" in blocks[1]


def test_clean_rss_returns_empty_string_when_no_articles():
    assert _clean_rss("<rss><channel></channel></rss>") == ""


def test_clean_rss_tolerates_missing_optional_fields():
    partial = "<rss><channel><item><title>Some headline</title></item></channel></rss>"
    text = _clean_rss(partial)
    assert text.startswith("Title: Some headline")
    assert text.count("Title:") == 1


def test_scrape_news_tags_the_page_as_news():
    mock_response = MagicMock(text=SAMPLE_RSS)
    mock_response.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response

    with (
        patch("munich_intel.scraper.httpx.Client", return_value=mock_client),
        patch("munich_intel.scraper._save"),
    ):
        page = scrape_news("Reverion", "reverion", "energy")

    assert page.source_type == "news"
    assert page.company_slug == "reverion"
    assert "Reverion raises Series B" in page.page_text


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


def test_scrape_company_always_appends_a_news_page():
    site_page = _page()
    news_page = _page(url="https://news.google.com/rss/search?q=Reverion", source_type="news")
    config = {"name": "Reverion", "slug": "reverion", "category": "energy", "urls": ["https://reverion.com"]}

    with (
        patch("munich_intel.scraper.scrape_page", return_value=site_page) as mock_scrape_page,
        patch("munich_intel.scraper.scrape_news", return_value=news_page) as mock_scrape_news,
    ):
        pages = scrape_company(config)

    mock_scrape_page.assert_called_once_with("https://reverion.com", "Reverion", "reverion", "energy")
    mock_scrape_news.assert_called_once_with("Reverion", "reverion", "energy")
    assert pages == [site_page, news_page]


def test_scrape_company_skips_blocked_site_but_still_scrapes_news():
    news_page = _page(
        company_name="Twaice",
        company_slug="twaice",
        category="battery-analytics",
        url="https://news.google.com/rss/search?q=Twaice",
        source_type="news",
    )
    config = {"name": "Twaice", "slug": "twaice", "category": "battery-analytics", "urls": [], "skip": True}

    with (
        patch("munich_intel.scraper.scrape_page") as mock_scrape_page,
        patch("munich_intel.scraper.scrape_news", return_value=news_page) as mock_scrape_news,
    ):
        pages = scrape_company(config)

    mock_scrape_page.assert_not_called()
    mock_scrape_news.assert_called_once_with("Twaice", "twaice", "battery-analytics")
    assert pages == [news_page]
