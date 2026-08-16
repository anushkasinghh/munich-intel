import json
from unittest.mock import MagicMock, patch

from munich_intel.entities import RoundType
from munich_intel.extractor import extract_funding_rounds, extract_job_postings, extract_news_mentions
from munich_intel.scraper import ScrapedPage


def _page(**overrides) -> ScrapedPage:
    fields = {
        "company_name": "Reverion",
        "company_slug": "reverion",
        "category": "energy",
        "url": "https://reverion.com/careers",
        "page_text": "Senior ML Engineer [https://reverion.com/jobs/123]",
        "scraped_at": "2026-01-01T00:00:00Z",
        "word_count": 50,  # above MIN_CAREERS_WORD_COUNT; the LLM call is mocked so page_text itself is a short stand-in
        "source_type": "careers",
    }
    fields.update(overrides)
    return ScrapedPage(**fields)


def test_extract_job_postings_returns_empty_for_non_careers_page():
    page = _page(source_type="site")
    assert extract_job_postings(page) == []


def test_extract_job_postings_returns_empty_for_blank_page_text():
    page = _page(page_text="   ", word_count=0)
    assert extract_job_postings(page) == []


def test_extract_job_postings_returns_empty_for_js_shell_page():
    # A page thin enough to be a JS-only shell ("enable JavaScript...") rather than
    # real content — the LLM shouldn't even be asked, since it's prone to hallucinating
    # postings on input this sparse instead of reporting nothing found.
    page = _page(page_text="You must enable JavaScript to run this app.", word_count=8)
    assert extract_job_postings(page) == []


def test_extract_job_postings_builds_entities_from_llm_output():
    raw = [{"title": "Senior ML Engineer", "url": "https://reverion.com/jobs/123", "posted_on": "2026-01-01", "location": "Munich"}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert postings[0].company_slug == "reverion"
    assert postings[0].title == "Senior ML Engineer"
    assert str(postings[0].url) == "https://reverion.com/jobs/123"


def test_extract_job_postings_falls_back_to_page_url_when_llm_omits_url():
    raw = [{"title": "Backend Engineer", "url": None}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert str(postings[0].url) == "https://reverion.com/careers"


def test_extract_job_postings_skips_malformed_entries():
    raw = [{"url": "https://reverion.com/jobs/1"}, {"title": "Valid Posting"}]  # first is missing required title
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert postings[0].title == "Valid Posting"


def test_extract_job_postings_skips_video_embed_urls():
    # e.g. Isar Aerospace's "meet your future colleagues" section — employee testimonial
    # clips whose only link is a youtube-nocookie.com embed, not an application link.
    raw = [
        {"title": "Team Leader Composites", "url": "https://www.youtube-nocookie.com/embed/3uR5L8IpcZo"},
        {"title": "Valid Posting", "url": "https://reverion.com/jobs/1"},
    ]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert postings[0].title == "Valid Posting"


def test_extract_job_postings_strips_leftover_link_marker_from_location():
    raw = [
        {
            "title": "Valid Posting",
            "url": "https://reverion.com/jobs/1",
            "location": "Munich [https://reverion.com/careers#munich]",
        }
    ]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page())

    assert postings[0].location == "Munich"


def test_extract_job_postings_saves_results():
    raw = [{"title": "Senior ML Engineer"}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs") as mock_save_jobs,
    ):
        postings = extract_job_postings(_page())

    mock_save_jobs.assert_called_once_with(postings, "reverion")


def test_extract_job_postings_stamps_scraped_at_from_page_scrape_time():
    # posted_on is almost never available from the source page (see JobPosting
    # docstring); scraped_at is what momentum questions rely on instead, so it must
    # come from the page's own scrape time, not e.g. today's date.
    raw = [{"title": "Senior ML Engineer", "url": "https://reverion.com/jobs/123"}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save_jobs"),
    ):
        postings = extract_job_postings(_page(scraped_at="2026-03-14T09:00:00Z"))

    assert postings[0].scraped_at.isoformat() == "2026-03-14"


def test_save_jobs_keeps_existing_row_on_url_collision(tmp_path):
    from munich_intel.entities import JobPosting
    from munich_intel.extractor import _save_jobs

    with patch("munich_intel.extractor.ENTITIES_DIR", tmp_path):
        existing_path = tmp_path / "reverion_jobs.json"
        existing_path.write_text(
            json.dumps(
                [
                    {
                        "company_slug": "reverion",
                        "title": "Senior ML Engineer",
                        "url": "https://reverion.com/jobs/123",
                        "scraped_at": "2026-01-01",
                    }
                ]
            )
        )

        new_postings = [
            JobPosting(
                company_slug="reverion",
                title="Senior ML Engineer (retitled)",
                url="https://reverion.com/jobs/123",  # same URL as the existing row
                scraped_at="2026-03-14",
            ),
            JobPosting(
                company_slug="reverion",
                title="Backend Engineer",
                url="https://reverion.com/jobs/456",  # new URL
                scraped_at="2026-03-14",
            ),
        ]
        _save_jobs(new_postings, "reverion")

        saved = json.loads(existing_path.read_text())

    assert len(saved) == 2
    by_url = {row["url"]: row for row in saved}
    # The pre-existing row's first-seen date and title win — a re-scrape shouldn't
    # erase when a listing was actually first observed.
    assert by_url["https://reverion.com/jobs/123"]["scraped_at"] == "2026-01-01"
    assert by_url["https://reverion.com/jobs/123"]["title"] == "Senior ML Engineer"
    assert by_url["https://reverion.com/jobs/456"]["scraped_at"] == "2026-03-14"


def test_raw_postings_parses_json_object_from_groq():
    from munich_intel.extractor import _raw_postings

    mock_message = MagicMock(content=json.dumps({"postings": [{"title": "ML Engineer"}]}))
    mock_choice = MagicMock(message=mock_message)
    mock_completion = MagicMock(choices=[mock_choice])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with (
        patch("munich_intel.extractor.settings.llm_provider", "groq"),
        patch("munich_intel.extractor._groq_client", return_value=mock_client),
    ):
        result = _raw_postings("some careers page text")

    assert result == [{"title": "ML Engineer"}]


def test_raw_postings_returns_empty_on_invalid_json():
    from munich_intel.extractor import _raw_postings

    mock_message = MagicMock(content="not json")
    mock_choice = MagicMock(message=mock_message)
    mock_completion = MagicMock(choices=[mock_choice])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with (
        patch("munich_intel.extractor.settings.llm_provider", "groq"),
        patch("munich_intel.extractor._groq_client", return_value=mock_client),
    ):
        result = _raw_postings("some careers page text")

    assert result == []


def _news_page(**overrides) -> ScrapedPage:
    fields = {
        "company_name": "Reverion",
        "company_slug": "reverion",
        "category": "energy",
        "url": "https://news.google.com/rss/search?q=%22Reverion%22",
        "page_text": (
            "Title: Reverion raises Series B\n"
            "Link: https://news.example.com/reverion-series-b\n"
            "Published: Mon, 01 Jan 2026 00:00:00 GMT\n"
            "Source: TechCrunch"
            "\n---\n"
            "Title: Reverion opens Munich office\n"
            "Link: https://news.example.com/reverion-office\n"
            "Published: Tue, 02 Jan 2026 00:00:00 GMT\n"
            "Source: Example News"
        ),
        "scraped_at": "2026-01-01T00:00:00Z",
        "word_count": 20,
        "source_type": "news",
    }
    fields.update(overrides)
    return ScrapedPage(**fields)


def test_extract_news_mentions_returns_empty_for_non_news_page():
    page = _news_page(source_type="site")
    assert extract_news_mentions(page) == []


def test_extract_news_mentions_returns_empty_for_blank_page_text():
    page = _news_page(page_text="")
    assert extract_news_mentions(page) == []


def test_extract_news_mentions_parses_blocks_without_calling_llm():
    # No _chat_json/_groq_client patch — proves this path never touches the LLM.
    with patch("munich_intel.extractor._save"):
        mentions = extract_news_mentions(_news_page())

    assert len(mentions) == 2
    assert mentions[0].company_slug == "reverion"
    assert mentions[0].title == "Reverion raises Series B"
    assert str(mentions[0].url) == "https://news.example.com/reverion-series-b"
    assert mentions[0].source == "TechCrunch"
    assert mentions[0].published_on.isoformat() == "2026-01-01"
    assert mentions[1].title == "Reverion opens Munich office"


def test_extract_news_mentions_skips_block_with_unparseable_date():
    page = _news_page(page_text="Title: No date here\nLink: https://news.example.com/x\nSource: Some Source")
    with patch("munich_intel.extractor._save"):
        mentions = extract_news_mentions(page)

    assert mentions == []


def test_extract_news_mentions_saves_results():
    with patch("munich_intel.extractor._save") as mock_save:
        mentions = extract_news_mentions(_news_page())

    mock_save.assert_called_once_with(mentions, "reverion", "news")


def test_extract_funding_rounds_returns_empty_for_non_news_page():
    page = _news_page(source_type="site")
    assert extract_funding_rounds(page) == []


def test_extract_funding_rounds_returns_empty_for_blank_page_text():
    page = _news_page(page_text="")
    assert extract_funding_rounds(page) == []


def test_extract_funding_rounds_builds_entities_from_llm_output():
    raw = [
        {
            "round_type": "series-b",
            "announced_on": "2026-01-01",
            "amount_eur": 20000000,
            "investor_names": ["Acme Ventures"],
            "source_url": "https://news.example.com/reverion-series-b",
        }
    ]
    with (
        patch("munich_intel.extractor._raw_funding_rounds", return_value=raw),
        patch("munich_intel.extractor._save"),
    ):
        rounds = extract_funding_rounds(_news_page())

    assert len(rounds) == 1
    assert rounds[0].company_slug == "reverion"
    assert rounds[0].round_type == RoundType.SERIES_B
    assert rounds[0].amount_eur == 20000000
    assert rounds[0].investor_names == ["Acme Ventures"]


def test_extract_funding_rounds_skips_malformed_entries():
    raw = [{"round_type": "series-b"}]  # missing required announced_on/source_url
    with (
        patch("munich_intel.extractor._raw_funding_rounds", return_value=raw),
        patch("munich_intel.extractor._save"),
    ):
        rounds = extract_funding_rounds(_news_page())

    assert rounds == []


def test_extract_funding_rounds_saves_results():
    raw = [
        {
            "round_type": "seed",
            "announced_on": "2026-01-01",
            "source_url": "https://news.example.com/reverion-seed",
        }
    ]
    with (
        patch("munich_intel.extractor._raw_funding_rounds", return_value=raw),
        patch("munich_intel.extractor._save") as mock_save,
    ):
        rounds = extract_funding_rounds(_news_page())

    mock_save.assert_called_once_with(rounds, "reverion", "funding")


def test_raw_funding_rounds_parses_json_object_from_groq():
    from munich_intel.extractor import _raw_funding_rounds

    mock_message = MagicMock(content=json.dumps({"funding_rounds": [{"round_type": "seed"}]}))
    mock_choice = MagicMock(message=mock_message)
    mock_completion = MagicMock(choices=[mock_choice])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with (
        patch("munich_intel.extractor.settings.llm_provider", "groq"),
        patch("munich_intel.extractor._groq_client", return_value=mock_client),
    ):
        result = _raw_funding_rounds("some news page text")

    assert result == [{"round_type": "seed"}]
