import json
from unittest.mock import MagicMock, patch

from munich_intel.extractor import extract_job_postings
from munich_intel.scraper import ScrapedPage


def _page(**overrides) -> ScrapedPage:
    fields = {
        "company_name": "Reverion",
        "company_slug": "reverion",
        "category": "energy",
        "url": "https://reverion.com/careers",
        "page_text": "Senior ML Engineer [https://reverion.com/jobs/123]",
        "scraped_at": "2026-01-01T00:00:00Z",
        "word_count": 4,
        "source_type": "careers",
    }
    fields.update(overrides)
    return ScrapedPage(**fields)


def test_extract_job_postings_returns_empty_for_non_careers_page():
    page = _page(source_type="site")
    assert extract_job_postings(page) == []


def test_extract_job_postings_returns_empty_for_blank_page_text():
    page = _page(page_text="   ")
    assert extract_job_postings(page) == []


def test_extract_job_postings_builds_entities_from_llm_output():
    raw = [{"title": "Senior ML Engineer", "url": "https://reverion.com/jobs/123", "posted_on": "2026-01-01", "location": "Munich"}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save"),
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
        patch("munich_intel.extractor._save"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert str(postings[0].url) == "https://reverion.com/careers"


def test_extract_job_postings_skips_malformed_entries():
    raw = [{"url": "https://reverion.com/jobs/1"}, {"title": "Valid Posting"}]  # first is missing required title
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save"),
    ):
        postings = extract_job_postings(_page())

    assert len(postings) == 1
    assert postings[0].title == "Valid Posting"


def test_extract_job_postings_saves_results():
    raw = [{"title": "Senior ML Engineer"}]
    with (
        patch("munich_intel.extractor._raw_postings", return_value=raw),
        patch("munich_intel.extractor._save") as mock_save,
    ):
        postings = extract_job_postings(_page())

    mock_save.assert_called_once_with(postings, "reverion")


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
