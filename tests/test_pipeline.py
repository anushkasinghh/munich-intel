from unittest.mock import MagicMock, patch

from munich_intel.pipeline import answer, answer_stream

FAKE_CHUNKS = [
    {"chunk_text": "Reverion builds biogas plants.", "company_name": "Reverion", "url": "https://reverion.com", "score": 0.9},
    {"chunk_text": "NavVis maps indoor spaces.", "company_name": "NavVis", "url": "https://navvis.com", "score": 0.8},
]


def test_answer_returns_fallback_when_no_chunks():
    with patch("munich_intel.pipeline.retrieve", return_value=[]):
        result = answer("any question", MagicMock(), MagicMock(), MagicMock())
    assert result["answer"] == "No relevant information found."
    assert result["sources"] == []


def test_answer_returns_answer_and_sources():
    with patch("munich_intel.pipeline.retrieve", return_value=FAKE_CHUNKS), \
         patch("munich_intel.pipeline.generate", return_value="Reverion builds biogas plants."):
        result = answer("What does Reverion do?", MagicMock(), MagicMock(), MagicMock())
    assert isinstance(result["answer"], str)
    assert len(result["sources"]) == 2


def test_answer_sources_are_correctly_formatted():
    # pipeline transforms company_name → company; this catches that rename breaking
    with patch("munich_intel.pipeline.retrieve", return_value=FAKE_CHUNKS), \
         patch("munich_intel.pipeline.generate", return_value="some answer"):
        result = answer("any question", MagicMock(), MagicMock(), MagicMock())
    for source in result["sources"]:
        assert "company" in source
        assert "url" in source
        assert "company_name" not in source
        assert "chunk_text" not in source


def test_answer_stream_returns_iterator_and_sources():
    def fake_stream(*_args, **_kwargs):
        yield "token1"
        yield "token2"

    with patch("munich_intel.pipeline.retrieve", return_value=FAKE_CHUNKS), \
         patch("munich_intel.pipeline.generate_stream", side_effect=fake_stream):
        tokens, sources = answer_stream("any question", MagicMock(), MagicMock(), MagicMock())

    assert list(tokens) == ["token1", "token2"]
    assert len(sources) == 2


def test_answer_stream_returns_fallback_when_no_chunks():
    with patch("munich_intel.pipeline.retrieve", return_value=[]):
        tokens, sources = answer_stream("any question", MagicMock(), MagicMock(), MagicMock())
    assert list(tokens) == ["No relevant information found."]
    assert sources == []
