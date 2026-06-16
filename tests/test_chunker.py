from munich_intel.chunker import chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_no_chunk_exceeds_chunk_size():
    text = " ".join(f"word{i}" for i in range(1000))
    for chunk in chunk_text(text, chunk_size=512):
        assert len(chunk.split()) <= 512


def test_no_empty_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    for chunk in chunk_text(text):
        assert chunk.strip() != ""


def test_all_input_words_appear_in_output():
    words = [f"word{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words))
    all_output_words = set(" ".join(chunks).split())
    assert set(words) == all_output_words
