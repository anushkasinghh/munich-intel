"""Hand-written examples for chunker.chunk_rss (Phase 3c)."""

from munich_intel.chunker import chunk_rss

BLOCK = (
    "Title: Proxima Fusion raises 411 million euros\n"
    "Link: https://news.google.com/rss/articles/CBMidkFVX3lxTFBXYUVFaHY1SC03UHNscHlHMElKY2s?oc=5\n"
    "Published: Wed, 08 Jul 2026 07:00:00 GMT\n"
    "Source: Max-Planck-Gesellschaft"
)
OTHER = (
    "Title: RWE invests 25 million euros in Proxima Fusion\n"
    "Link: https://news.google.com/rss/articles/CBMivgFBVV95cUxNSFBta0s5cXVpSUVuQzFF?oc=5\n"
    "Published: Tue, 07 Jul 2026 07:07:03 GMT\n"
    "Source: RWE"
)
FEED = f"{BLOCK}\n---\n{OTHER}"


def test_one_chunk_per_article():
    # The whole point: a 512-word chunk averaged ~19 unrelated headlines into one
    # vector, so a question about one funding round matched it only weakly.
    assert len(chunk_rss(FEED)) == 2


def test_the_base64_redirect_is_dropped():
    # 64% of the corpus's RSS text by character count, and meaningless to both the
    # embedder and the generator — nobody reads a redirect blob.
    chunks = chunk_rss(FEED)
    assert not any("news.google.com/rss/articles" in c for c in chunks)
    assert not any(c.startswith("Link:") or "\nLink:" in c for c in chunks)


def test_the_parts_worth_embedding_survive():
    first = chunk_rss(FEED)[0]
    assert "Proxima Fusion raises 411 million euros" in first
    assert "Published: Wed, 08 Jul 2026" in first
    assert "Source: Max-Planck-Gesellschaft" in first


def test_stripping_links_removes_most_of_the_text():
    # The fixture's links are abbreviated (~110 chars) to stay readable; real Google
    # News redirects are ~290, where the saving measured across all 902 blocks in the
    # corpus is 64% and a block goes from ~111 tokens to ~40. So this asserts the
    # direction and rough scale, not the production ratio.
    kept = sum(map(len, chunk_rss(FEED, strip_links=False)))
    stripped = sum(map(len, chunk_rss(FEED)))
    assert stripped < kept * 0.7


def test_links_can_be_kept_for_callers_that_need_them():
    # data/raw keeps them because extract_news_mentions reads them for
    # NewsMention.url. Only the INDEXED text drops them.
    assert all("Link:" in c for c in chunk_rss(FEED, strip_links=False))


def test_an_empty_feed_is_no_chunks():
    # A company with zero coverage returns a valid but empty feed, which is a
    # legitimate zero rather than an error (see scraper.scrape_news).
    assert chunk_rss("") == []


def test_blank_blocks_are_dropped_not_embedded():
    assert len(chunk_rss(f"{BLOCK}\n---\n   \n---\n{OTHER}")) == 2


def test_a_block_that_is_only_a_link_disappears_entirely():
    # Would otherwise embed an empty string and waste a slot in the top-k.
    only_link = "Link: https://news.google.com/rss/articles/AAAA?oc=5"
    assert chunk_rss(f"{BLOCK}\n---\n{only_link}") == chunk_rss(BLOCK)
