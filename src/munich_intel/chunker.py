import re

# `scraper._clean_rss` emits one Title/Link/Published/Source block per article,
# joined by this separator. It is a stable contract: extract_news_mentions parses
# the same shape back out of data/raw/.
RSS_BLOCK_SEPARATOR = "\n---\n"

# Google News wraps every article in a redirect URL whose path is a ~290-character
# base64 blob. Measured across all 902 blocks in this corpus, that Link line is 64%
# of the text by character count.
_RSS_LINK_LINE = re.compile(r"^Link: \S+\n?", re.M)


def chunk_rss(text: str, strip_links: bool = True) -> list[str]:
    """One chunk per news article, instead of a 512-word dump of ~19 articles.

    Two separate wins, worth keeping distinct:

    1. **Granularity.** A 512-word chunk of an RSS feed is nineteen unrelated
       headlines averaged into one vector, so a question about one funding round
       matches it only weakly. One article per chunk is one topic per vector.

    2. **Token cost.** The base64 redirect URLs are 64% of the text and carry no
       meaning for either the embedder or the generator — nobody reads a redirect
       blob. Dropping them takes a block from ~111 tokens to ~40.

    That second point is why the old `config.py` note was right and a word-count
    estimate is not: a 512-*word* news chunk is ~1800 tokens, not the ~525 its word
    count implies, because one "word" can be a 290-character URL. It is what made
    k=5 cost ~12800 tokens.

    Only the INDEXED text changes. `data/raw/` keeps the links, because
    `extract_news_mentions` reads them to build `NewsMention.url`.
    """
    blocks = [b.strip() for b in text.split(RSS_BLOCK_SEPARATOR) if b.strip()]
    if not strip_links:
        return blocks
    return [stripped for b in blocks if (stripped := _RSS_LINK_LINE.sub("", b).strip())]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:]

        current_words.extend(para_words)

        # paragraph longer than chunk_size on its own
        while len(current_words) > chunk_size:
            chunks.append(" ".join(current_words[:chunk_size]))
            current_words = current_words[chunk_size - overlap:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
