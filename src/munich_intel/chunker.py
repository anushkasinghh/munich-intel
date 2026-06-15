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
