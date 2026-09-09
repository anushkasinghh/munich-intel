from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    # Set qdrant_url to use Qdrant Cloud instead of a local container.
    # When set, qdrant_host and qdrant_port are ignored.
    qdrant_url: str | None = None
    qdrant_api_key: str = ""
    collection_name: str = "munich_intel"
    embedding_model: str = "BAAI/bge-m3"
    # Pin to a HuggingFace commit hash so the model never silently changes.
    # Get the hash from: https://huggingface.co/BAAI/bge-m3/commits/main
    # Leave empty to always use the latest commit (acceptable during dev).
    embedding_model_revision: str | None = None
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    llm_provider: str = "groq"  # "groq" | "ollama"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # Words, and applied to site/careers pages only — news feeds are chunked per
    # article by chunker.chunk_rss, which ignores these. Reduced 512/50 -> 200/25 in
    # Phase 3c so more distinct pages fit inside the same token budget.
    chunk_size: int = 200
    chunk_overlap: int = 25
    # Estimate token cost by CHARACTERS, never by word count. Google News wraps each
    # article in a ~290-character base64 redirect URL, which is one "word" but ~75
    # tokens, and those links were 64% of every news chunk by character count. A
    # 512-word news chunk was ~1802 tokens, not the ~525 its word count implied —
    # which is exactly why k=5 once cost ~12800 tokens and 413'd.
    #
    # 3c fixed the cause rather than the symptom: news is now chunked per article
    # with the redirect stripped (~39 tokens a chunk), and site/careers chunks are
    # 200 words (~325 tokens). Raised 2 -> 5 in 3b on recall@2 0.35 vs recall@5 0.62.
    retrieval_top_k: int = 5
    # Secret token required in X-Ingest-Token header to call POST /ingest.
    # Set a random string here and in HF Space secrets. Never leave empty in production.
    ingest_secret: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
