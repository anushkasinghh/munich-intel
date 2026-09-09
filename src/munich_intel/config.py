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
    chunk_size: int = 512
    chunk_overlap: int = 50
    # chunk_size counts WORDS. Measured on the live index: chunks average 389 words
    # (~525 tokens) with a 512-word cap, so k=5 costs ~2600 tokens of context and
    # k=10 ~5300, against Groq's free-tier ~6000 TPM. Note TPM is per MINUTE and
    # cumulative across requests, which is what the earlier "k=5 asked for 12800
    # tokens" note was really measuring — not one oversized request.
    #
    # Raised 2 -> 5 in Phase 3b: the eval showed recall@2 0.35 vs recall@5 0.62, and
    # single-company recall reaching 1.00 at k=5. k=10 would buy another +0.13 but
    # doubles the token cost, so it waits for 3c to shrink chunks first.
    retrieval_top_k: int = 5
    # Secret token required in X-Ingest-Token header to call POST /ingest.
    # Set a random string here and in HF Space secrets. Never leave empty in production.
    ingest_secret: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
