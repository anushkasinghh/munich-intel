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
    groq_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
