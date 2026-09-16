from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vector_store_path: str = "data/vector_store"
    collection_name: str = "ml_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2"
    retrieval_k: int = 3
    cors_origins: str = "http://localhost:8501"

    # If the closest retrieved chunk's distance is above this, treat the
    # question as out-of-scope for the document collection. Chroma's default
    # metric is L2 distance (lower = more similar); tune this by testing a
    # few clearly on-topic vs. clearly off-topic questions and checking the
    # printed distances (see the notebook's guardrails section).
    out_of_scope_threshold: float = 1.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
