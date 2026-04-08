from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG System API"
    api_prefix: str = "/api/v1"
    debug: bool = False

    supabase_db_url: str = Field(..., alias="SUPABASE_DB_URL")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL_NAME",
    )
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    top_k: int = Field(default=6, alias="TOP_K")
    vector_weight: float = Field(default=0.75, alias="VECTOR_WEIGHT")
    keyword_weight: float = Field(default=0.25, alias="KEYWORD_WEIGHT")
    conversation_window: int = Field(default=8, alias="CONVERSATION_WINDOW")
    max_context_characters: int = Field(default=12000, alias="MAX_CONTEXT_CHARACTERS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
