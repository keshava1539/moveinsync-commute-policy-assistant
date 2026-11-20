from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Project Information
    PROJECT_NAME: str = "MoveInSync Commute Policy Assistant"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Security & JWT Auth
    JWT_SECRET_KEY: str = "super-secret-key-moveinsync-dev-interview-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # ChromaDB & Storage
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "commute_policies"
    POLICY_STORE_FILE: str = "./policy_versions.json"
    POLICIES_DIR: str = "./data/policies"

    # Retrieval & RAG Constraints
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K_CHUNKS: int = 3
    MAX_LLM_CALLS_PER_QUERY: int = 2
    MAX_LATENCY_SECONDS: float = 3.0
    SIMILARITY_DISTANCE_THRESHOLD: float = 0.85  # Fallback to desk if distance > threshold

    # Bonus: Toggleable Hybrid Search (Chroma Dense + BM25 + Reciprocal Rank Fusion)
    ENABLE_HYBRID_SEARCH: bool = False

    # LLM Settings ("mock", "gemini", "groq")
    LLM_PROVIDER: str = "mock"
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "gemini-1.5-flash"

    # Startup Lifespan
    AUTO_SEED_ON_STARTUP: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
