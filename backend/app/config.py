from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    database_url: str = "mysql+pymysql://username:password@localhost:3306/supplychain"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    dataset_path: str = "backend/data/supply_chain_disruption_recovery.csv"
    chroma_path: str = "backend/data/chroma"
    chroma_collection: str = "supply_chain_disruptions"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    bm25_index_path: str = "backend/rag/bm25_index.joblib"
    model_dir: str = "backend/ml/models"
    model_metrics_path: str = "evaluation/model_metrics.json"
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_endpoint: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    def path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


settings = Settings()
