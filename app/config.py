"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


def _load_dotenv(path: str = ".env") -> None:
    """Load .env when running without Docker. Real environment variables always win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split(" #")[0].strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _today() -> date:
    raw = os.getenv("APP_TODAY", "").strip()
    return date.fromisoformat(raw) if raw else date.today()


@dataclass
class Settings:
    corpus_dir: str = os.getenv("CORPUS_DIR", "./data/corpus")
    manifest_path: str = os.getenv("MANIFEST_PATH", "./data/manifest.json")
    # Legacy SQLite path kept so migration script can read the old DB.
    db_path: str = os.getenv("DB_PATH", "./data/veltrix.db")
    # PostgreSQL connection string (required in production).
    database_url: str = os.getenv("DATABASE_URL", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "12"))
    embeddings: str = os.getenv("EMBEDDINGS", "fastembed").lower()  # fastembed | none
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    corpus_token_budget: int = int(os.getenv("CORPUS_TOKEN_BUDGET", "2800"))  # hard limit is 3000
    log_file: str = os.getenv("LOG_FILE", "./logs/app.log")

    @property
    def today(self) -> date:
        return _today()


settings = Settings()

HARD_CORPUS_TOKEN_LIMIT = 3000
LEAVE_YEAR = 2026