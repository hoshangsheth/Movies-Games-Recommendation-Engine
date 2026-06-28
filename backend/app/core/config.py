"""
Application configuration.

Centralizes everything that used to be scattered as module-level constants
in the original Streamlit `app.py`: the Google Drive file IDs for the
precomputed model artifacts, the download directory, CORS settings, and
the Google Sheets contact-form configuration.

Values can be overridden via environment variables (see `.env.example` at
the repo root) so the same code runs locally, in Docker, and in any cloud
deployment without code changes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General ---
    APP_NAME: str = "Movies & Games Recommendation Engine API"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # --- CORS ---
    # The Vite dev server defaults to 5173. Add production frontend origin
    # via FRONTEND_ORIGIN env var when deploying.
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    FRONTEND_ORIGIN: str | None = None

    # --- Model artifact storage ---
    # The TF-IDF / cosine-similarity artifacts are generated offline (see
    # backend/notebooks/) and are too large to commit to git. They are
    # hosted on Google Drive and downloaded once at startup, exactly as the
    # original Streamlit app did with `gdown`.
    MODEL_DOWNLOAD_DIR: str = "/tmp/recommender-artifacts"

    MOVIES_DATA_FILE_ID: str = "1Pk9VAuaX8fVpHM7HHZqgTR6G8fONCXtp"
    MOVIES_SIMILARITY_FILE_ID: str = "1LA3XqdfJtyjvQEhXBg7INSuHeZAQhpk8"
    GAMES_DATA_FILE_ID: str = "1w_KvquJhyogtwPUkXTG1E7FFkECeLkj0"
    GAMES_SIMILARITY_FILE_ID: str = "1GoeHQQh-ngwVHtATbfWWPXPY5XBT-v61"

    MOVIES_DATA_FILENAME: str = "movies_recommended.pkl"
    MOVIES_SIMILARITY_FILENAME: str = "cosine_sim_movies.pkl"
    GAMES_DATA_FILENAME: str = "games_recommended.pkl"
    GAMES_SIMILARITY_FILENAME: str = "cosine_sim_games.npy"

    # --- Recommendation defaults ---
    DEFAULT_TOP_N: int = 10
    FUZZY_MATCH_MIN_SCORE: float = 0.0  # kept permissive, matches original behavior

    # --- Contact form (Google Sheets) ---
    GOOGLE_SERVICE_ACCOUNT_JSON: str | None = None  # path to credentials JSON, or raw JSON string
    CONTACT_SHEET_KEY: str = "1dlXnan4bMdcbdoXngU_15u4A0OVI_m4uUnRew3traXY"
    CONTACT_SHEET_WORKSHEET: str = "Sheet1"

    @property
    def model_dir(self) -> Path:
        path = Path(self.MODEL_DOWNLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def allowed_origins(self) -> list[str]:
        origins = list(self.CORS_ORIGINS)
        if self.FRONTEND_ORIGIN:
            origins.append(self.FRONTEND_ORIGIN)
        return origins


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, don't instantiate Settings() directly."""
    return Settings()
