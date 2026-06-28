"""
Application configuration.

What changed from the original
--------------------------------
Old: MOVIES_SIMILARITY_FILE_ID / MOVIES_SIMILARITY_FILENAME pointed to the
     precomputed cosine-similarity matrix (cosine_sim_movies.pkl).
     Likewise for GAMES_SIMILARITY_FILE_ID / GAMES_SIMILARITY_FILENAME.

New: Those four settings are removed and replaced with:
     - MOVIES_TFIDF_FILE_ID / MOVIES_TFIDF_FILENAME  → sparse TF-IDF matrix (.npz)
     - GAMES_TFIDF_FILE_ID  / GAMES_TFIDF_FILENAME   → sparse TF-IDF matrix (.npz)

The vectorizers are stored inside the dataset pickle (movies_recommended.pkl
/ games_recommended.pkl) or alongside them — update MOVIES_TFIDF_FILE_ID
and GAMES_TFIDF_FILE_ID to point to your Google Drive uploads of the new
.npz artifacts produced by scripts/build_tfidf_artifacts.py.

Everything else (CORS, contact form, general settings) is unchanged.
"""
from __future__ import annotations

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
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    FRONTEND_ORIGIN: str | None = None

    # --- Model artifact storage ---
    # Artifacts are generated offline (see scripts/build_tfidf_artifacts.py),
    # hosted on Google Drive, and downloaded once at startup via gdown.
    MODEL_DOWNLOAD_DIR: str = "/tmp/recommender-artifacts"

    # Dataset pickles (unchanged from original)
    MOVIES_DATA_FILE_ID: str = "1Pk9VAuaX8fVpHM7HHZqgTR6G8fONCXtp"
    GAMES_DATA_FILE_ID: str = "1w_KvquJhyogtwPUkXTG1E7FFkECeLkj0"
    MOVIES_DATA_FILENAME: str = "movies_recommended.pkl"
    GAMES_DATA_FILENAME: str = "games_recommended.pkl"

    # Sparse TF-IDF matrices — REPLACE these placeholders with the real
    # Google Drive file IDs after uploading the .npz files produced by
    # scripts/build_tfidf_artifacts.py
    MOVIES_TFIDF_FILE_ID: str = "REPLACE_WITH_MOVIES_TFIDF_NPZ_FILE_ID"
    GAMES_TFIDF_FILE_ID: str = "REPLACE_WITH_GAMES_TFIDF_NPZ_FILE_ID"
    MOVIES_TFIDF_FILENAME: str = "movies_tfidf.npz"
    GAMES_TFIDF_FILENAME: str = "games_tfidf.npz"

    # --- Recommendation defaults ---
    DEFAULT_TOP_N: int = 10
    FUZZY_MATCH_MIN_SCORE: float = 0.0  # kept permissive, matches original behavior

    # --- Contact form (Google Sheets) ---
    GOOGLE_SERVICE_ACCOUNT_JSON: str | None = None
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
