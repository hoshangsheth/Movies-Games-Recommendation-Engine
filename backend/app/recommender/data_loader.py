"""
Loads the precomputed model artifacts (dataframes + similarity matrices).

This replicates the original `Deployment/app.py` bootstrap: the TF-IDF +
cosine-similarity artifacts are generated offline (see
`backend/notebooks/`) and are too large to commit to the repo, so they're
hosted on Google Drive and fetched once via `gdown` if not already
present on disk. Results are cached in-process so repeated calls (e.g.
across requests) don't re-download or re-deserialize.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import gdown
import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _download_if_missing(file_id: str, output_path: Path) -> Path:
    if not output_path.exists():
        logger.info("Downloading %s from Google Drive...", output_path.name)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False, use_cookies=False)
    return output_path


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_numpy(path: Path):
    matrix = np.load(path, allow_pickle=True)
    # Similarity scores only need float32 precision; this halves the
    # matrix's memory footprint once loaded. Note: this does not reduce
    # *peak* memory during the load itself — if the on-disk file is
    # already too large for available RAM, shrink it ahead of time with
    # scripts/shrink_artifacts.py instead.
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    return matrix


class RecommenderArtifacts:
    """Container for the four loaded artifacts needed to serve recommendations."""

    def __init__(
        self,
        movies: pd.DataFrame,
        movies_matrix,
        games: pd.DataFrame,
        games_matrix,
    ) -> None:
        self.movies = movies
        self.movies_matrix = movies_matrix
        self.games = games
        self.games_matrix = games_matrix


def _download_all(settings: Settings) -> dict[str, Path]:
    download_dir = settings.model_dir

    targets = {
        "movies": (settings.MOVIES_DATA_FILE_ID, settings.MOVIES_DATA_FILENAME),
        "movies_matrix": (settings.MOVIES_SIMILARITY_FILE_ID, settings.MOVIES_SIMILARITY_FILENAME),
        "games": (settings.GAMES_DATA_FILE_ID, settings.GAMES_DATA_FILENAME),
        "games_matrix": (settings.GAMES_SIMILARITY_FILE_ID, settings.GAMES_SIMILARITY_FILENAME),
    }

    paths: dict[str, Path] = {}
    for key, (file_id, filename) in targets.items():
        paths[key] = _download_if_missing(file_id, download_dir / filename)
    return paths


@lru_cache
def load_artifacts() -> RecommenderArtifacts:
    """
    Download (if needed) and load all four recommender artifacts.

    Cached for the lifetime of the process — call this once at FastAPI
    startup and reuse the result, exactly as the original app's
    `@st.cache_resource` decorators did for the lifetime of the Streamlit
    session.
    """
    settings = get_settings()
    paths = _download_all(settings)

    logger.info("Loading recommender artifacts into memory...")
    movies = _load_pickle(paths["movies"])
    movies_matrix = _load_pickle(paths["movies_matrix"])
    games = _load_pickle(paths["games"])
    games_matrix = _load_numpy(paths["games_matrix"])
    logger.info("Recommender artifacts loaded: %d movies, %d games", len(movies), len(games))

    return RecommenderArtifacts(movies, movies_matrix, games, games_matrix)
