"""
Loads the model artifacts (dataframes + TF-IDF vectorizer + sparse TF-IDF matrix).

What changed from the original
--------------------------------
Old: loaded a precomputed cosine-similarity matrix (movies_matrix / games_matrix)
     — a full N×N float array that was ~1.5 GB combined on disk and in memory.

New: loads three smaller artifacts per domain:
     - dataset (.pkl)                  → same as before
     - fitted TfidfVectorizer (.pkl)   → NEW (tiny; ~few MB)
     - sparse TF-IDF matrix (.npz)     → NEW (sparse; typical size ~50–100 MB)

The cosine similarity for a single item is then computed on-demand per
request via `cosine_similarity(tfidf_matrix[idx], tfidf_matrix)`.  This
shifts the work from startup (load a huge dense matrix) to request time
(one dot-product row operation in ~milliseconds), while keeping TF-IDF
computation entirely offline.

Everything else is unchanged: the download flow, the lru_cache lifetime,
and the artifact container interface the engines see.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import gdown
import pandas as pd
import scipy.sparse

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


def _load_sparse(path: Path) -> scipy.sparse.csr_matrix:
    """Load a scipy sparse matrix saved with save_npz and ensure CSR format
    (required by cosine_similarity for efficient row slicing)."""
    matrix = scipy.sparse.load_npz(str(path))
    if not isinstance(matrix, scipy.sparse.csr_matrix):
        matrix = matrix.tocsr()
    return matrix


class RecommenderArtifacts:
    """
    Container for the loaded artifacts needed to serve recommendations.

    Attributes
    ----------
    movies : pd.DataFrame
        Preprocessed movie dataset.
    movies_tfidf : scipy.sparse.csr_matrix
        Sparse TF-IDF feature matrix for movies (shape: n_movies × vocab).
    games : pd.DataFrame
        Preprocessed game dataset.
    games_tfidf : scipy.sparse.csr_matrix
        Sparse TF-IDF feature matrix for games (shape: n_games × vocab).
    """

    def __init__(
        self,
        movies: pd.DataFrame,
        movies_tfidf: scipy.sparse.csr_matrix,
        games: pd.DataFrame,
        games_tfidf: scipy.sparse.csr_matrix,
    ) -> None:
        self.movies = movies
        self.movies_tfidf = movies_tfidf
        self.games = games
        self.games_tfidf = games_tfidf


def _download_all(settings: Settings) -> dict[str, Path]:
    download_dir = settings.model_dir

    targets = {
        "movies":            (settings.MOVIES_DATA_FILE_ID,            settings.MOVIES_DATA_FILENAME),
        "movies_tfidf":      (settings.MOVIES_TFIDF_FILE_ID,           settings.MOVIES_TFIDF_FILENAME),
        "games":             (settings.GAMES_DATA_FILE_ID,             settings.GAMES_DATA_FILENAME),
        "games_tfidf":       (settings.GAMES_TFIDF_FILE_ID,            settings.GAMES_TFIDF_FILENAME),
    }

    paths: dict[str, Path] = {}
    for key, (file_id, filename) in targets.items():
        paths[key] = _download_if_missing(file_id, download_dir / filename)
    return paths


@lru_cache
def load_artifacts() -> RecommenderArtifacts:
    """
    Download (if needed) and load all recommender artifacts.

    Cached for the lifetime of the process — called once at FastAPI
    startup and reused across all requests.
    """
    settings = get_settings()
    paths = _download_all(settings)

    logger.info("Loading recommender artifacts into memory...")
    movies      = _load_pickle(paths["movies"])
    movies_tfidf = _load_sparse(paths["movies_tfidf"])
    games       = _load_pickle(paths["games"])
    games_tfidf  = _load_sparse(paths["games_tfidf"])

    logger.info(
        "Artifacts loaded: %d movies (tfidf %s), %d games (tfidf %s)",
        len(movies), movies_tfidf.shape,
        len(games),  games_tfidf.shape,
    )

    return RecommenderArtifacts(movies, movies_tfidf, games, games_tfidf)
