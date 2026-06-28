"""
Movie recommendation engine.

What changed from the original
--------------------------------
Old: stored `self.similarity_matrix` (dense N×N precomputed cosine matrix)
     and passed `self.similarity_matrix[idx]` (one dense row) to
     `top_n_similar_indices`.

New: stores `self.tfidf_matrix` (sparse TF-IDF, loaded once at startup)
     and passes the full matrix + idx to `top_n_similar_indices`, which
     computes cosine similarity on demand for that single row.

Everything else — fuzzy matching, alias resolution, result field mapping,
exception handling, and API response shape — is completely unchanged.
"""
from __future__ import annotations

import pandas as pd
import scipy.sparse

from app.core.constants import MOVIE_ALIASES, REQUIRED_MOVIE_FIELDS
from app.core.logger import get_logger
from app.recommender.preprocessing import NoMatchFoundError, resolve_best_title_match
from app.recommender.similarity import top_n_similar_indices

logger = get_logger(__name__)


class MovieRecommendationEngine:
    """
    Wraps the movie dataframe and sparse TF-IDF matrix; computes cosine
    similarity on demand per request.
    """

    def __init__(self, movies: pd.DataFrame, tfidf_matrix: scipy.sparse.csr_matrix) -> None:
        self.movies = movies
        self.tfidf_matrix = tfidf_matrix

    def recommend(self, user_input: str, top_n: int = 10) -> list[dict]:
        """
        Return up to `top_n` movies similar to `user_input`.

        Raises:
            ValueError: empty input, no fuzzy match found, or index out of
                range — same exception semantics as the original.
        """
        try:
            cleaned_input, best_match = resolve_best_title_match(
                user_input, self.movies["title_clean"], MOVIE_ALIASES
            )
            logger.info("Best match for movie input '%s' is: %s", user_input, best_match)

            idx = self.movies[self.movies["title_clean"] == best_match].index[0]

            if idx < 0 or idx >= len(self.movies):
                raise IndexError(f"Index {idx} is out of range.")

            # On-demand cosine similarity — replaces precomputed matrix row lookup
            similar = top_n_similar_indices(self.tfidf_matrix, idx, top_n)

            results: list[dict] = []
            for i, _score in similar:
                movie_data = self.movies.loc[i]

                if any(field not in movie_data for field in REQUIRED_MOVIE_FIELDS):
                    continue

                video_key = movie_data.get("video_key")
                trailer_url = (
                    f"https://www.youtube.com/watch?v={video_key}"
                    if pd.notna(video_key)
                    else None
                )

                results.append(
                    {
                        "Title": self.movies.loc[i, "title"],
                        "Top Cast": self.movies.loc[i, "top_cast"],
                        "Cast Picture": self.movies.loc[i, "cast_profile_path"],
                        "Description": self.movies.loc[i, "description"],
                        "Genre": self.movies.loc[i, "genres"],
                        "Language": self.movies.loc[i, "languages"],
                        "Release Date": self.movies.loc[i, "release_date"],
                        "Rating": self.movies.loc[i, "rating"],
                        "Poster": self.movies.loc[i, "poster_path"],
                        "Stream": self.movies.loc[i, "watch_link"],
                        "Trailer": trailer_url,
                    }
                )

            return results

        except NoMatchFoundError as e:
            raise ValueError(str(e)) from e
        except ValueError:
            raise
        except IndexError as ie:
            raise ValueError(f"Index error: {ie}") from ie
        except Exception as e:  # noqa: BLE001 - preserve original catch-all behavior
            raise ValueError(f"An unexpected error occurred: {e}") from e
