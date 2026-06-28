"""
Game recommendation engine.

This is `recommend_games()` from the original `Deployment/app.py`,
preserved field-for-field and behavior-for-behavior, but restructured as a
class that receives its dataframe and similarity matrix as dependencies
instead of closing over module-level globals.
"""
from __future__ import annotations

import pandas as pd

from app.core.constants import GAME_ALIASES, REQUIRED_GAME_FIELDS
from app.core.logger import get_logger
from app.recommender.preprocessing import NoMatchFoundError, resolve_best_title_match
from app.recommender.similarity import top_n_similar_indices

logger = get_logger(__name__)


class GameRecommendationEngine:
    """
    Wraps the precomputed game dataframe + cosine similarity matrix and
    exposes the same recommendation behavior as the original
    `recommend_games()` function.
    """

    def __init__(self, games: pd.DataFrame, similarity_matrix) -> None:
        self.games = games
        self.similarity_matrix = similarity_matrix

    def recommend(self, user_input: str, top_n: int = 10) -> list[dict]:
        """
        Return up to `top_n` games similar to `user_input`.

        Raises:
            ValueError: empty input, or no fuzzy match found, or the
                matched row's index is out of range.
        """
        try:
            cleaned_input, best_match = resolve_best_title_match(
                user_input, self.games["title_clean"], GAME_ALIASES
            )
            logger.info("Best match for game input '%s' is: %s", user_input, best_match)

            idx = self.games[self.games["title_clean"] == best_match].index[0]

            if idx < 0 or idx >= len(self.games):
                raise IndexError(f"Index {idx} is out of range.")

            similar = top_n_similar_indices(self.similarity_matrix[idx], top_n)

            results: list[dict] = []
            for i, _score in similar:
                game_data = self.games.loc[i]

                # Safely split store names and domains into lists, or use empty lists
                store_names = game_data["store_name"].split(", ")
                store_domains = game_data["store_domain"].split(", ")

                store_display = ", ".join(
                    f"{name} : https://{domain}" for name, domain in zip(store_names, store_domains)
                )

                if any(field not in game_data for field in REQUIRED_GAME_FIELDS):
                    continue

                results.append(
                    {
                        "Title": self.games.loc[i, "title"],
                        "Description": self.games.loc[i, "description_clean"],
                        "Genre": self.games.loc[i, "genres"],
                        "Release Date": self.games.loc[i, "release_date"],
                        "Rating": self.games.loc[i, "rating"],
                        "Platforms": self.games.loc[i, "platforms"],
                        "Stores": store_display,
                        "Tags": self.games.loc[i, "tags"],
                        "Developer": self.games.loc[i, "developers"],
                        "Publisher": self.games.loc[i, "publishers"],
                        "ESRB_Rating": self.games.loc[i, "esrb_rating"],
                        "Poster": self.games.loc[i, "background_image_url"],
                        "Website": self.games.loc[i, "website"],
                        "Screenshots": self.games.loc[i, "screenshots"],
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
