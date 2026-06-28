"""
Unified recommendation service.

Provides a single entry point that dispatches to either the movie or
game service based on a `media_type` discriminator. This backs the
combined `/api/v1/recommendations` endpoint, while `movies.py` and
`games.py` keep their own dedicated endpoints for callers that already
know which domain they want.
"""
from __future__ import annotations

from enum import Enum

from app.recommender.data_loader import RecommenderArtifacts
from app.services.game_service import GameService
from app.services.movie_service import MovieService


class MediaType(str, Enum):
    MOVIE = "movie"
    GAME = "game"


class RecommendationService:
    def __init__(self, artifacts: RecommenderArtifacts) -> None:
        self._movie_service = MovieService(artifacts)
        self._game_service = GameService(artifacts)

    def get_recommendations(self, media_type: MediaType, title: str, top_n: int = 10):
        if media_type == MediaType.MOVIE:
            return self._movie_service.get_recommendations(title, top_n=top_n)
        if media_type == MediaType.GAME:
            return self._game_service.get_recommendations(title, top_n=top_n)
        raise ValueError(f"Unsupported media_type: {media_type}")
