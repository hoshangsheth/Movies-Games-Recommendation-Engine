"""
Movie service: the business-logic layer between the API routes and the
recommendation engine.

Keeps route handlers thin — they call `MovieService.get_recommendations()`
and translate the result into an HTTP response, with no business logic of
their own.
"""
from __future__ import annotations

from app.recommender.data_loader import RecommenderArtifacts
from app.recommender.movie_engine import MovieRecommendationEngine
from app.schemas.movie import MovieRecommendationResponse, MovieResult


class MovieService:
    def __init__(self, artifacts: RecommenderArtifacts) -> None:
        self._engine = MovieRecommendationEngine(artifacts.movies, artifacts.movies_matrix)

    def get_recommendations(self, title: str, top_n: int = 10) -> MovieRecommendationResponse:
        """
        Run the recommendation engine and shape the result into the API
        response schema. Raises ValueError on invalid/unmatched input,
        which the route layer maps to an HTTP 404/400.
        """
        raw_results = self._engine.recommend(title, top_n=top_n)
        results = [MovieResult.model_validate(item) for item in raw_results]
        return MovieRecommendationResponse(query=title, results=results)
