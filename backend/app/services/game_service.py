"""
Game service: the business-logic layer between the API routes and the
recommendation engine.
"""
from __future__ import annotations

from app.recommender.data_loader import RecommenderArtifacts
from app.recommender.game_engine import GameRecommendationEngine
from app.schemas.game import GameRecommendationResponse, GameResult


class GameService:
    def __init__(self, artifacts: RecommenderArtifacts) -> None:
        self._engine = GameRecommendationEngine(artifacts.games, artifacts.games_matrix)

    def get_recommendations(self, title: str, top_n: int = 10) -> GameRecommendationResponse:
        """
        Run the recommendation engine and shape the result into the API
        response schema. Raises ValueError on invalid/unmatched input,
        which the route layer maps to an HTTP 404/400.
        """
        raw_results = self._engine.recommend(title, top_n=top_n)
        results = [GameResult.model_validate(item) for item in raw_results]
        return GameRecommendationResponse(query=title, results=results)
