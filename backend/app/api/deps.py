"""
FastAPI dependency providers.

Routes depend on these functions (via `Depends(...)`) rather than
importing services directly, which keeps route handlers decoupled from
how services are constructed and makes them easy to override in tests.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.recommender.data_loader import load_artifacts
from app.services.contact_service import ContactService
from app.services.game_service import GameService
from app.services.movie_service import MovieService
from app.services.recommendation_service import RecommendationService


def get_movie_service() -> MovieService:
    return MovieService(load_artifacts())


def get_game_service() -> GameService:
    return GameService(load_artifacts())


def get_recommendation_service() -> RecommendationService:
    return RecommendationService(load_artifacts())


@lru_cache
def get_contact_service() -> ContactService:
    return ContactService(get_settings())
