"""Aggregates all route modules into a single router mounted by `main.py`."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import games, health, movies, recommendations

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(movies.router)
api_router.include_router(games.router)
api_router.include_router(recommendations.router)
api_router.include_router(recommendations.contact_router)
