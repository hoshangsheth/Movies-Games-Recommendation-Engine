"""Health check route — used for uptime monitoring and container orchestration probes."""
from __future__ import annotations

from fastapi import APIRouter

from app.recommender.data_loader import load_artifacts
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    artifacts = load_artifacts()
    return HealthResponse(
        status="ok",
        movies_loaded=len(artifacts.movies),
        games_loaded=len(artifacts.games),
    )
