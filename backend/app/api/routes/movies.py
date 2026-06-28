"""Movie recommendation routes. HTTP concerns only — no business logic."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_movie_service
from app.schemas.movie import MovieRecommendationRequest, MovieRecommendationResponse
from app.services.movie_service import MovieService

router = APIRouter(prefix="/movies", tags=["movies"])


@router.post("/recommendations", response_model=MovieRecommendationResponse)
def recommend_movies(
    payload: MovieRecommendationRequest,
    service: MovieService = Depends(get_movie_service),
) -> MovieRecommendationResponse:
    try:
        return service.get_recommendations(payload.title, top_n=payload.top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
