"""Game recommendation routes. HTTP concerns only — no business logic."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_game_service
from app.schemas.game import GameRecommendationRequest, GameRecommendationResponse
from app.services.game_service import GameService

router = APIRouter(prefix="/games", tags=["games"])


@router.post("/recommendations", response_model=GameRecommendationResponse)
def recommend_games(
    payload: GameRecommendationRequest,
    service: GameService = Depends(get_game_service),
) -> GameRecommendationResponse:
    try:
        return service.get_recommendations(payload.title, top_n=payload.top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
