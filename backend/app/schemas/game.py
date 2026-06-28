"""Pydantic schemas for the games API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GameRecommendationRequest(BaseModel):
    """Request body for POST /api/v1/games/recommendations."""

    title: str = Field(..., description="A game title, nickname, or alias to search for.")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")


class GameResult(BaseModel):
    """A single recommended game, matching the fields the original UI displayed."""

    title: str = Field(..., alias="Title")
    description: str = Field(..., alias="Description")
    genre: str = Field(..., alias="Genre")
    release_date: str = Field(..., alias="Release Date")
    rating: float = Field(..., alias="Rating")
    platforms: str = Field(..., alias="Platforms")
    stores: str = Field(..., alias="Stores")
    tags: str = Field(..., alias="Tags")
    developer: str = Field(..., alias="Developer")
    publisher: str = Field(..., alias="Publisher")
    esrb_rating: str = Field(..., alias="ESRB_Rating")
    poster: str = Field(..., alias="Poster")
    website: str | None = Field(None, alias="Website")
    screenshots: str | None = Field(None, alias="Screenshots")

    model_config = {"populate_by_name": True}


class GameRecommendationResponse(BaseModel):
    """Response body for POST /api/v1/games/recommendations."""

    query: str
    results: list[GameResult]
