"""Pydantic schemas for the movies API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MovieRecommendationRequest(BaseModel):
    """Request body for POST /api/v1/movies/recommendations."""

    title: str = Field(..., description="A movie title, nickname, or alias to search for.")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")


class MovieResult(BaseModel):
    """A single recommended movie, matching the fields the original UI displayed."""

    title: str = Field(..., alias="Title")
    top_cast: str = Field(..., alias="Top Cast")
    cast_picture: str = Field(..., alias="Cast Picture")
    description: str = Field(..., alias="Description")
    genre: str = Field(..., alias="Genre")
    language: str = Field(..., alias="Language")
    release_date: str = Field(..., alias="Release Date")
    rating: float = Field(..., alias="Rating")
    poster: str = Field(..., alias="Poster")
    stream: str | None = Field(None, alias="Stream")
    trailer: str | None = Field(None, alias="Trailer")

    model_config = {"populate_by_name": True}


class MovieRecommendationResponse(BaseModel):
    """Response body for POST /api/v1/movies/recommendations."""

    query: str
    results: list[MovieResult]
