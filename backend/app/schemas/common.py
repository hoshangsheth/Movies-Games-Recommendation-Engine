"""Shared/common schemas used across multiple routes."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""

    status: str
    movies_loaded: int
    games_loaded: int


class ErrorResponse(BaseModel):
    """Standard error envelope returned by the global exception handler."""

    error: str
