"""Pydantic schemas for the contact form API."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class ContactRequest(BaseModel):
    """Request body for POST /api/v1/contact — mirrors the original Streamlit contact form fields."""

    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    message: str = Field(..., min_length=1, max_length=5000)


class ContactResponse(BaseModel):
    """Response body for POST /api/v1/contact."""

    success: bool
    detail: str
