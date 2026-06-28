"""
Unified recommendations route (media_type-agnostic), plus the contact
form route. Both are HTTP-only — logic lives in the service layer.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_contact_service, get_recommendation_service
from app.schemas.contact import ContactRequest, ContactResponse
from app.services.contact_service import ContactService, ContactServiceError
from app.services.recommendation_service import MediaType, RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/{media_type}")
def recommend(
    media_type: MediaType,
    title: str,
    top_n: int = 10,
    service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        return service.get_recommendations(media_type, title, top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


contact_router = APIRouter(prefix="/contact", tags=["contact"])


@contact_router.post("", response_model=ContactResponse)
def submit_contact_form(
    payload: ContactRequest,
    service: ContactService = Depends(get_contact_service),
) -> ContactResponse:
    try:
        service.save_message(payload.name, payload.email, payload.message)
        return ContactResponse(success=True, detail="Message sent. I'll get back to you soon.")
    except ContactServiceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
