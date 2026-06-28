"""
FastAPI application entrypoint.

Replaces `st.set_page_config(...)` and the top-level script execution of
the original Streamlit app with a proper ASGI app: CORS for the React
frontend, a startup hook that preloads the recommender artifacts (same
role as the original's `@st.cache_resource` loaders), and a global
exception handler so unhandled errors return a consistent JSON shape
instead of leaking stack traces.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.recommender.data_loader import load_artifacts

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "API for the Movies & Games Recommendation Engine — content-based "
        "recommendations using TF-IDF + cosine similarity, with fuzzy "
        "title matching and alias resolution."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
def preload_recommender_artifacts() -> None:
    """
    Download and load the movie/game dataframes + similarity matrices once
    at startup, so the first request isn't slowed down by a cold load.
    """
    logger.info("Preloading recommender artifacts at startup...")
    load_artifacts()
    logger.info("Startup artifact preload complete.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})


@app.get("/")
def root() -> dict:
    return {"name": settings.APP_NAME, "docs": "/docs", "health": f"{settings.API_V1_PREFIX}/health"}
