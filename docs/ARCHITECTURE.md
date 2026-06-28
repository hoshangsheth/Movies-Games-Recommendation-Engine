# Architecture

## Overview

```
React (Vite)  →  FastAPI  →  Service Layer  →  Recommendation Engine  →  Precomputed Artifacts
```

The system is split into two independently deployable applications joined by a REST API:

- **`frontend/`** — a React + Vite single-page app. Renders the UI, calls the backend, and owns no
  business logic.
- **`backend/`** — a FastAPI service that exposes recommendation and contact-form endpoints. Owns all
  business logic, data loading, and the recommendation engine itself.

## Why this split

The original project was a single 1,200-line Streamlit script that mixed data loading, recommendation
logic, and page rendering in one file. That made the recommendation engine impossible to reuse outside
of Streamlit and impossible to unit test without running the whole UI. Splitting frontend and backend,
and layering the backend internally, fixes both problems without changing what the system actually does.

## Backend layers

```
app/api/routes/      → HTTP only: parse request, call a service, shape the response, map errors to status codes
app/services/        → business logic: orchestrates the recommender, builds response schemas
app/recommender/      → the actual recommendation engine: alias resolution, fuzzy matching, similarity lookup
app/schemas/          → Pydantic request/response models (also produces the OpenAPI docs at /docs)
app/core/             → configuration, constants (alias dictionaries), logging setup
```

Each layer only talks to the layer directly below it. Routes never touch a dataframe directly; they call
a service. Services never do fuzzy matching directly; they call the recommender. This means the
recommender package has zero FastAPI or HTTP imports — it can be imported and tested as a plain Python
library, or reused in a CLI, notebook, or batch job with no changes.

### Request flow for `POST /api/v1/movies/recommendations`

1. `api/routes/movies.py` receives the request, validates it against `MovieRecommendationRequest`.
2. It calls `MovieService.get_recommendations(title, top_n)`.
3. `MovieService` delegates to `MovieRecommendationEngine.recommend()`.
4. The engine: cleans the input → checks the alias dictionary → fuzzy-matches against the dataset's
   `title_clean` column (`rapidfuzz`) → looks up the matched row's index → slices that row out of the
   precomputed cosine-similarity matrix → sorts and takes the top N → builds the result dicts.
5. `MovieService` validates each dict against the `MovieResult` schema and returns a
   `MovieRecommendationResponse`.
6. The route returns that response as JSON; FastAPI handles serialization.

Movies and games follow an identical shape — `GameRecommendationEngine` mirrors `MovieRecommendationEngine`
field-for-field, since that's what the original `recommend_movies()` / `recommend_games()` functions did.

## Where the model lives

There is no live model training or vectorization happening in the API. The TF-IDF vectorization and
cosine-similarity computation are done **offline**, in the notebooks under `backend/notebooks/`, exactly
as in the original project. Those notebooks produce four artifacts:

- `movies_recommended.pkl` — the movies dataframe with all display fields and a `title_clean` column
- `cosine_sim_movies.pkl` — the precomputed movie-to-movie cosine similarity matrix
- `games_recommended.pkl` — the games dataframe
- `cosine_sim_games.npy` — the precomputed game-to-game cosine similarity matrix

These are too large to commit to git, so they're hosted on Google Drive and downloaded once at FastAPI
startup via `gdown` (see `app/recommender/data_loader.py`) — the same mechanism the original Streamlit app
used, just moved into a proper startup hook instead of running at module import time.

## Frontend structure

```
components/   → reusable UI pieces (Navbar, Hero, SearchBar, RecommendationCard, modals, etc.)
pages/        → one component per route: Home, Movies, Games, Contact
hooks/        → useRecommendations — the loading/error/result state machine for a search
services/     → api.js — the only file that calls fetch(); every component goes through it
```

Routing is handled by `react-router-dom`. Visual styling intentionally replicates the original
Streamlit app's "CineVerse" dark theme — same color values, same fonts (Cormorant Garamond + Outfit),
same per-page background gradients — just expressed as CSS files and React components instead of
inline `st.markdown(..., unsafe_allow_html=True)` strings.

## Error handling

The recommendation engine raises a plain `ValueError` for: empty input, no fuzzy match found, or an
out-of-range index — collapsing the original's three separate `except` blocks (`ValueError`, `IndexError`,
bare `Exception`) into one exception type at the engine boundary. The service layer lets that propagate;
the route layer catches `ValueError` and returns an HTTP 404 with the error message as `detail`. A global
exception handler in `main.py` catches anything else and returns a generic 500, so no internal stack trace
or implementation detail ever reaches the client.
