# FilmOracle Architecture

# Overview

FilmOracle follows a modular, service-oriented architecture designed to separate presentation, API, business logic, and recommendation logic. This separation keeps the project maintainable, testable, and easy to extend.

```
User
  │
  ▼
React + Vite Frontend
  │
  ▼
FastAPI REST API
  │
  ├──────────────┐
  ▼              ▼
Recommendation   Contact
Service          Service
  │
  ▼
Recommendation Engine
  │
  ├── Dataset Loader
  ├── TF-IDF Vectorizer
  ├── Cosine Similarity
  └── Metadata Processing
  │
  ▼
Movie / Game Datasets
```

---

# Project Layers

## Frontend

Responsible for:

- Navigation
- Search experience
- Recommendation cards
- Detail modals
- Contact form
- Theme switching

The frontend communicates exclusively with the FastAPI backend through REST endpoints.

---

## API Layer

The API layer exposes endpoints consumed by the React application.

Responsibilities include:

- Request validation
- Error handling
- Delegating work to services
- Returning JSON responses

Business logic is intentionally kept out of route handlers.

---

## Service Layer

The service layer contains application logic.

Typical responsibilities:

- Search orchestration
- Recommendation generation
- Metadata enrichment
- Contact processing
- Response formatting

Keeping this layer independent makes the API thin and easier to test.

---

## Recommendation Engine

The recommendation engine is the core of FilmOracle.

Current workflow:

1. Accept user title.
2. Validate input.
3. Resolve title against dataset.
4. Vectorize using TF-IDF.
5. Compute cosine similarity.
6. Rank candidates.
7. Return the Top-N recommendations.
8. Enrich with additional metadata.

The engine is intentionally isolated so future recommendation algorithms can be introduced without affecting the API or frontend.

---

# Request Lifecycle

```
User Search
    │
    ▼
React UI
    │
    ▼
FastAPI Endpoint
    │
    ▼
Recommendation Service
    │
    ▼
Recommendation Engine
    │
    ▼
Similarity Ranking
    │
    ▼
JSON Response
    │
    ▼
Frontend Rendering
```

---

# Design Principles

- Modular architecture
- Clear separation of concerns
- Stateless API
- Reusable services
- Scalable folder structure
- Production-oriented organization

---

# Future Evolution

The current architecture is designed to accommodate future capabilities including:

- Hybrid recommendation models
- Vector databases
- LLM-powered recommendations
- User profiles
- Recommendation history
- Authentication
- Analytics
- Caching
- Background jobs

These additions can be introduced with minimal changes to the existing architecture because responsibilities are already separated.

---

# Documentation

See the remaining documents for implementation details:

- API.md
- SETUP.md
- DEPLOYMENT.md
