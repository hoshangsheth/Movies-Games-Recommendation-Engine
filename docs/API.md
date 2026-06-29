# FilmOracle API Documentation

# Overview

FilmOracle exposes a REST API built with FastAPI. The frontend communicates exclusively with these endpoints to retrieve recommendations, fetch metadata, and submit contact requests.

---

# Base URL

Development

```
http://localhost:8000
```

Production

```
https://<your-render-backend>
```

---

# API Structure

```
/api
├── movies
├── games
├── contact
└── health
```

> Endpoint names may evolve as the project grows. Keep this document synchronized with the implementation.

---

# Response Format

Successful responses return JSON.

Typical structure:

```json
{
  "success": true,
  "data": {}
}
```

Errors return an appropriate HTTP status code with an explanatory message.

---

# Movie Recommendation

Purpose

- Accept a movie title
- Generate Top-N similar movies
- Return enriched recommendation metadata

Example Request

```http
POST /api/movies/recommend
```

---

# Game Recommendation

Purpose

- Accept a game title
- Generate Top-N similar games
- Return enriched recommendation metadata

Example Request

```http
POST /api/games/recommend
```

---

# Contact

Accepts user messages from the Contact page.

Typical fields

- Name
- Email
- Message

---

# Health Check

Used by deployment platforms to verify backend availability.

```http
GET /health
```

---

# Status Codes

| Code | Meaning |
|------|---------|
|200|Success|
|400|Invalid Request|
|404|Resource Not Found|
|422|Validation Error|
|500|Internal Server Error|

---

# API Design Principles

- RESTful endpoints
- JSON communication
- Stateless requests
- Request validation
- Clear error responses
- Thin controllers with service-layer logic

---

# Future API

Planned additions include:

- Authentication
- User profiles
- Watchlists
- Recommendation history
- Personalized recommendations
- Analytics
- AI-powered recommendation endpoints
