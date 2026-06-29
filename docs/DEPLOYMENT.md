# FilmOracle Deployment Guide

# Overview

FilmOracle uses a split deployment architecture:

- Frontend → Vercel
- Backend → Render
- Source Code → GitHub

This approach provides fast frontend delivery while allowing the backend to run independently.

---

# Production Architecture

```text
                 GitHub Repository
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   Vercel Deployment             Render Deployment
   React + Vite                  FastAPI Backend
        │                               │
        └──────────────┬────────────────┘
                       ▼
                    End Users
```

---

# Backend Deployment (Render)

## Create a Web Service

Connect the GitHub repository to Render.

Typical configuration:

- Runtime: Python
- Root Directory: backend
- Build Command:

```bash
pip install -r requirements.txt
```

Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

# Frontend Deployment (Vercel)

Import the repository into Vercel.

Typical settings:

- Framework: Vite
- Root Directory: frontend

Build Command

```bash
npm run build
```

Output Directory

```
dist
```

---

# Environment Variables

Backend

```
TMDB_API_KEY=
RAWG_API_KEY=
```

Frontend

```
VITE_API_BASE_URL=
```

Only configure variables actually used by the project.

---

# Docker Deployment

Build containers.

```bash
docker compose up --build
```

Stop containers.

```bash
docker compose down
```

---

# CORS

Ensure the backend allows requests from the deployed frontend domain.

Typical production origin:

```
https://your-vercel-app.vercel.app
```

---

# Health Check

Expose a health endpoint for deployment verification.

Example

```http
GET /health
```

Deployment platforms can periodically call this endpoint to verify availability.

---

# Production Checklist

Before deployment:

- Install dependencies
- Configure environment variables
- Verify API URLs
- Enable CORS
- Test health endpoint
- Build frontend
- Verify backend starts successfully

---

# Troubleshooting

## Backend Build Failed

- Check Python version
- Verify requirements.txt
- Inspect Render logs

## Frontend Build Failed

- Verify Node.js version
- Run npm install
- Fix TypeScript or build errors

## CORS Errors

- Confirm frontend URL is allowed by the backend.
- Restart the backend after updating CORS settings.

## API Connection Issues

- Verify the frontend points to the production backend URL.
- Confirm backend health endpoint is reachable.

---

# Future Improvements

Potential production enhancements:

- CI/CD pipelines
- Automated testing
- GitHub Actions
- Monitoring
- Logging
- Caching
- CDN optimization
- Container orchestration
