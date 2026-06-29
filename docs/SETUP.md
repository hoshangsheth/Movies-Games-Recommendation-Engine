# FilmOracle Setup Guide

# Overview

This guide explains how to set up FilmOracle for local development.

FilmOracle consists of:

- React + Vite frontend
- FastAPI backend
- Movie and Game recommendation engine

---

# Prerequisites

Install the following before getting started:

- Python 3.11+
- Node.js 20+
- npm
- Git

(Optional)

- Docker Desktop

---

# Clone Repository

```bash
git clone <repository-url>

cd FilmOracle
```

---

# Backend Setup

Navigate to the backend directory.

```bash
cd backend
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the backend.

```bash
uvicorn app.main:app --reload
```

Backend default URL

```
http://localhost:8000
```

---

# Frontend Setup

Navigate to frontend.

```bash
cd frontend
```

Install packages.

```bash
npm install
```

Start development server.

```bash
npm run dev
```

Default frontend URL

```
http://localhost:5173
```

---

# Environment Variables

Create a `.env` file if required by the project.

Typical variables include:

```
API_BASE_URL=
TMDB_API_KEY=
RAWG_API_KEY=
```

Only include variables that are actually used in the implementation.

---

# Docker

Build the application.

```bash
docker compose up --build
```

Stop containers.

```bash
docker compose down
```

---

# Development Workflow

1. Start backend.
2. Start frontend.
3. Open browser.
4. Search movies or games.
5. Verify API responses.
6. Test contact form.

---

# Troubleshooting

## Backend won't start

- Verify Python version.
- Activate the virtual environment.
- Install requirements again.

## Frontend won't connect

- Ensure backend is running.
- Verify API URL.
- Check CORS configuration.

## Missing packages

Run:

```bash
pip install -r requirements.txt

npm install
```

---

# Next Steps

After setup is complete:

- Read `ARCHITECTURE.md`
- Explore `API.md`
- Review `DEPLOYMENT.md`
