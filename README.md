# 🎬 Movies & Games 🎮 Recommendation Engine

A content-based recommendation system that lets you search for movies and games — then delivers
personalized, relevant suggestions based on your input. Powered by cosine similarity on
**TF-IDF vectorized features** and enhanced with smart fuzzy search for seamless, intuitive matching.

This repository is a **production-grade monorepo refactor** of the original Streamlit prototype: the
same recommendation engine and 100% of the same functionality, now split into a FastAPI backend and a
React frontend with proper separation of concerns, tests, Docker support, and documentation.

---

## 📌 Features

🔍 **Smart Search with Aliases & Fuzzy Matching**
🧠 **Cosine Similarity-Based Recommendations**
🎞️ **Movies**: Posters, trailers, cast pictures, descriptions, genres, watch links, and ratings
🕹️ **Games**: Store links, developer/publisher, tags, ESRB ratings, website, and screenshots
🎨 **CineVerse UI**: Dark, purple-accented theme with animated transitions and a responsive sidebar
🧭 **Intuitive Navigation**: Home, Recommend Movies, Recommend Games, and Contact pages
📩 **Google Sheets Integration** for the contact form
🧪 **Tested recommendation engine**, decoupled from any web framework
🐳 **Docker-ready**, with a `docker-compose.yml` for one-command local deployment

---

## 🏗️ Architecture

```
React (Frontend)  →  FastAPI (Backend)  →  Service Layer  →  Recommendation Engine  →  Precomputed Artifacts
```

```
movies-games-recommendation-engine/
├── backend/             FastAPI app: API routes, services, recommendation engine, tests
│   ├── app/
│   │   ├── api/         Route handlers + dependency injection (HTTP only, no business logic)
│   │   ├── core/        Config, alias dictionaries, logging
│   │   ├── recommender/ The engine itself: preprocessing, similarity lookup, movie/game engines
│   │   ├── services/    Business logic between routes and the engine
│   │   └── schemas/     Pydantic request/response models
│   ├── data/             Raw movie/game CSVs
│   ├── notebooks/        Offline TF-IDF + cosine-similarity model building
│   └── tests/             Unit tests for the engine and utilities
├── frontend/            React + Vite SPA
│   └── src/
│       ├── components/   Navbar, Hero, SearchBar, RecommendationCard/Grid, detail modals, etc.
│       ├── pages/         Home, Movies, Games, Contact
│       ├── hooks/          useRecommendations — search state management
│       └── services/       api.js — the only file that talks to the backend
├── docs/                 Architecture, API, setup, and deployment docs
├── scripts/               Local dev convenience scripts
└── docker-compose.yml
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full breakdown of how requests flow through
the system and why it's structured this way.

---

## 🧠 Recommendation Logic

- **Data Cleaning & Preprocessing**: Titles cleaned using regex (`app/utils/text.py`)
- **Fuzzy Search**: Implemented via `rapidfuzz` to handle partial and alias-based searches
- **Similarity Computation**: Precomputed Cosine Similarity Matrix (Pickle + Numpy), built offline in
  `backend/notebooks/`
- **Smart Aliasing**: Robust dictionaries for common abbreviations (e.g., "ZNMD" → *Zindagi Na Milegi
  Dobara*), kept in `app/core/constants.py`
- **Metadata Enhancement**: Enriched recommendations with trailers, store links, cast, ratings,
  screenshots, etc.

The recommendation algorithm itself is unchanged from the original project — this refactor is about
*where the code lives and how it's organized*, not about how recommendations are computed.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | React, Vite, React Router |
| Data / ML | Pandas, NumPy, RapidFuzz, scikit-learn (offline, in notebooks) |
| Storage | Pickle / `.npy` similarity matrices, downloaded via `gdown` |
| Integrations | Google Sheets (`gspread`) for the contact form |
| Infra | Docker, Docker Compose, nginx |
| Testing | pytest |

---

## 🚀 Getting Started

**With Docker Compose:**
```bash
git clone https://github.com/hoshangsheth/Movies-Games-Recommendation-Engine.git
cd Movies-Games-Recommendation-Engine
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
docker compose up --build
```
Frontend: http://localhost · Backend docs: http://localhost:8000/docs

**Without Docker:**
```bash
./scripts/dev_backend.sh    # terminal 1 — FastAPI on :8000
./scripts/dev_frontend.sh   # terminal 2 — Vite on :5173
```

Full setup, contact-form configuration, and test instructions: [`docs/SETUP.md`](docs/SETUP.md)
Deployment instructions: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
API reference: [`docs/API.md`](docs/API.md)

---

## 📈 Why This Project?

Blending NLP techniques like TF-IDF with classic ML similarity measures and real-time API data fetching,
this project showcases how to build scalable, intelligent recommendation engines — and, in its current
form, how to take that engine out of a notebook-adjacent script and structure it the way a real product
backend would be structured: layered, tested, documented, and containerized.

---

## 📄 License

[MIT](LICENSE)
