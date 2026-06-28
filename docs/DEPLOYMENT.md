# Deployment

## Build instructions

**Backend image:**
```bash
docker build -t recommendation-engine-backend ./backend
```

**Frontend image** (static build served via nginx):
```bash
docker build -t recommendation-engine-frontend ./frontend
```

Or build both at once:
```bash
docker compose build
```

## Deploying the backend

The backend is a standard FastAPI app served by `uvicorn`, so it runs on any platform that accepts a
Docker container or a Python process: Render, Railway, Fly.io, AWS ECS/Fargate, Google Cloud Run, etc.

Environment variables to set in production:

| Variable | Purpose |
|---|---|
| `FRONTEND_ORIGIN` | Your deployed frontend's origin, added to the CORS allow-list |
| `MODEL_DOWNLOAD_DIR` | Writable directory for cached model artifacts (defaults to `/tmp/recommender-artifacts`) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service account credentials for the contact form (optional) |

The four model artifacts are downloaded from Google Drive on first startup and cached on disk for the
life of the container — there's no separate "build the model" deploy step. On platforms with ephemeral
or read-only filesystems, mount a writable volume at `MODEL_DOWNLOAD_DIR` (see the `model-artifacts`
volume in `docker-compose.yml` for the pattern) so the ~1–2 minute download doesn't repeat on every
cold start.

## Deploying the frontend

The frontend builds to static files (`npm run build` → `frontend/dist/`), so it can be deployed to any
static host: Vercel, Netlify, Cloudflare Pages, S3 + CloudFront, or the included nginx Docker image.

Set `VITE_API_BASE_URL` at build time to point at your deployed backend's URL, e.g.:
```bash
VITE_API_BASE_URL=https://api.yourdomain.com/api/v1 npm run build
```

If serving frontend and backend from the same domain behind a reverse proxy (as `frontend/nginx.conf`
does for the Docker Compose setup), you can leave `VITE_API_BASE_URL` as the default `/api/v1` and proxy
`/api/*` to the backend service instead.

## Docker Compose (single-host deployment)

For a quick single-VM deployment, `docker-compose.yml` at the repo root runs both services together,
with nginx in the frontend container proxying `/api/*` to the backend container over Docker's internal
network. This is sufficient for a portfolio demo or small-scale deployment; for production traffic,
prefer deploying the two services independently behind a real load balancer / CDN.

```bash
docker compose up --build -d
```

## Health checks

`GET /api/v1/health` returns `200` once the model artifacts have loaded successfully, and includes
`movies_loaded` / `games_loaded` counts — wire this into your platform's health-check / readiness probe
configuration.
