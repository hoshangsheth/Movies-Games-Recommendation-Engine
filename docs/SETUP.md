# Setup & Development

## Prerequisites

- Python 3.12+
- Node.js 20+
- Docker & Docker Compose (optional, for containerized run)
- A Google Cloud service account with Sheets API access (optional, only needed for the contact form)

## Option A: Run with Docker Compose (recommended)

From the repo root:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# edit backend/.env if you want the contact form to work (see "Contact form setup" below)

docker compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)

## Option B: Run locally without Docker

**Backend:**
```bash
./scripts/dev_backend.sh
```
This creates a virtualenv, installs `backend/requirements.txt`, copies `.env.example` to `.env` if missing,
and starts `uvicorn` with `--reload` on port 8000.

**Frontend** (in a separate terminal):
```bash
./scripts/dev_frontend.sh
```
This installs npm dependencies and starts the Vite dev server on port 5173. Vite is configured to proxy
`/api/*` requests to `http://localhost:8000`, so no CORS configuration is needed in local dev.

## Contact form setup

The contact form writes submissions to a Google Sheet, exactly as the original Streamlit app did. To
enable it:

1. Create a Google Cloud service account with access to the Sheets API and Drive API.
2. Share the target spreadsheet with that service account's email address (Editor access).
3. Download the service account's JSON key.
4. Set `GOOGLE_SERVICE_ACCOUNT_JSON` in `backend/.env` to either:
   - the file path to that JSON key, or
   - the raw JSON content itself (useful when injecting via a CI/CD secret).
5. Update `CONTACT_SHEET_KEY` in `backend/app/core/config.py` (or via env var) if you're using a different
   spreadsheet than the original.

If this isn't configured, every other feature still works — only `POST /api/v1/contact` will return a
502 until credentials are set.

## Running tests

```bash
cd backend
source .venv/bin/activate
pytest
```

The test suite covers the text-normalization helpers, the similarity ranking logic, and both
recommendation engines (alias resolution, fuzzy matching, store-link formatting, trailer URL building,
and error handling) against small synthetic fixtures in `tests/conftest.py`.

## Linting

```bash
cd frontend
npm run lint
```

## Working with the notebooks

`backend/notebooks/` contains the offline data-collection and model-building notebooks, unchanged from
the original project. Re-running them regenerates the four pickled/`.npy` artifacts that the backend
downloads at startup. If you regenerate them, re-upload to Google Drive and update the corresponding
`*_FILE_ID` values in `backend/app/core/config.py`.
