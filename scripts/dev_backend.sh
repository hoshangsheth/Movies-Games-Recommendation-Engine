#!/usr/bin/env bash
# Sets up and runs the backend locally (without Docker).
set -euo pipefail

cd "$(dirname "$0")/../backend"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created backend/.env from .env.example — fill in GOOGLE_SERVICE_ACCOUNT_JSON if you need the contact form."
fi

echo "Starting FastAPI on http://localhost:8000 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
