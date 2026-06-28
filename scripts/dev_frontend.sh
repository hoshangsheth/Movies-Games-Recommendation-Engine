#!/usr/bin/env bash
# Sets up and runs the frontend locally (without Docker).
set -euo pipefail

cd "$(dirname "$0")/../frontend"

if [ ! -f ".env" ]; then
  cp .env.example .env
fi

if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

echo "Starting Vite dev server on http://localhost:5173 ..."
npm run dev
