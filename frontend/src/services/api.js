/**
 * Thin wrapper around the FastAPI backend. Every network call the app
 * makes goes through here, so components never construct fetch() calls
 * directly — this mirrors the original Streamlit app calling
 * `recommend_movies()` / `recommend_games()` as the single source of
 * data, just over HTTP now instead of an in-process function call.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || body.error || detail;
    } catch {
      // response had no JSON body; fall back to the generic message
    }
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

export function fetchMovieRecommendations(title, topN = 10) {
  return request("/movies/recommendations", {
    method: "POST",
    body: JSON.stringify({ title, top_n: topN }),
  });
}

export function fetchGameRecommendations(title, topN = 10) {
  return request("/games/recommendations", {
    method: "POST",
    body: JSON.stringify({ title, top_n: topN }),
  });
}

export function submitContactForm({ name, email, message }) {
  return request("/contact", {
    method: "POST",
    body: JSON.stringify({ name, email, message }),
  });
}

export { ApiError };
