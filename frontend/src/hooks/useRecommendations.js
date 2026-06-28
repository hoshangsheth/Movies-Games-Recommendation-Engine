import { useCallback, useState } from "react";
import { ApiError } from "../services/api";

/**
 * Encapsulates the loading / error / results state for a recommendation
 * search, regardless of whether it's backed by the movies or games
 * endpoint. Replaces the original's `st.session_state.recommend_triggered`
 * + re-run-the-whole-script pattern with a normal async request lifecycle.
 */
export default function useRecommendations(fetchFn) {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");

  const search = useCallback(
    async (title) => {
      setLoading(true);
      setError(null);
      setQuery(title);
      try {
        const data = await fetchFn(title);
        setResults(data.results);
      } catch (err) {
        setResults(null);
        setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [fetchFn]
  );

  return { results, error, loading, query, search };
}
