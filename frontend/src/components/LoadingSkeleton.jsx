import "./LoadingSkeleton.css";

/**
 * Placeholder grid shown while a recommendation request is in flight.
 * The original Streamlit app had no loading state of its own (the whole
 * script just re-ran), so this is a genuine UX addition made possible by
 * the move to a real frontend with async requests.
 */
export default function LoadingSkeleton({ count = 8 }) {
  return (
    <div className="skeleton-grid" aria-label="Loading recommendations" role="status">
      {Array.from({ length: count }).map((_, i) => (
        <div className="skeleton-card" key={i}>
          <div className="skeleton-poster" />
          <div className="skeleton-title" />
        </div>
      ))}
    </div>
  );
}
