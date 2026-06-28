import "./RecommendationGrid.css";

/**
 * One poster + title + "Details" button. Matches the original's
 * `st.markdown` poster block + title chip + `st.button("Details", ...)`
 * that opened the `@st.dialog` modal.
 */
export default function RecommendationCard({ title, poster, posterVariant, onDetails }) {
  return (
    <div className="recommendation-card">
      <div className="recommendation-card-poster-wrap">
        <img
          className={`recommendation-card-poster${posterVariant === "game" ? " poster-game" : ""}`}
          src={poster}
          alt={title}
          loading="lazy"
        />
      </div>
      <div className="recommendation-card-title">{title}</div>
      <button className="recommendation-card-button" onClick={onDetails}>
        Details
      </button>
    </div>
  );
}
