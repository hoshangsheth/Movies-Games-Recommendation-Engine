import RecommendationCard from "./RecommendationCard";
import "./RecommendationGrid.css";

/**
 * 4-column responsive grid of recommendation cards, matching the
 * original's `num_cols = 4` Streamlit column layout for both the movies
 * and games pages.
 */
export default function RecommendationGrid({ items, posterVariant, onSelect }) {
  return (
    <div className="recommendation-grid">
      {items.map((item, index) => (
        <RecommendationCard
          key={`${item.Title}-${index}`}
          title={item.Title}
          poster={item.Poster}
          posterVariant={posterVariant}
          onDetails={() => onSelect(item)}
        />
      ))}
    </div>
  );
}
