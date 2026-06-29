import RecommendationCard from "./RecommendationCard";
import "./RecommendationGrid.css";

export default function RecommendationGrid({ items, posterVariant, onSelect }) {
  return (
    <div className="recommendation-grid">
      {items.map((item, index) => (
        <RecommendationCard
          key={`${item.Title}-${index}`}
          title={item.Title}
          poster={item.Poster}
          posterVariant={posterVariant}
          rating={item.Rating || item.Vote_Average}
          genre={item.Genre || item.Genres}
          year={item.Year || item.Released}
          onDetails={() => onSelect(item)}
        />
      ))}
    </div>
  );
}
