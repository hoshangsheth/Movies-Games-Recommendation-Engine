import { Star, Plus } from "lucide-react";
import "./RecommendationGrid.css";

export default function RecommendationCard({ title, poster, posterVariant, rating, genre, year, onDetails }) {
  return (
    <div className="rec-card" onClick={onDetails} role="button" tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onDetails()}>
      <div className="rec-card-poster-wrap">
        <img
          className="rec-card-poster"
          src={poster}
          alt={title}
          loading="lazy"
          onError={(e) => {
            e.target.src = posterVariant === "game"
              ? "https://placehold.co/300x400/a855f7/white?text=Game"
              : "https://placehold.co/300x450/f97316/white?text=Movie";
          }}
        />
        {rating && (
          <div className="rec-card-rating">
            <Star size={10} fill="currentColor" strokeWidth={0} />
            {rating}
          </div>
        )}
        <div className="rec-card-add">
          <Plus size={18} strokeWidth={2.5} />
        </div>
      </div>
      <div className="rec-card-info">
        <div className="rec-card-title">{title}</div>
        {(genre || year) && (
          <div className="rec-card-meta">
            {genre && <span>{genre}</span>}
            {genre && year && <span className="rec-card-dot">•</span>}
            {year && <span>{year}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
