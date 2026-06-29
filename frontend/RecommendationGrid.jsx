import "./GenreBadge.css";

/**
 * Renders a comma-separated genre/tag string (e.g. "Action, Sci-Fi") as a
 * row of pill badges. The original app displayed this as a plain
 * markdown line ("**Genre:** Action, Sci-Fi"); badges read better in a
 * card-based UI while showing the exact same data.
 */
export default function GenreBadge({ genres }) {
  if (!genres) return null;
  const items = String(genres)
    .split(",")
    .map((g) => g.trim())
    .filter(Boolean);

  if (items.length === 0) return null;

  return (
    <div className="genre-badge-row">
      {items.map((genre) => (
        <span className="genre-badge" key={genre}>
          {genre}
        </span>
      ))}
    </div>
  );
}
