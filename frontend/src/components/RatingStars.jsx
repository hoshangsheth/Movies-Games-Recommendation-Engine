/**
 * Renders a numeric rating as star emoji, exactly matching the original
 * `render_rating_stars()` function: a full star per whole point, plus a
 * half-star glyph if the remainder is >= 0.5.
 */
export default function RatingStars({ rating }) {
  const numeric = Number(rating);
  if (Number.isNaN(numeric)) return null;

  const rounded = Math.round(numeric * 10) / 10;
  const fullStars = Math.floor(rounded);
  const hasHalfStar = rounded - fullStars >= 0.5;

  const stars = "⭐".repeat(fullStars) + (hasHalfStar ? "✬" : "");

  return (
    <span className="modal-rating-stars">
      {rounded} {stars}
    </span>
  );
}
