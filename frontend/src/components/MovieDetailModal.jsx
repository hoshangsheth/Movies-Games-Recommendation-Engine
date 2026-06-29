import Modal from "./Modal";

/**
 * Movie detail modal — redesigned to match FilmOracle's theme.
 * Left col: poster + Watch Now CTA.
 * Right col: title, tags, description, metadata grid, cast row, trailer.
 */
export default function MovieDetailModal({ movie, onClose }) {
  const castNames  = (movie["Top Cast"]     || "").split(",").map((s) => s.trim()).filter(Boolean);
  const castImages = (movie["Cast Picture"] || "").split(",").map((s) => s.trim());

  // Build star display out of 5
  const ratingNum = parseFloat(movie.Rating);
  const fullStars = isNaN(ratingNum) ? 0 : Math.round(ratingNum / 2); // assumes /10 scale

  // Split genres into individual tags
  const genres = (movie.Genre || "").split(",").map((g) => g.trim()).filter(Boolean);

  return (
    <Modal onClose={onClose}>
      <div className="modal-body">

        {/* ── Left: Poster + CTA ── */}
        <div className="modal-poster-col">
          <div className="modal-poster-wrap">
            <img
              className="modal-poster"
              src={movie.Poster}
              alt={movie.Title}
              onError={(e) => {
                e.target.src = "https://placehold.co/300x450/f97316/white?text=Movie";
              }}
            />
            {movie.Rating && (
              <div className="modal-rating-badge">
                ★ {movie.Rating}
              </div>
            )}
          </div>

          <div className="modal-cta-stack">
            {movie.Stream && (
              <a
                className="modal-btn-primary"
                href={movie.Stream}
                target="_blank"
                rel="noreferrer"
              >
                ▶ Watch Now
              </a>
            )}
          </div>
        </div>

        {/* ── Right: Content ── */}
        <div className="modal-content-col">

          {/* Title */}
          <h2 className="modal-title">{movie.Title}</h2>

          {/* Genre + Language + Year pills */}
          <div className="modal-tag-row">
            {genres.map((g) => (
              <span key={g} className="modal-tag">{g}</span>
            ))}
            {movie.Language && (
              <span className="modal-tag orange">{movie.Language}</span>
            )}
            {movie["Release Date"] && (
              <span className="modal-tag neutral">
                {String(movie["Release Date"]).slice(0, 4)}
              </span>
            )}
          </div>

          {/* Description */}
          {movie.Description && (
            <p className="modal-description">{movie.Description}</p>
          )}

          <hr className="modal-divider" />

          {/* Metadata grid */}
          <div className="modal-meta-grid">
            {movie["Release Date"] && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Release Date</span>
                <span className="modal-meta-value">{movie["Release Date"]}</span>
              </div>
            )}
            {movie.Language && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Language</span>
                <span className="modal-meta-value">{movie.Language}</span>
              </div>
            )}
            {movie.Rating && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Rating</span>
                <span className="modal-meta-value">⭐ {movie.Rating} / 10</span>
              </div>
            )}
            {movie.Genre && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Genre</span>
                <span className="modal-meta-value">{movie.Genre}</span>
              </div>
            )}
          </div>

          {/* Cast */}
          {castNames.length > 0 && (
            <>
              <hr className="modal-divider" />
              <div>
                <p className="modal-section-heading">Top Cast</p>
                <div className="modal-cast-row">
                  {castNames.map((name, i) => (
                    <div className="modal-cast-member" key={`${name}-${i}`}>
                      {castImages[i] ? (
                        <img
                          className="modal-cast-photo"
                          src={castImages[i]}
                          alt={name}
                          onError={(e) => { e.target.style.display = "none"; }}
                        />
                      ) : (
                        <div
                          className="modal-cast-photo"
                          style={{
                            background: "rgba(168,85,247,0.10)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "1.1rem",
                          }}
                        >
                          🎭
                        </div>
                      )}
                      <span className="modal-cast-name">{name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Trailer */}
          {movie.Trailer && (
            <>
              <hr className="modal-divider" />
              <div>
                <p className="modal-section-heading">Trailer</p>
                <iframe
                  className="modal-trailer"
                  src={movie.Trailer.replace("watch?v=", "embed/")}
                  title={`${movie.Title} trailer`}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            </>
          )}

        </div>
      </div>
    </Modal>
  );
}
