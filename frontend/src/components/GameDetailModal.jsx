import { useState } from "react";
import Modal from "./Modal";

/**
 * Game detail modal — redesigned to match FilmOracle's theme.
 * Left col: poster + store link CTAs.
 * Right col: title, tags, description, metadata grid, screenshot carousel.
 */
export default function GameDetailModal({ game, onClose }) {
  const [screenshotIndex, setScreenshotIndex] = useState(0);

  const screenshots = (game.Screenshots || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const storeLinks = (game.Stores || "")
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.includes(":"))
    .map((entry) => {
      const [name, ...rest] = entry.split(":");
      return { name: name.trim(), url: rest.join(":").trim() };
    })
    .slice(0, 3);

  const platforms = (game.Platforms || "").split(",").map((p) => p.trim()).filter(Boolean);
  const tags      = (game.Tags      || "").split(",").map((t) => t.trim()).filter(Boolean).slice(0, 6);
  const genres    = (game.Genre     || "").split(",").map((g) => g.trim()).filter(Boolean);

  function showPrev() {
    setScreenshotIndex((i) => (i - 1 + screenshots.length) % screenshots.length);
  }
  function showNext() {
    setScreenshotIndex((i) => (i + 1) % screenshots.length);
  }

  return (
    <Modal onClose={onClose}>
      <div className="modal-body">

        {/* ── Left: Poster + Store CTAs ── */}
        <div className="modal-poster-col">
          <div className="modal-poster-wrap">
            <img
              className="modal-poster"
              src={game.Poster}
              alt={game.Title}
              onError={(e) => {
                e.target.src = "https://placehold.co/300x400/a855f7/white?text=Game";
              }}
            />
            {game.ESRB_Rating && (
              <div className="modal-rating-badge" style={{ color: "white" }}>
                {game.ESRB_Rating}
              </div>
            )}
          </div>

          {/* Store links as CTA buttons */}
          {(storeLinks.length > 0 || game.Website) && (
            <div className="modal-cta-stack">
              {storeLinks.map(({ name, url }) => (
                <a
                  key={name}
                  className="modal-btn-primary"
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                >
                  🛒 {name}
                </a>
              ))}
              {game.Website && (
                <a
                  className="modal-btn-ghost"
                  href={game.Website}
                  target="_blank"
                  rel="noreferrer"
                >
                  🌐 Official Site
                </a>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Content ── */}
        <div className="modal-content-col">

          {/* Title */}
          <h2 className="modal-title">{game.Title}</h2>

          {/* Genre + platform pills */}
          <div className="modal-tag-row">
            {genres.map((g) => (
              <span key={g} className="modal-tag">{g}</span>
            ))}
            {platforms.slice(0, 3).map((p) => (
              <span key={p} className="modal-tag orange">{p}</span>
            ))}
            {game["Release Date"] && (
              <span className="modal-tag neutral">
                {String(game["Release Date"]).slice(0, 4)}
              </span>
            )}
          </div>

          {/* Description */}
          {game.Description && (
            <p className="modal-description">{game.Description}</p>
          )}

          <hr className="modal-divider" />

          {/* Metadata grid */}
          <div className="modal-meta-grid">
            {game.Developer && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Developer</span>
                <span className="modal-meta-value">{game.Developer}</span>
              </div>
            )}
            {game.Publisher && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Publisher</span>
                <span className="modal-meta-value">{game.Publisher}</span>
              </div>
            )}
            {game["Release Date"] && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">Release Date</span>
                <span className="modal-meta-value">{game["Release Date"]}</span>
              </div>
            )}
            {game.ESRB_Rating && (
              <div className="modal-meta-item">
                <span className="modal-meta-label">ESRB Rating</span>
                <span className="modal-meta-value">{game.ESRB_Rating}</span>
              </div>
            )}
            {game.Platforms && (
              <div className="modal-meta-item" style={{ gridColumn: "1 / -1" }}>
                <span className="modal-meta-label">Available On</span>
                <span className="modal-meta-value">{game.Platforms}</span>
              </div>
            )}
          </div>

          {/* Tags */}
          {tags.length > 0 && (
            <>
              <hr className="modal-divider" />
              <div>
                <p className="modal-section-heading">Tags</p>
                <div className="modal-tag-row">
                  {tags.map((tag) => (
                    <span key={tag} className="modal-tag neutral">{tag}</span>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Screenshot carousel */}
          {screenshots.length > 0 && (
            <>
              <hr className="modal-divider" />
              <div className="modal-screenshots-wrap">
                <p className="modal-section-heading">Screenshots</p>
                <img
                  className="modal-screenshot"
                  src={screenshots[screenshotIndex]}
                  alt={`${game.Title} screenshot ${screenshotIndex + 1}`}
                />
                <div className="modal-screenshot-controls">
                  <button
                    className="modal-screenshot-btn"
                    onClick={showPrev}
                    disabled={screenshots.length <= 1}
                  >
                    ← Prev
                  </button>
                  <span className="modal-screenshot-counter">
                    {screenshotIndex + 1} / {screenshots.length}
                  </span>
                  <button
                    className="modal-screenshot-btn"
                    onClick={showNext}
                    disabled={screenshots.length <= 1}
                  >
                    Next →
                  </button>
                </div>
              </div>
            </>
          )}

        </div>
      </div>
    </Modal>
  );
}
