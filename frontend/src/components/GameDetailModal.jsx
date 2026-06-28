import { useState } from "react";
import Modal from "./Modal";

/**
 * Game details dialog, matching the original's `show_game_details()`
 * function: poster, metadata fields, up to 3 store links, official
 * website link, and a screenshot carousel with Previous/Next buttons
 * (the original tracked the current index in `st.session_state`; here
 * it's plain `useState`).
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

  function showPrevious() {
    setScreenshotIndex((i) => (i - 1 + screenshots.length) % screenshots.length);
  }

  function showNext() {
    setScreenshotIndex((i) => (i + 1) % screenshots.length);
  }

  return (
    <Modal onClose={onClose}>
      <div className="modal-body">
        <div>
          <img className="modal-poster" src={game.Poster} alt={game.Title} />
        </div>
        <div>
          <div className="modal-title">{game.Title}</div>
          <p className="modal-field">
            <strong>Description:</strong> {game.Description}
          </p>
          <p className="modal-field">
            <strong>Developer:</strong> {game.Developer}
          </p>
          <p className="modal-field">
            <strong>Publisher:</strong> {game.Publisher}
          </p>
          <p className="modal-field">
            <strong>Genre:</strong> {game.Genre}
          </p>
          <p className="modal-field">
            <strong>Release Date:</strong> {game["Release Date"]}
          </p>
          <p className="modal-field">
            <strong>Available On:</strong> {game.Platforms}
          </p>
          <p className="modal-field">
            <strong>Tags:</strong> {game.Tags}
          </p>
          <p className="modal-field">
            <strong>ESRB Rating:</strong> {game.ESRB_Rating}
          </p>

          {storeLinks.map(({ name, url }) => (
            <a key={name} className="modal-link-button small" href={url} target="_blank" rel="noreferrer">
              🛒 {name}
            </a>
          ))}

          {game.Website && (
            <a className="modal-link-button small" href={game.Website} target="_blank" rel="noreferrer">
              🌐 Official Website
            </a>
          )}

          <h3 className="modal-screenshots-heading">📸 Screenshots</h3>
          {screenshots.length > 0 ? (
            <>
              <img
                className="modal-screenshot"
                src={screenshots[screenshotIndex]}
                alt={`${game.Title} screenshot ${screenshotIndex + 1}`}
              />
              <div className="modal-screenshot-controls">
                <button onClick={showPrevious}>⬅️ Previous</button>
                <button onClick={showNext}>Next ➡️</button>
              </div>
            </>
          ) : (
            <p className="modal-field">No screenshots available.</p>
          )}
        </div>
      </div>
    </Modal>
  );
}
