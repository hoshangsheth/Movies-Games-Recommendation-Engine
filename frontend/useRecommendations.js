import Modal from "./Modal";
import RatingStars from "./RatingStars";

/**
 * Movie details dialog, matching the original's `show_movie_details()`
 * function: poster, description, genre, language, release date, rating
 * stars, cast row with photos, "Watch Now" link, and embedded trailer.
 */
export default function MovieDetailModal({ movie, onClose }) {
  const castNames = (movie["Top Cast"] || "").split(",").map((s) => s.trim());
  const castImages = (movie["Cast Picture"] || "").split(",").map((s) => s.trim());

  return (
    <Modal onClose={onClose}>
      <div className="modal-body">
        <div>
          <img className="modal-poster" src={movie.Poster} alt={movie.Title} />
        </div>
        <div>
          <div className="modal-title">{movie.Title}</div>
          <p className="modal-field">
            <strong>Description:</strong> {movie.Description}
          </p>
          <p className="modal-field">
            <strong>Genre:</strong> {movie.Genre}
          </p>
          <p className="modal-field">
            <strong>Language:</strong> {movie.Language}
          </p>
          <p className="modal-field">
            <strong>Release Date:</strong> {movie["Release Date"]}
          </p>
          <p className="modal-field">
            <strong>Rating:</strong> <RatingStars rating={movie.Rating} />
          </p>

          <p className="modal-field">
            <strong>Top Cast:</strong>
          </p>
          <div className="modal-cast-row">
            {castNames.map((name, i) => (
              <div className="modal-cast-member" key={`${name}-${i}`}>
                {castImages[i] && <img className="modal-cast-photo" src={castImages[i]} alt={name} />}
                <span className="modal-cast-name">{name}</span>
              </div>
            ))}
          </div>

          {movie.Stream && (
            <a className="modal-link-button" href={movie.Stream} target="_blank" rel="noreferrer">
              ▶ Watch Now
            </a>
          )}

          {movie.Trailer && (
            <iframe
              className="modal-trailer"
              src={movie.Trailer.replace("watch?v=", "embed/")}
              title={`${movie.Title} trailer`}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          )}
        </div>
      </div>
    </Modal>
  );
}
