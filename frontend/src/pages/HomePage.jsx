import Hero from "../components/Hero";
import Footer from "../components/Footer";
import "../styles/layout.css";
import "./HomePage.css";

export default function HomePage() {
  return (
    <>
      <div className="page-bg-home" />
      <Hero
        variant="home"
        eyebrow="AI-Powered Discovery Engine"
        title="Your next obsession"
        titleAccent="is one search away."
        subtitle="Tell us what you love — a movie, a game, a vibe — and we'll surface titles you didn't know you needed. Powered by cosine similarity across thousands of titles."
      />

      <div className="feature-cards">
        <div className="feature-card feature-movies">
          <div className="feature-card-icon">🎬</div>
          <div className="feature-card-title">Cinema Picks</div>
          <div className="feature-card-body">
            From Bollywood classics to Hollywood blockbusters — our algorithm finds films that match your
            exact taste profile.
          </div>
        </div>
        <div className="feature-card feature-games">
          <div className="feature-card-icon">🎮</div>
          <div className="feature-card-title">Game Curation</div>
          <div className="feature-card-body">
            PC, PS5, Xbox — enter any game title and we'll map your preferences across genres, tags, and
            playstyle.
          </div>
        </div>
      </div>

      <div className="why-section">
        <div className="why-eyebrow">Why CineVerse</div>
        <div className="why-grid">
          <div className="why-card">
            <div className="why-card-icon">⚡</div>
            <div className="why-card-title">Instant Results</div>
            <div className="why-card-body">No scrolling. No algorithms you can't see. Just results.</div>
          </div>
          <div className="why-card">
            <div className="why-card-icon">🧠</div>
            <div className="why-card-title">Fuzzy Search</div>
            <div className="why-card-body">
              Typos? Partial titles? Nicknames? We've got you covered.
            </div>
          </div>
          <div className="why-card">
            <div className="why-card-icon">🎞️</div>
            <div className="why-card-title">Rich Details</div>
            <div className="why-card-body">
              Trailers, ratings, cast, and streaming links — all in one place.
            </div>
          </div>
        </div>

        <div className="home-callout">
          ← Use the sidebar to navigate between Movies, Games, and Contact. Start by selecting a section
          and entering a title you already love.
        </div>
      </div>

      <Footer />
    </>
  );
}
