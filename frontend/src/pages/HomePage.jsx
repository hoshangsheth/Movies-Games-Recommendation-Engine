import { Clapperboard, Gamepad2, Zap, SearchCheck, LayoutList, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import "../styles/layout.css";
import "./HomePage.css";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <>
      <div className="page-bg-home" />
      <Hero
        variant="home"
        eyebrow="AI-Powered Discovery Engine"
        title="Your next obsession"
        titleAccent="is one search away."
        subtitle="Drop a title you love and FilmOracle surfaces what to watch or play next — powered by TF-IDF cosine similarity across 8,000+ movies and 10,000+ games."
      />

      <div className="feature-cards">
        <div className="feature-card feature-movies">
          <div className="feature-card-icon">
            <Clapperboard size={22} strokeWidth={1.5} />
          </div>
          <div className="feature-card-title">Cinema Picks</div>
          <div className="feature-card-body">
            From arthouse indie to mainstream blockbuster — enter any film and get ten
            recommendations calibrated to its genre, cast, and narrative DNA.
          </div>
          <button className="feature-card-cta" onClick={() => navigate("/movies")}>
            Find Movies <ArrowRight size={13} strokeWidth={2} />
          </button>
        </div>

        <div className="feature-card feature-games">
          <div className="feature-card-icon">
            <Gamepad2 size={22} strokeWidth={1.5} />
          </div>
          <div className="feature-card-title">Game Curation</div>
          <div className="feature-card-body">
            PC, PS5, or Xbox — type any game and we'll match it against tags, genres,
            developer style, and playstyle signals across a decade of titles.
          </div>
          <button className="feature-card-cta" onClick={() => navigate("/games")}>
            Find Games <ArrowRight size={13} strokeWidth={2} />
          </button>
        </div>
      </div>

      <div className="why-section">
        <div className="why-eyebrow">Why FilmOracle</div>
        <div className="why-grid">
          <div className="why-card">
            <div className="why-card-icon">
              <Zap size={16} strokeWidth={1.8} />
            </div>
            <div className="why-card-title">Instant Results</div>
            <div className="why-card-body">
              No accounts. No waitlists. No black-box feeds. Type a title, get answers.
            </div>
          </div>
          <div className="why-card">
            <div className="why-card-icon">
              <SearchCheck size={16} strokeWidth={1.8} />
            </div>
            <div className="why-card-title">Typo-Tolerant Search</div>
            <div className="why-card-body">
              Fuzzy matching handles partial titles, misspellings, and shorthand — ZNMD works just fine.
            </div>
          </div>
          <div className="why-card">
            <div className="why-card-icon">
              <LayoutList size={16} strokeWidth={1.8} />
            </div>
            <div className="why-card-title">Full Context</div>
            <div className="why-card-body">
              Trailers, ratings, cast, streaming links, and store pages — everything you need to decide.
            </div>
          </div>
        </div>

        <div className="home-callout">
          Use the top nav to switch between Movies, Games, and Contact. Pick a section,
          enter a title you already love, and let the engine do the rest.
        </div>
      </div>

      <Footer />
    </>
  );
}
