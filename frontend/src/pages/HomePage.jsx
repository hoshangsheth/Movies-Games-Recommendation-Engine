import { Clapperboard, Gamepad2, Brain, Shield, Zap, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import "../styles/layout.css";
import "./HomePage.css";

// const MOODS = [
//   { emoji: "😄", label: "Happy", sub: "Feel-good picks" },
//   { emoji: "🚀", label: "Excited", sub: "High energy fun", active: true },
//   { emoji: "🍿", label: "Relaxed", sub: "Sit back & unwind" },
//   { emoji: "🌧️", label: "Sad", sub: "Emotional stories" },
//   { emoji: "🔮", label: "Curious", sub: "Mind-bending picks" },
//   { emoji: "🎃", label: "Scared", sub: "Thrilling & spooky" },
// ];

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <>
      <div className="page-bg-home" />

      {/* Hero */}
      <Hero
        variant="home"
        eyebrow="ML-Powered Discovery Engine"
        title="Find your next"
        titleAccent="obsession."
        subtitle="Get ML-powered recommendations for movies and games tailored to your taste. Discover hidden gems, trending titles, and all-time classics in one place."
      />

      {/* CTA buttons */}
      <div className="home-cta-group">
        <button className="home-btn-primary" onClick={() => navigate("/movies")}>
          Explore Cinema <ArrowRight size={16} strokeWidth={2.5} />
        </button>
        <button className="home-btn-primary" onClick={() => navigate("/games")}>
          Ready to Play <ArrowRight size={16} strokeWidth={2.5} />
        </button>
      </div>

      {/* Mood section */}
      {/* <section className="mood-section">
        <div className="mood-header">
          <div className="hero-eyebrow" style={{ display:"inline-flex", marginBottom:"12px" }}>
            ✨ &nbsp; AI-Powered Recommendations
          </div>
          <h2 className="mood-title">
            How are you <span className="mood-title-accent">feeling?</span>
          </h2>
          <p className="mood-subtitle">
            Tell us your mood and we'll find the perfect movies and games for you.
          </p>
        </div>

        <div className="mood-cards-scroll">
          {MOODS.map((mood) => (
            <div
              key={mood.label}
              className={`mood-card${mood.active ? " mood-card-active" : ""}`}
            >
              <div className="mood-card-emoji">{mood.emoji}</div>
              <div className={`mood-card-label${mood.active ? " mood-card-label-active" : ""}`}>
                {mood.label}
              </div>
              <div className="mood-card-sub">{mood.sub}</div>
            </div>
          ))}
        </div>

        <div className="mood-actions">
          <button className="home-btn-primary" onClick={() => navigate("/movies")}>
            Get Recommendations ✦
          </button>
          <button className="home-btn-ghost">
            🎲 Surprise Me
          </button>
        </div>
      </section> */}

      {/* Feature cards */}
      <section className="feature-cards">
        <div className="feature-card">
          <div className="feature-card-icon feature-icon-brain">🧠</div>
          <div className="feature-card-title">AI That Gets You</div>
          <div className="feature-card-body">
            Our AI learns your taste and mood to deliver spot-on recommendations.
          </div>
        </div>
        <div className="feature-card">
          <div className="feature-card-icon feature-icon-shield">🛡️</div>
          <div className="feature-card-title">Personalized For You</div>
          <div className="feature-card-body">
            Tailored picks across movies, games, and shows you'll love.
          </div>
        </div>
        <div className="feature-card">
          <div className="feature-card-icon feature-icon-zap">⚡</div>
          <div className="feature-card-title">Discover More</div>
          <div className="feature-card-body">
            Find hidden gems, trending hits, and all-time classics.
          </div>
        </div>
      </section>

      {/* Category launch cards */}
      <section className="launch-cards">
        <div className="launch-card launch-movies" onClick={() => navigate("/movies")}>
          <Clapperboard size={24} strokeWidth={1.5} />
          <div className="launch-card-title">Movies</div>
          <div className="launch-card-body">
            8,000+ titles across Bollywood, Hollywood, and beyond
          </div>
          <span className="launch-card-link">Explore <ArrowRight size={13} /></span>
        </div>
        <div className="launch-card launch-games" onClick={() => navigate("/games")}>
          <Gamepad2 size={24} strokeWidth={1.5} />
          <div className="launch-card-title">Games</div>
          <div className="launch-card-body">
            10,000+ games matched by genre, tags, and playstyle
          </div>
          <span className="launch-card-link">Explore <ArrowRight size={13} /></span>
        </div>
      </section>

      <Footer />
    </>
  );
}
