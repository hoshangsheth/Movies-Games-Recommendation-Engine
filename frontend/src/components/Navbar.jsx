import { NavLink } from "react-router-dom";
import { Sparkles, Sun, Moon, Home, Film, Gamepad2, Mail } from "lucide-react";
import "./Navbar.css";

export default function Navbar({ isDark, onToggleTheme }) {
  return (
    <>
      {/* ── Desktop topbar ── */}
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="topbar-brand">
            <div className="topbar-brand-icon">
              <Sparkles size={14} strokeWidth={2.5} />
            </div>
            <span className="topbar-brand-title">FilmOracle</span>
          </NavLink>

          <nav className="topbar-nav">
            {[
              { to: "/", label: "Home", end: true },
              { to: "/movies", label: "Movies" },
              { to: "/games", label: "Games" },
              { to: "/contact", label: "Contact" },
            ].map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `topbar-link${isActive ? " active" : ""}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="topbar-right">
            <button
              className="topbar-theme-toggle"
              onClick={onToggleTheme}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark ? <Sun size={17} strokeWidth={2} /> : <Moon size={17} strokeWidth={2} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Mobile floating bottom nav ── */}
      <nav className="mobile-bottom-nav" aria-label="Mobile navigation">
        <div className="mobile-bottom-nav-pill">

          {/* Slot 1 — Theme toggle */}
          <button
            className="mbn-item mbn-theme"
            onClick={onToggleTheme}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? <Sun size={22} strokeWidth={1.8} /> : <Moon size={22} strokeWidth={1.8} />}
            <span className="mbn-label">{isDark ? "Light" : "Dark"}</span>
          </button>

          {/* Slot 2 — Movies */}
          <NavLink
            to="/movies"
            className={({ isActive }) => `mbn-item${isActive ? " active" : ""}`}
          >
            <Film size={22} strokeWidth={1.8} />
            <span className="mbn-label">Movies</span>
          </NavLink>

          {/* Slot 3 — Center: Home with gradient */}
          <NavLink to="/" end className="mbn-center" aria-label="Home">
            <Home size={20} strokeWidth={2} />
          </NavLink>

          {/* Slot 4 — Games */}
          <NavLink
            to="/games"
            className={({ isActive }) => `mbn-item${isActive ? " active" : ""}`}
          >
            <Gamepad2 size={22} strokeWidth={1.8} />
            <span className="mbn-label">Games</span>
          </NavLink>

          {/* Slot 5 — Contact */}
          <NavLink
            to="/contact"
            className={({ isActive }) => `mbn-item${isActive ? " active" : ""}`}
          >
            <Mail size={22} strokeWidth={1.8} />
            <span className="mbn-label">Contact</span>
          </NavLink>

        </div>
      </nav>
    </>
  );
}
