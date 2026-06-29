import { NavLink } from "react-router-dom";
import { Sparkles, Sun, Moon, Home, Film, Gamepad2, Mail } from "lucide-react";
import "./Navbar.css";

const NAV_LEFT = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/movies", label: "Movies", icon: Film },
];

const NAV_RIGHT = [
  { to: "/games", label: "Games", icon: Gamepad2 },
  { to: "/contact", label: "Contact", icon: Mail },
];

export default function Navbar({ isDark, onToggleTheme }) {
  return (
    <>
      {/* ── Desktop topbar (unchanged) ── */}
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="topbar-brand">
            <div className="topbar-brand-icon">
              <Sparkles size={14} strokeWidth={2.5} />
            </div>
            <span className="topbar-brand-title">FilmOracle</span>
          </NavLink>

          <nav className="topbar-nav">
            {[...NAV_LEFT, ...NAV_RIGHT].map((item) => (
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
          {NAV_LEFT.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `mbn-item${isActive ? " active" : ""}`
              }
            >
              <item.icon size={22} strokeWidth={1.8} />
              <span className="mbn-label">{item.label}</span>
            </NavLink>
          ))}

          {/* Center accent — theme toggle */}
          <button
            className="mbn-center"
            onClick={onToggleTheme}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? <Sun size={20} strokeWidth={2} /> : <Moon size={20} strokeWidth={2} />}
          </button>

          {NAV_RIGHT.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `mbn-item${isActive ? " active" : ""}`
              }
            >
              <item.icon size={22} strokeWidth={1.8} />
              <span className="mbn-label">{item.label}</span>
            </NavLink>
          ))}
        </div>
      </nav>
    </>
  );
}
