import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Sparkles, Menu, X, Sun, Moon } from "lucide-react";
import "./Navbar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", end: true },
  { to: "/movies", label: "Movies" },
  { to: "/games", label: "Games" },
  { to: "/contact", label: "Contact" },
];

export default function Navbar({ isDark, onToggleTheme }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="topbar">
      <div className="topbar-inner">
        {/* Brand */}
        <NavLink to="/" className="topbar-brand">
          <div className="topbar-brand-icon">
            <Sparkles size={14} strokeWidth={2.5} />
          </div>
          <span className="topbar-brand-title">FilmOracle</span>
        </NavLink>

        {/* Desktop nav */}
        <nav className="topbar-nav">
          {NAV_ITEMS.map((item) => (
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

        {/* Desktop right — dark mode toggle only */}
        <div className="topbar-right">
          <button
            className="topbar-theme-toggle"
            onClick={onToggleTheme}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? <Sun size={17} strokeWidth={2} /> : <Moon size={17} strokeWidth={2} />}
          </button>
        </div>

        {/* Mobile hamburger */}
        <button
          className="topbar-hamburger"
          onClick={() => setMenuOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          {menuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <nav className="topbar-mobile-menu">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `topbar-mobile-link${isActive ? " active" : ""}`
              }
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </NavLink>
          ))}
          {/* Dark mode toggle in mobile menu too */}
          <button
            className="topbar-mobile-theme-toggle"
            onClick={onToggleTheme}
          >
            {isDark ? <Sun size={15} strokeWidth={2} /> : <Moon size={15} strokeWidth={2} />}
            {isDark ? "Light Mode" : "Dark Mode"}
          </button>
        </nav>
      )}
    </header>
  );
}
