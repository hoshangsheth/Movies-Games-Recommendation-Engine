import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Sparkles, Film, Gamepad2, Phone, Home, Menu, X } from "lucide-react";
import "./Navbar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", end: true },
  { to: "/movies", label: "Movies" },
  { to: "/games", label: "Games" },
  { to: "/contact", label: "Contact" },
];

export default function Navbar() {
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

        {/* Desktop right */}
        <div className="topbar-right">
          <a
            href="https://github.com/hoshangsheth"
            target="_blank"
            rel="noreferrer"
            className="topbar-icon-btn"
            aria-label="GitHub"
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/733/733553.png"
              alt="GitHub"
              style={{ width: 18, height: 18, filter: "opacity(0.6)" }}
            />
          </a>
          <a
            href="https://www.hoshangsheth.com"
            target="_blank"
            rel="noreferrer"
            className="topbar-cta-btn"
          >
            Portfolio
          </a>
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
        </nav>
      )}
    </header>
  );
}
