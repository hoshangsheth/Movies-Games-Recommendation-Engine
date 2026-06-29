import { useState } from "react";
import { NavLink } from "react-router-dom";
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
          <span className="topbar-brand-dot" />
          <span className="topbar-brand-title">FilmOracle</span>
        </NavLink>

        {/* Desktop nav links */}
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

        {/* Social links */}
        <div className="topbar-social">
          <a
            href="https://www.linkedin.com/in/hoshangsheth"
            target="_blank"
            rel="noreferrer"
            aria-label="LinkedIn"
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
              alt="LinkedIn"
            />
          </a>
          <a
            href="https://github.com/hoshangsheth"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/733/733553.png"
              alt="GitHub"
              style={{ filter: "invert(1)" }}
            />
          </a>
          <a
            href="https://hoshangsheth.com"
            target="_blank"
            rel="noreferrer"
            aria-label="Portfolio"
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
              alt="Portfolio"
            />
          </a>
        </div>

        {/* Mobile hamburger */}
        <button
          className={`topbar-hamburger${menuOpen ? " open" : ""}`}
          onClick={() => setMenuOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          <span />
          <span />
          <span />
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
