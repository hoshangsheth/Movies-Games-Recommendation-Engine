import { NavLink } from "react-router-dom";
import "./Navbar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: "🏠" },
  { to: "/movies", label: "Recommend Movies", icon: "🎬" },
  { to: "/games", label: "Recommend Games", icon: "🎮" },
  { to: "/contact", label: "Contact Me", icon: "✉️" },
];

export default function Navbar() {
  return (
    <nav className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-row">
          <span className="sidebar-brand-dot" />
          <span className="sidebar-brand-title">CineVerse</span>
        </div>
        <div className="sidebar-brand-sub">Movie &amp; Game Discovery</div>
      </div>

      <div className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
          >
            <span className="sidebar-link-icon">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </div>

      <div className="sidebar-footer">
        <p className="sidebar-footer-label">Connect</p>
        <div className="sidebar-footer-links">
          <a href="https://www.linkedin.com/in/hoshangsheth" target="_blank" rel="noreferrer">
            <img
              src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
              alt="LinkedIn"
              style={{ filter: "grayscale(40%)" }}
            />
          </a>
          <a href="https://github.com/hoshangsheth" target="_blank" rel="noreferrer">
            <img
              src="https://cdn-icons-png.flaticon.com/512/733/733553.png"
              alt="GitHub"
              style={{ filter: "grayscale(40%) invert(1)" }}
            />
          </a>
          <a href="https://hoshangsheth.com" target="_blank" rel="noreferrer">
            <img
              src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
              alt="Portfolio"
              style={{ filter: "grayscale(40%)" }}
            />
          </a>
        </div>
      </div>
    </nav>
  );
}
