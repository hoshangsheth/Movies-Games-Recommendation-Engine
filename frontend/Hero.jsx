/* ── Topbar shell (floating glass) ── */
.topbar {
  position: fixed;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  width: calc(100% - 48px);
  max-width: 1200px;
  z-index: 100;
  background: var(--glass-bg-strong);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-navbar);
}

.topbar-inner {
  display: flex;
  align-items: center;
  height: 56px;
  padding: 0 20px;
  gap: 32px;
}

/* ── Brand ── */
.topbar-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  flex-shrink: 0;
}

.topbar-brand-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 10px;
  background: var(--grad-cta);
  color: white;
  flex-shrink: 0;
}

.topbar-brand-title {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--text-heading);
  letter-spacing: -0.02em;
}

/* ── Desktop nav ── */
.topbar-nav {
  display: flex;
  align-items: center;
  gap: 2px;
  flex: 1;
}

.topbar-link {
  color: var(--text-muted);
  font-family: var(--font-body);
  font-size: 0.875rem;
  font-weight: 500;
  text-decoration: none;
  padding: 6px 16px;
  border-radius: var(--radius-pill);
  border: 1px solid transparent;
  transition: all 0.18s ease;
}

.topbar-link:hover {
  color: var(--text-heading);
  background: rgba(168,85,247,0.08);
}

.topbar-link.active {
  color: var(--accent-purple);
  background: rgba(168,85,247,0.10);
  border-color: rgba(168,85,247,0.20);
  font-weight: 600;
}

/* ── Right side ── */
.topbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  margin-left: auto;
}

.topbar-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(0,0,0,0.05);
  transition: background 0.15s;
  text-decoration: none;
}

.topbar-icon-btn:hover { background: rgba(0,0,0,0.1); }

.topbar-cta-btn {
  background: var(--grad-cta);
  color: white;
  font-family: var(--font-body);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 7px 18px;
  border-radius: var(--radius-pill);
  text-decoration: none;
  transition: opacity 0.15s, box-shadow 0.15s;
  white-space: nowrap;
}

.topbar-cta-btn:hover {
  opacity: 0.9;
  box-shadow: var(--shadow-btn);
}

/* ── Hamburger ── */
.topbar-hamburger {
  display: none;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.06);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  padding: 6px;
  color: var(--text-heading);
  margin-left: auto;
  flex-shrink: 0;
}

/* ── Mobile dropdown ── */
.topbar-mobile-menu {
  display: flex;
  flex-direction: column;
  padding: 8px 12px 12px;
  border-top: 1px solid rgba(168,85,247,0.1);
  gap: 2px;
}

.topbar-mobile-link {
  color: var(--text-muted);
  font-family: var(--font-body);
  font-size: 0.9rem;
  font-weight: 500;
  text-decoration: none;
  padding: 10px 16px;
  border-radius: var(--radius-md);
  transition: all 0.15s ease;
}

.topbar-mobile-link:hover,
.topbar-mobile-link.active {
  color: var(--accent-purple);
  background: rgba(168,85,247,0.08);
}

/* ── Responsive ── */
@media (max-width: 680px) {
  .topbar {
    top: 8px;
    width: calc(100% - 24px);
  }

  .topbar-nav,
  .topbar-right {
    display: none;
  }

  .topbar-hamburger {
    display: flex;
  }
}
