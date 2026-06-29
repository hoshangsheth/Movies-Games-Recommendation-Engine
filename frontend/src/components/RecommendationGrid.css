/* ── Grid layout ── */
.recommendation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 20px;
  margin-bottom: 48px;
}

@media (max-width: 680px) {
  .recommendation-grid {
    display: flex;
    overflow-x: auto;
    gap: 14px;
    padding-bottom: 12px;
    scrollbar-width: none;
  }
  .recommendation-grid::-webkit-scrollbar { display: none; }
}

/* ── Card ── */
.rec-card {
  cursor: pointer;
  transition: transform 0.2s ease;
  outline: none;
}

.rec-card:hover { transform: translateY(-4px); }
.rec-card:focus-visible { outline: 2px solid var(--accent-purple); border-radius: var(--radius-lg); }

/* ── Poster wrap ── */
.rec-card-poster-wrap {
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  aspect-ratio: 2/3;
  background: rgba(168,85,247,0.06);
  box-shadow: var(--shadow-card);
  transition: box-shadow 0.2s;
}

.rec-card:hover .rec-card-poster-wrap {
  box-shadow: var(--shadow-card-hover);
}

.rec-card-poster {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* ── Rating badge ── */
.rec-card-rating {
  position: absolute;
  top: 10px;
  left: 10px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(8px);
  color: #fbbf24;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
}

/* ── Add button ── */
.rec-card-add {
  position: absolute;
  bottom: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(8px);
  color: var(--text-heading);
  opacity: 0;
  transform: scale(0.85);
  transition: opacity 0.18s, transform 0.18s;
}

.rec-card:hover .rec-card-add {
  opacity: 1;
  transform: scale(1);
}

/* ── Info ── */
.rec-card-info {
  padding: 10px 2px 4px;
}

.rec-card-title {
  font-family: var(--font-body);
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-heading);
  line-height: 1.3;
  margin-bottom: 4px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.rec-card-meta {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.rec-card-dot { opacity: 0.4; }

/* Mobile card sizing */
@media (max-width: 680px) {
  .rec-card {
    flex-shrink: 0;
    width: 130px;
  }
}

/* ── Old class kept for compatibility ── */
.recommendation-card { display: none; }
.recommendation-card-poster-wrap { display: none; }
