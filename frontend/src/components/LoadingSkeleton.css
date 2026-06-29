.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 20px;
  margin-bottom: 48px;
}

@media (max-width: 680px) {
  .skeleton-grid {
    display: flex;
    overflow-x: auto;
    gap: 14px;
  }
}

.skeleton-card {
  flex-shrink: 0;
}

.skeleton-poster {
  aspect-ratio: 2/3;
  border-radius: var(--radius-lg);
  margin-bottom: 10px;
  background: linear-gradient(
    100deg,
    rgba(168,85,247,0.06) 30%,
    rgba(249,115,22,0.10) 50%,
    rgba(168,85,247,0.06) 70%
  );
  background-size: 200% 100%;
  animation: shimmer 1.6s ease-in-out infinite;
}

.skeleton-title {
  height: 16px;
  border-radius: var(--radius-sm);
  background: rgba(168,85,247,0.08);
  margin-bottom: 6px;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

@media (prefers-reduced-motion: reduce) {
  .skeleton-poster { animation: none; }
}
