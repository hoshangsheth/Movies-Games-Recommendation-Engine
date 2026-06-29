.search-wrap {
  margin-bottom: 40px;
}

.search-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--glass-bg-strong);
  border: 1.5px solid var(--glass-border);
  border-radius: var(--radius-pill);
  padding: 6px 6px 6px 20px;
  box-shadow: var(--shadow-card);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.search-bar:focus-within {
  border-color: rgba(168,85,247,0.40);
  box-shadow: 0 0 0 4px rgba(168,85,247,0.08), var(--shadow-card);
}

.search-bar-input-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
}

.search-bar-icon {
  color: var(--text-muted);
  flex-shrink: 0;
}

.search-bar-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  font-family: var(--font-body);
  font-size: 0.95rem;
  color: var(--text-heading);
  padding: 8px 0;
}

.search-bar-input::placeholder {
  color: var(--text-faint);
}

.search-bar-button {
  background: var(--grad-cta);
  color: white;
  font-family: var(--font-body);
  font-size: 0.88rem;
  font-weight: 700;
  padding: 12px 28px;
  border-radius: var(--radius-pill);
  border: none;
  cursor: pointer;
  white-space: nowrap;
  transition: opacity 0.15s, box-shadow 0.15s;
  box-shadow: var(--shadow-btn);
  flex-shrink: 0;
}

.search-bar-button:hover:not(:disabled) {
  opacity: 0.9;
  box-shadow: var(--shadow-btn-hover);
}

.search-bar-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 560px) {
  .search-bar {
    flex-direction: column;
    border-radius: var(--radius-xl);
    padding: 14px 16px;
    gap: 10px;
    align-items: stretch;
  }
  .search-bar-button {
    padding: 12px;
    text-align: center;
  }
}
