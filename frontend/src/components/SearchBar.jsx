import { useState } from "react";
import "./SearchBar.css";

/**
 * Title input + "Find Similar X" button. Replaces the original's
 * `st.text_input` + `st.button` pair that toggled
 * `st.session_state.recommend_triggered`.
 */
export default function SearchBar({ label, placeholder, buttonLabel, onSearch, loading }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (value.trim()) {
      onSearch(value.trim());
    }
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <label className="search-bar-label" htmlFor="search-input">
        {label}
      </label>
      <input
        id="search-input"
        className="search-bar-input"
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
      />
      <button className="search-bar-button" type="submit" disabled={loading}>
        {loading ? "Searching…" : buttonLabel}
      </button>
    </form>
  );
}
