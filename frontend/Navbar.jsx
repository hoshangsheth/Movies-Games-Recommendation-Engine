import { useState } from "react";
import { Search } from "lucide-react";
import "./SearchBar.css";

export default function SearchBar({ label, placeholder, buttonLabel, onSearch, loading }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  }

  return (
    <div className="search-wrap">
      <form className="search-bar" onSubmit={handleSubmit}>
        <div className="search-bar-input-wrap">
          <Search className="search-bar-icon" size={20} strokeWidth={2} />
          <input
            id="search-input"
            className="search-bar-input"
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={placeholder}
            aria-label={label}
          />
        </div>
        <button className="search-bar-button" type="submit" disabled={loading}>
          {loading ? "Searching…" : (buttonLabel || "Search")}
        </button>
      </form>
    </div>
  );
}
