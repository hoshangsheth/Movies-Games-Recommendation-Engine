import "./Footer.css";

export default function Footer() {
  return (
    <footer className="app-footer">
      <span>FilmOracle — Powered by ML · content-based filtering & cosine similarity.</span>
      <span>
        Built by{" "}
        <a href="https://hoshangsheth.com" target="_blank" rel="noreferrer">
          Hoshang Sheth
        </a>
      </span>
    </footer>
  );
}
