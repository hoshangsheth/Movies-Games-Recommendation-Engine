import { useState } from "react";
import Hero from "../components/Hero";
import SearchBar from "../components/SearchBar";
import RecommendationGrid from "../components/RecommendationGrid";
import LoadingSkeleton from "../components/LoadingSkeleton";
import GameDetailModal from "../components/GameDetailModal";
import Footer from "../components/Footer";
import useRecommendations from "../hooks/useRecommendations";
import { fetchGameRecommendations } from "../services/api";
import "../styles/layout.css";

export default function GamesPage() {
  const { results, error, loading, search } = useRecommendations(fetchGameRecommendations);
  const [selectedGame, setSelectedGame] = useState(null);

  return (
    <>
      <div className="page-bg-games" />
      <Hero
        eyebrow="Game Discovery"
        title="Ready. Set. Play."
        subtitle="Drop a game title you love and we'll find 10 titles across PC, PS5, and Xbox that match your playstyle."
      />

      <SearchBar
        label="Game Title"
        placeholder="e.g. Elden Ring, God of War, Spider-Man, Need for Speed..."
        buttonLabel="→  Find Similar Games"
        onSearch={search}
        loading={loading}
      />

      {error && <div className="error-banner">{error}</div>}

      {loading && <LoadingSkeleton count={8} />}

      {!loading && results && (
        <RecommendationGrid items={results} posterVariant="game" onSelect={setSelectedGame} />
      )}

      {selectedGame && <GameDetailModal game={selectedGame} onClose={() => setSelectedGame(null)} />}

      <Footer />
    </>
  );
}
