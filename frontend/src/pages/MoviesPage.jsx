import { useState } from "react";
import Hero from "../components/Hero";
import SearchBar from "../components/SearchBar";
import RecommendationGrid from "../components/RecommendationGrid";
import LoadingSkeleton from "../components/LoadingSkeleton";
import MovieDetailModal from "../components/MovieDetailModal";
import Footer from "../components/Footer";
import useRecommendations from "../hooks/useRecommendations";
import { fetchMovieRecommendations } from "../services/api";
import "../styles/layout.css";

export default function MoviesPage() {
  const { results, error, loading, search } = useRecommendations(fetchMovieRecommendations);
  const [selectedMovie, setSelectedMovie] = useState(null);

  return (
    <>
      <div className="page-bg-movies" />
      <Hero
        eyebrow="Cinema Discovery"
        title="Lights. Camera. Discover."
        subtitle="Enter a film you love and we'll surface 10 titles you'll want to watch next — across Bollywood, Hollywood, and beyond."
      />

      <SearchBar
        label="Movie Title"
        placeholder="e.g. Interstellar, Zindagi Na Milegi Dobara, RRR, Inception..."
        buttonLabel="→  Find Similar Movies"
        onSearch={search}
        loading={loading}
      />

      {error && <div className="error-banner">{error}</div>}

      {loading && <LoadingSkeleton count={8} />}

      {!loading && results && (
        <RecommendationGrid items={results} posterVariant="movie" onSelect={setSelectedMovie} />
      )}

      {selectedMovie && (
        <MovieDetailModal movie={selectedMovie} onClose={() => setSelectedMovie(null)} />
      )}

      <Footer />
    </>
  );
}
