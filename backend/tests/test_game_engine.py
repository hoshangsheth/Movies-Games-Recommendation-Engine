import pytest

from app.recommender.game_engine import GameRecommendationEngine


def test_recommend_returns_most_similar_first(games_df, games_similarity_matrix):
    engine = GameRecommendationEngine(games_df, games_similarity_matrix)
    results = engine.recommend("Elden Ring", top_n=2)

    assert [r["Title"] for r in results] == ["Dark Souls III", "FIFA 23"]


def test_store_links_are_formatted_as_name_url_pairs(games_df, games_similarity_matrix):
    engine = GameRecommendationEngine(games_df, games_similarity_matrix)
    results = engine.recommend("Elden Ring", top_n=1)

    assert results[0]["Title"] == "Dark Souls III"
    assert results[0]["Stores"] == "Steam : https://store.steampowered.com"


def test_multiple_store_links_formatted_correctly(games_df, games_similarity_matrix):
    engine = GameRecommendationEngine(games_df, games_similarity_matrix)
    results = engine.recommend("Dark Souls III", top_n=1)

    assert results[0]["Title"] == "Elden Ring"
    assert results[0]["Stores"] == (
        "Steam : https://store.steampowered.com, "
        "PlayStation Store : https://store.playstation.com"
    )


def test_recommend_resolves_known_alias(games_df, games_similarity_matrix):
    engine = GameRecommendationEngine(games_df, games_similarity_matrix)
    results = engine.recommend("elden", top_n=1)  # "elden" -> "elden ring" alias
    assert results[0]["Title"] == "Dark Souls III"


def test_recommend_raises_value_error_on_empty_input(games_df, games_similarity_matrix):
    engine = GameRecommendationEngine(games_df, games_similarity_matrix)
    with pytest.raises(ValueError):
        engine.recommend("   ", top_n=2)
