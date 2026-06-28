import pytest

from app.recommender.movie_engine import MovieRecommendationEngine


def test_recommend_returns_most_similar_first(movies_df, movies_similarity_matrix):
    engine = MovieRecommendationEngine(movies_df, movies_similarity_matrix)
    results = engine.recommend("Inception", top_n=2)

    assert [r["Title"] for r in results] == ["Interstellar", "The Dark Knight"]


def test_recommend_resolves_known_alias():
    # "lotr" is aliased to "The Lord of the Rings"; here we just check the
    # alias dict resolves before fuzzy matching is attempted, using the
    # public engine surface against a minimal frame.
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        [
            {
                "title": "The Lord of the Rings",
                "title_clean": "the lord of the rings",
                "top_cast": "Elijah Wood",
                "cast_profile_path": "http://x/a.jpg",
                "description": "A hobbit's journey.",
                "genres": "Fantasy",
                "languages": "English",
                "rating": 9.0,
                "poster_path": "http://x/lotr.jpg",
                "release_date": "2001-12-19",
                "watch_link": "http://watch/lotr",
                "video_key": None,
            },
            {
                "title": "Unrelated Movie",
                "title_clean": "unrelated movie",
                "top_cast": "Nobody",
                "cast_profile_path": "http://x/b.jpg",
                "description": "Something else entirely.",
                "genres": "Drama",
                "languages": "English",
                "rating": 5.0,
                "poster_path": "http://x/unrelated.jpg",
                "release_date": "1999-01-01",
                "watch_link": "http://watch/unrelated",
                "video_key": None,
            },
        ]
    )
    matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
    engine = MovieRecommendationEngine(df, matrix)
    results = engine.recommend("lotr", top_n=1)
    assert results[0]["Title"] == "Unrelated Movie"  # the only "other" movie


def test_recommend_raises_value_error_on_empty_input(movies_df, movies_similarity_matrix):
    engine = MovieRecommendationEngine(movies_df, movies_similarity_matrix)
    with pytest.raises(ValueError):
        engine.recommend("", top_n=2)


def test_trailer_url_built_from_video_key_when_present(movies_df, movies_similarity_matrix):
    engine = MovieRecommendationEngine(movies_df, movies_similarity_matrix)
    results = engine.recommend("Interstellar", top_n=1)
    # Most similar to Interstellar is Inception, which has a video_key set
    assert results[0]["Title"] == "Inception"
    assert results[0]["Trailer"] == "https://www.youtube.com/watch?v=YoHD9XEInc0"


def test_trailer_is_none_when_video_key_missing(movies_df, movies_similarity_matrix):
    engine = MovieRecommendationEngine(movies_df, movies_similarity_matrix)
    results = engine.recommend("Inception", top_n=1)
    assert results[0]["Title"] == "Interstellar"
    assert results[0]["Trailer"] is None
