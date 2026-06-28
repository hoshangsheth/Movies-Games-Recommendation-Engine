"""Shared pytest fixtures: small synthetic dataframes + similarity matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def movies_df() -> pd.DataFrame:
    data = [
        {
            "title": "Inception",
            "title_clean": "inception",
            "top_cast": "Leonardo DiCaprio, Joseph Gordon-Levitt",
            "cast_profile_path": "http://x/a.jpg,http://x/b.jpg",
            "description": "A thief who steals corporate secrets through dream-sharing.",
            "genres": "Action, Sci-Fi",
            "languages": "English",
            "rating": 8.8,
            "poster_path": "http://x/inception.jpg",
            "release_date": "2010-07-16",
            "watch_link": "http://watch/inception",
            "video_key": "YoHD9XEInc0",
        },
        {
            "title": "Interstellar",
            "title_clean": "interstellar",
            "top_cast": "Matthew McConaughey, Anne Hathaway",
            "cast_profile_path": "http://x/c.jpg,http://x/d.jpg",
            "description": "A team travels through a wormhole in space.",
            "genres": "Adventure, Sci-Fi",
            "languages": "English",
            "rating": 8.6,
            "poster_path": "http://x/interstellar.jpg",
            "release_date": "2014-11-07",
            "watch_link": "http://watch/interstellar",
            "video_key": None,
        },
        {
            "title": "The Dark Knight",
            "title_clean": "the dark knight",
            "top_cast": "Christian Bale, Heath Ledger",
            "cast_profile_path": "http://x/e.jpg,http://x/f.jpg",
            "description": "Batman faces the Joker.",
            "genres": "Action, Crime",
            "languages": "English",
            "rating": 9.0,
            "poster_path": "http://x/tdk.jpg",
            "release_date": "2008-07-18",
            "watch_link": "http://watch/tdk",
            "video_key": None,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def movies_similarity_matrix() -> np.ndarray:
    # Inception(0) most similar to Interstellar(1); Dark Knight(2) least similar to both
    return np.array(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )


@pytest.fixture
def games_df() -> pd.DataFrame:
    data = [
        {
            "title": "Elden Ring",
            "title_clean": "elden ring",
            "description_clean": "An action RPG set in a vast fantasy world.",
            "genres": "RPG, Action",
            "release_date": "2022-02-25",
            "rating": 4.8,
            "platforms": "PC, PlayStation 5, Xbox Series X",
            "tags": "Souls-like, Open World",
            "developers": "FromSoftware",
            "publishers": "Bandai Namco",
            "esrb_rating": "Mature",
            "background_image_url": "http://x/elden.jpg",
            "website": "http://eldenring.com",
            "screenshots": "http://x/s1.jpg,http://x/s2.jpg",
            "store_name": "Steam, PlayStation Store",
            "store_domain": "store.steampowered.com, store.playstation.com",
        },
        {
            "title": "Dark Souls III",
            "title_clean": "dark souls iii",
            "description_clean": "A grim, atmospheric action RPG.",
            "genres": "RPG, Action",
            "release_date": "2016-04-12",
            "rating": 4.5,
            "platforms": "PC, PlayStation 4",
            "tags": "Souls-like, Difficult",
            "developers": "FromSoftware",
            "publishers": "Bandai Namco",
            "esrb_rating": "Mature",
            "background_image_url": "http://x/ds3.jpg",
            "website": "http://darksouls.com",
            "screenshots": "http://x/s3.jpg",
            "store_name": "Steam",
            "store_domain": "store.steampowered.com",
        },
        {
            "title": "FIFA 23",
            "title_clean": "fifa 23",
            "description_clean": "A football simulation game.",
            "genres": "Sports",
            "release_date": "2022-09-30",
            "rating": 4.0,
            "platforms": "PC, PlayStation 5",
            "tags": "Sports, Multiplayer",
            "developers": "EA Vancouver",
            "publishers": "EA Sports",
            "esrb_rating": "Everyone",
            "background_image_url": "http://x/fifa.jpg",
            "website": "http://ea.com/fifa",
            "screenshots": "http://x/s4.jpg",
            "store_name": "Origin",
            "store_domain": "origin.com",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def games_similarity_matrix() -> np.ndarray:
    # Elden Ring(0) most similar to Dark Souls III(1); FIFA(2) unrelated to both
    return np.array(
        [
            [1.0, 0.9, 0.05],
            [0.9, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ]
    )
