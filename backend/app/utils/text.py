"""
Text-normalization helpers shared by both the movie and game engines.

`clean_title` mirrors the `clean_title()` function from the original data
notebooks (used at data-prep time on the dataset's own titles) and the
inline regex used in `recommend_movies()` / `recommend_games()` (used at
request time on user input). Keeping a single shared implementation avoids
the original's duplication across the two call sites.
"""
from __future__ import annotations

import re
import unicodedata


def clean_title(title: str) -> str:
    """
    Normalize a title for matching purposes: strip accents, lowercase,
    remove punctuation, collapse whitespace.

    Matches `Notebooks/movies_recommendation.ipynb` and
    `Notebooks/games_recommendation.ipynb` `clean_title()` exactly.
    """
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8", "ignore")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def clean_user_input(user_input: str) -> str:
    """
    Normalize raw user search input before alias lookup / fuzzy matching.

    Matches the inline regex used directly inside `recommend_movies()` /
    `recommend_games()` in the original `Deployment/app.py`:
    `re.sub(r'[^a-zA-Z0-9\\s]', '', user_input.lower().strip())`
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", user_input.lower().strip())
