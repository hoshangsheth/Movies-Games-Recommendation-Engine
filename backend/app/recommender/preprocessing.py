"""
Pre-similarity-lookup steps: alias resolution and fuzzy title matching.

This factors out the part of `recommend_movies()` / `recommend_games()`
in the original `Deployment/app.py` that runs *before* the cosine
similarity matrix is touched: cleaning the user's input, swapping in a
known alias if one exists, then fuzzy-matching against the dataset's
`title_clean` column with `rapidfuzz`.

Both engines used this exact sequence; only the alias dictionary and the
candidate title list differ, so it's extracted once here instead of
duplicated.
"""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process

from app.utils.text import clean_user_input


class NoMatchFoundError(ValueError):
    """Raised when rapidfuzz cannot find any match for the cleaned input."""


def resolve_best_title_match(
    user_input: str,
    titles: pd.Series,
    aliases: dict[str, str],
) -> tuple[str, str]:
    """
    Clean the user's input, apply an alias if one matches, then fuzzy-match
    against `titles` (expected to be the dataset's `title_clean` column).

    Returns a tuple of (cleaned_input_used_for_matching, best_match_title).

    Raises:
        ValueError: if `user_input` is empty/blank (matches original behavior).
        NoMatchFoundError: if rapidfuzz returns no match at all.
    """
    if not isinstance(user_input, str) or not user_input.lower().strip():
        raise ValueError("User input must not be empty. Please add a title to get recommendations.")

    cleaned = clean_user_input(user_input)

    if cleaned in aliases:
        cleaned = aliases[cleaned]

    match_result = process.extractOne(cleaned, titles.to_list(), scorer=fuzz.ratio)
    if match_result is None:
        raise NoMatchFoundError(
            f"Title '{user_input}' is not in the data yet. It will be added in a future update."
        )

    best_match = match_result[0]
    return cleaned, best_match
