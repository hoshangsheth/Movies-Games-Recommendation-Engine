"""
Cosine-similarity lookup against a precomputed matrix.

The original `recommend_movies()` and `recommend_games()` both did the
exact same thing once a best-match row index was found:

    sim_scores = list(enumerate(matrix[idx]))
    top = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

This is that step, extracted once. No vectorization happens here — the
TF-IDF + cosine_similarity computation is done offline in
`backend/notebooks/`, and this module only ever reads the precomputed
matrix that results from it.
"""
from __future__ import annotations

import numpy as np


def top_n_similar_indices(matrix_row: np.ndarray, top_n: int) -> list[tuple[int, float]]:
    """
    Given one row of a precomputed similarity matrix, return the indices
    and scores of the `top_n` most similar *other* items (the item itself,
    always the highest score at its own index, is excluded — matches the
    original's `[1:top_n+1]` slice).
    """
    sim_scores = list(enumerate(matrix_row))
    ranked = sorted(sim_scores, key=lambda pair: pair[1], reverse=True)
    return ranked[1 : top_n + 1]
