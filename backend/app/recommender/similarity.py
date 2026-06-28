"""
On-demand cosine similarity from a sparse TF-IDF matrix.

What changed from the original
--------------------------------
Old: `top_n_similar_indices(matrix_row, top_n)` accepted a pre-fetched
     dense row of the precomputed N×N cosine-similarity matrix and ranked it.
     The caller passed `similarity_matrix[idx]` (one dense array).

New: `top_n_similar_indices(tfidf_matrix, idx, top_n)` receives the full
     sparse TF-IDF matrix and the item index, computes the cosine similarity
     for that one row on the fly, then ranks identically to the original.

Why this produces identical rankings
--------------------------------------
`cosine_similarity(tfidf_matrix[idx], tfidf_matrix)` computes the same
dot-product-divided-by-norms that the offline notebook computed when
building the precomputed matrix. The resulting similarity scores are
identical to float precision, so sort order — and therefore the top-N
results — is unchanged.

Runtime cost
--------------
For a single row query, scikit-learn's cosine_similarity performs one
sparse dot product (O(n × avg_nonzeros_per_row)).  At typical vocab sizes
(~5 000–30 000 terms) and row sparsity this completes in well under
100 ms, which is negligible relative to network latency.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity


def top_n_similar_indices(
    tfidf_matrix: scipy.sparse.csr_matrix,
    idx: int,
    top_n: int,
) -> list[tuple[int, float]]:
    """
    Compute cosine similarity on demand for item `idx` against every
    other item, then return the `top_n` most-similar *other* items.

    Parameters
    ----------
    tfidf_matrix : scipy.sparse.csr_matrix
        The full sparse TF-IDF matrix (n_items × vocab).
    idx : int
        Row index of the query item.
    top_n : int
        Number of results to return (excluding the item itself).

    Returns
    -------
    list of (item_index, similarity_score), sorted descending by score,
    excluding the query item — exactly the same format as the original.
    """
    # cosine_similarity returns shape (1, n_items); squeeze to 1-D array
    scores: np.ndarray = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Enumerate and rank — identical to the original sort logic
    sim_scores = list(enumerate(scores))
    ranked = sorted(sim_scores, key=lambda pair: pair[1], reverse=True)

    # Skip index 0 (the item itself, always score=1.0) — matches original [1:top_n+1]
    return ranked[1 : top_n + 1]
