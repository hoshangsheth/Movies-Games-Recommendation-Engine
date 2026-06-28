import numpy as np

from app.recommender.similarity import top_n_similar_indices


def test_excludes_self_and_returns_top_n_sorted_desc():
    row = np.array([1.0, 0.5, 0.9, 0.2, 0.7])
    result = top_n_similar_indices(row, top_n=3)
    assert [idx for idx, _ in result] == [2, 4, 1]


def test_top_n_larger_than_available_returns_all_remaining():
    row = np.array([1.0, 0.3])
    result = top_n_similar_indices(row, top_n=10)
    assert len(result) == 1
    assert result[0][0] == 1
