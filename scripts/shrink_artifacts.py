"""
One-time artifact shrink script.

`cosine_similarity()` (scikit-learn) returns float64 by default, and the
original notebooks saved it as-is. Cosine similarity values only need
~7 significant digits of precision for ranking purposes — float32 is
more than enough and HALVES the in-memory + on-disk size of both
similarity matrices, with zero change to recommendation behavior (the
top-N ranking order is unaffected by the precision drop).

Usage:
    python scripts/shrink_artifacts.py

Run this once, locally, after downloading the current artifacts (or
point it at files you already have on disk). It re-saves smaller
versions in place. Re-upload the shrunk files to Google Drive and your
memory usage on Render drops by roughly half with no code changes
elsewhere.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def shrink_npy(path: Path) -> None:
    before = path.stat().st_size
    matrix = np.load(path, allow_pickle=True)
    print(f"{path.name}: loaded {matrix.shape} dtype={matrix.dtype}, {before / 1e6:.1f} MB")

    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)

    np.save(path, matrix)
    after = path.stat().st_size
    print(f"{path.name}: now {after / 1e6:.1f} MB (was {before / 1e6:.1f} MB)")


def shrink_pickled_matrix(path: Path) -> None:
    """For matrices that were pickled instead of saved with np.save (e.g. cosine_sim_movies.pkl)."""
    before = path.stat().st_size
    with open(path, "rb") as f:
        matrix = pickle.load(f)

    if not isinstance(matrix, np.ndarray):
        print(f"{path.name}: not a numpy array (type={type(matrix)}), skipping")
        return

    print(f"{path.name}: loaded {matrix.shape} dtype={matrix.dtype}, {before / 1e6:.1f} MB")

    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)

    with open(path, "wb") as f:
        pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    after = path.stat().st_size
    print(f"{path.name}: now {after / 1e6:.1f} MB (was {before / 1e6:.1f} MB)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python shrink_artifacts.py <path-to-artifacts-dir>")
        print("Expects: movies_recommended.pkl, cosine_sim_movies.pkl, games_recommended.pkl, cosine_sim_games.npy")
        sys.exit(1)

    artifacts_dir = Path(sys.argv[1])

    npy_path = artifacts_dir / "cosine_sim_games.npy"
    if npy_path.exists():
        shrink_npy(npy_path)
    else:
        print(f"Not found, skipping: {npy_path}")

    pkl_path = artifacts_dir / "cosine_sim_movies.pkl"
    if pkl_path.exists():
        shrink_pickled_matrix(pkl_path)
    else:
        print(f"Not found, skipping: {pkl_path}")

    print("\nDone. Re-upload the shrunk files to Google Drive and update the file IDs if they changed.")


if __name__ == "__main__":
    main()
