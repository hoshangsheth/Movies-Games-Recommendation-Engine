"""
One-time artifact shrink script.

`cosine_similarity()` (scikit-learn) returns float64 by default, and the
original notebooks saved it as-is. Cosine similarity values only need
~7 significant digits of precision for ranking purposes -- float32 is
more than enough and HALVES the in-memory + on-disk size of both
similarity matrices, with zero change to recommendation behavior (the
top-N ranking order is unaffected by the precision drop).

For even tighter memory budgets (e.g. fitting under a 512 MB free-tier
ceiling), pass --dtype float16 to halve the size again. float16 still
gives ~3 significant decimal digits, far more than needed to distinguish
relative rank order among similarity scores bounded in [-1, 1] -- but if
you're already at float32 and re-running with float16, this is a second,
independent downcast, not a re-run of the same conversion.

This script also prunes the two dataframe pickles (movies_recommended.pkl,
games_recommended.pkl) down to only the columns actually read at request
time by app/recommender/movie_engine.py and game_engine.py. Columns like
`keywords` exist only because they fed the offline TF-IDF "soup" -- the
similarity matrix already encodes that signal, so the raw column is dead
weight once the matrix is precomputed.

Usage:
    python scripts/shrink_artifacts.py <artifacts_dir> [--dtype float32|float16] [--prune-columns]

Run this once, locally, after downloading the current artifacts (or
point it at files you already have on disk). It re-saves smaller
versions in place. Re-upload the shrunk files to Google Drive and your
memory usage on Render drops accordingly with no code changes elsewhere.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
}

MOVIE_COLUMNS_NEEDED = [
    "title",
    "title_clean",
    "top_cast",
    "cast_profile_path",
    "description",
    "genres",
    "languages",
    "release_date",
    "rating",
    "poster_path",
    "watch_link",
    "video_key",
]

GAME_COLUMNS_NEEDED = [
    "title",
    "title_clean",
    "description_clean",
    "genres",
    "release_date",
    "rating",
    "platforms",
    "store_name",
    "store_domain",
    "tags",
    "developers",
    "publishers",
    "esrb_rating",
    "background_image_url",
    "website",
    "screenshots",
]


def shrink_npy(path: Path, dtype: np.dtype) -> None:
    before = path.stat().st_size
    matrix = np.load(path, allow_pickle=True)
    print(f"{path.name}: loaded {matrix.shape} dtype={matrix.dtype}, {before / 1e6:.1f} MB")

    if matrix.dtype != dtype:
        matrix = matrix.astype(dtype)

    np.save(path, matrix)
    after = path.stat().st_size
    print(f"{path.name}: now {after / 1e6:.1f} MB (was {before / 1e6:.1f} MB)")


def shrink_pickled_matrix(path: Path, dtype: np.dtype) -> None:
    before = path.stat().st_size
    with open(path, "rb") as f:
        matrix = pickle.load(f)

    if not isinstance(matrix, np.ndarray):
        print(f"{path.name}: not a numpy array (type={type(matrix)}), skipping")
        return

    print(f"{path.name}: loaded {matrix.shape} dtype={matrix.dtype}, {before / 1e6:.1f} MB")

    if matrix.dtype != dtype:
        matrix = matrix.astype(dtype)

    with open(path, "wb") as f:
        pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    after = path.stat().st_size
    print(f"{path.name}: now {after / 1e6:.1f} MB (was {before / 1e6:.1f} MB)")


def prune_dataframe(path: Path, columns_needed: list[str]) -> None:
    before = path.stat().st_size
    with open(path, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        print(f"{path.name}: not a DataFrame (type={type(df)}), skipping")
        return

    original_columns = set(df.columns)
    keep = [c for c in columns_needed if c in original_columns]
    dropped = sorted(original_columns - set(keep))

    print(f"{path.name}: {len(original_columns)} columns, {before / 1e6:.1f} MB")
    if dropped:
        print(f"{path.name}: dropping unused columns: {dropped}")
    else:
        print(f"{path.name}: no unused columns found to drop")

    df = df[keep].reset_index(drop=True)

    with open(path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    after = path.stat().st_size
    print(f"{path.name}: now {after / 1e6:.1f} MB (was {before / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrink the precomputed similarity matrices and dataframes.")
    parser.add_argument("artifacts_dir", help="Directory containing the four downloaded artifact files")
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="float32",
        help="Target dtype for the similarity matrices (default: float32). Use float16 for maximum size reduction.",
    )
    parser.add_argument(
        "--prune-columns",
        action="store_true",
        help="Also drop unused columns from movies_recommended.pkl and games_recommended.pkl",
    )
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    artifacts_dir = Path(args.artifacts_dir)

    npy_path = artifacts_dir / "cosine_sim_games.npy"
    if npy_path.exists():
        shrink_npy(npy_path, dtype)
    else:
        print(f"Not found, skipping: {npy_path}")

    pkl_path = artifacts_dir / "cosine_sim_movies.pkl"
    if pkl_path.exists():
        shrink_pickled_matrix(pkl_path, dtype)
    else:
        print(f"Not found, skipping: {pkl_path}")

    if args.prune_columns:
        movies_df_path = artifacts_dir / "movies_recommended.pkl"
        if movies_df_path.exists():
            prune_dataframe(movies_df_path, MOVIE_COLUMNS_NEEDED)
        else:
            print(f"Not found, skipping: {movies_df_path}")

        games_df_path = artifacts_dir / "games_recommended.pkl"
        if games_df_path.exists():
            prune_dataframe(games_df_path, GAME_COLUMNS_NEEDED)
        else:
            print(f"Not found, skipping: {games_df_path}")

    print("\nDone. Re-upload the shrunk files to Google Drive and update the file IDs if they changed.")


if __name__ == "__main__":
    main()
