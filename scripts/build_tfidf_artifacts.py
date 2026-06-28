"""
scripts/build_tfidf_artifacts.py
---------------------------------
One-time offline script: reads the existing dataset pickles produced by
the notebooks and writes the sparse TF-IDF matrix (.npz) that the
refactored backend now expects at startup.

Run this locally (not on Render) where memory isn't constrained, then
upload the two .npz files to Google Drive and update MOVIES_TFIDF_FILE_ID
/ GAMES_TFIDF_FILE_ID in config.py (or your .env file).

Usage
------
    python scripts/build_tfidf_artifacts.py \
        --movies-pkl /path/to/movies_recommended.pkl \
        --games-pkl  /path/to/games_recommended.pkl \
        --out-dir    ./artifacts

Outputs (in --out-dir)
-----------------------
    movies_tfidf.npz   — sparse TF-IDF matrix for movies
    games_tfidf.npz    — sparse TF-IDF matrix for games

These are the only two new files you need to upload to Google Drive.
The dataset pickles (movies_recommended.pkl, games_recommended.pkl)
are unchanged and do not need to be re-uploaded.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pickle(path: Path):
    print(f"  Loading {path.name} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print("done")
    return obj


def build_tfidf(df, text_column: str) -> scipy.sparse.csr_matrix:
    """
    Fit a TF-IDF vectorizer on `df[text_column]` and return the sparse
    feature matrix.  Mirrors the vectorizer settings used in the notebooks
    (no explicit max_features cap so vocab exactly matches the original;
    stop_words='english' as in the notebook).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(""))
    print(f"    vocab size : {len(vectorizer.vocabulary_):,}")
    print(f"    matrix     : {tfidf_matrix.shape}, {tfidf_matrix.nnz:,} nonzeros")
    return tfidf_matrix


def save_sparse(matrix: scipy.sparse.csr_matrix, path: Path) -> None:
    scipy.sparse.save_npz(str(path), matrix)
    size_mb = path.stat().st_size / 1_048_576
    print(f"  Saved {path.name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Per-domain processing
# ---------------------------------------------------------------------------

def process_movies(pkl_path: Path, out_dir: Path) -> None:
    print("\n[MOVIES]")
    df = load_pickle(pkl_path)

    # The notebook uses a 'tags' column (concatenated genres + keywords +
    # cast + overview) as the TF-IDF input.  Check which column exists.
    text_col = _detect_text_column(df, ["tags", "soup", "combined_features", "description"])
    print(f"  text column : '{text_col}'  ({len(df):,} rows)")

    tfidf = build_tfidf(df, text_col)
    save_sparse(tfidf, out_dir / "movies_tfidf.npz")


def process_games(pkl_path: Path, out_dir: Path) -> None:
    print("\n[GAMES]")
    df = load_pickle(pkl_path)

    text_col = _detect_text_column(df, ["tags", "soup", "combined_features", "description_clean"])
    print(f"  text column : '{text_col}'  ({len(df):,} rows)")

    tfidf = build_tfidf(df, text_col)
    save_sparse(tfidf, out_dir / "games_tfidf.npz")


def _detect_text_column(df, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    print(f"  WARNING: none of {candidates} found — available columns:")
    print("  ", list(df.columns))
    raise SystemExit(
        "Cannot determine text column.  Pass --movies-text-col / --games-text-col explicitly."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sparse TF-IDF artifacts for the recommender backend.")
    p.add_argument("--movies-pkl", required=True, type=Path, help="Path to movies_recommended.pkl")
    p.add_argument("--games-pkl",  required=True, type=Path, help="Path to games_recommended.pkl")
    p.add_argument("--out-dir",    default=Path("./artifacts"), type=Path,
                   help="Directory to write .npz files (default: ./artifacts)")
    p.add_argument("--movies-text-col", default=None,
                   help="Override auto-detection of the text column for movies")
    p.add_argument("--games-text-col",  default=None,
                   help="Override auto-detection of the text column for games")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.out_dir.resolve()}")

    # Allow manual column override
    import types, functools
    if args.movies_text_col:
        globals()["_detect_text_column"] = lambda df, _: args.movies_text_col
    if args.games_text_col:
        globals()["_detect_text_column"] = lambda df, _: args.games_text_col

    process_movies(args.movies_pkl, args.out_dir)
    process_games(args.games_pkl,  args.out_dir)

    print("\nDone.  Upload the two .npz files to Google Drive, then update")
    print("MOVIES_TFIDF_FILE_ID and GAMES_TFIDF_FILE_ID in config.py / .env.")


if __name__ == "__main__":
    main()
