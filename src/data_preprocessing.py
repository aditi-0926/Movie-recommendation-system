"""
data_preprocessing.py

Loads the raw MovieLens ml-latest-small CSVs, cleans them, and produces
tidy DataFrames that the rest of the pipeline (EDA, content model,
collaborative model, MongoDB import) all build on.

Run directly to sanity-check the pipeline:
    python src/data_preprocessing.py
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_raw(data_dir: str = DATA_DIR) -> dict:
    """Load the raw MovieLens CSVs into a dict of DataFrames."""
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    links = pd.read_csv(os.path.join(data_dir, "links.csv"))
    tags = pd.read_csv(os.path.join(data_dir, "tags.csv"))
    return {"movies": movies, "ratings": ratings, "links": links, "tags": tags}


_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _extract_year(title: str):
    match = _YEAR_RE.search(title)
    return int(match.group(1)) if match else None


def _clean_title(title: str) -> str:
    return _YEAR_RE.sub("", title).strip()


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Split MovieLens 'title (year)' into clean title + year, turn the
    pipe-separated genre string into a list, and flag movies with no
    genre info.
    """
    df = movies.copy()
    df["year"] = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].apply(_clean_title)
    df["genre_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    df["genre_str"] = df["genre_list"].apply(lambda g: " ".join(g))
    return df


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Convert the unix timestamp to a real datetime and drop exact dupes."""
    df = ratings.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")
    return df


def aggregate_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Collapse free-text tags per movie into one space-joined string per movieId."""
    if tags.empty:
        return pd.DataFrame(columns=["movieId", "tag_str"])
    grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda vals: " ".join(str(v) for v in vals))
        .reset_index()
        .rename(columns={"tag": "tag_str"})
    )
    return grouped


def build_movie_features(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned movie metadata with aggregated tags into a single
    'content' text field used later for TF-IDF (content-based model).
    """
    movies_clean = clean_movies(movies)
    tag_agg = aggregate_tags(tags)
    merged = movies_clean.merge(tag_agg, on="movieId", how="left")
    merged["tag_str"] = merged["tag_str"].fillna("")
    merged["content"] = (merged["genre_str"] + " " + merged["tag_str"]).str.strip()
    return merged


def compute_movie_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Average rating and rating count per movie -- used for popularity fallback."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", num_ratings="count")
        .reset_index()
    )
    return stats


def load_and_clean(data_dir: str = DATA_DIR) -> dict:
    """One-call entry point: load raw CSVs and return all cleaned artifacts."""
    raw = load_raw(data_dir)
    movie_features = build_movie_features(raw["movies"], raw["tags"])
    ratings_clean = clean_ratings(raw["ratings"])
    movie_stats = compute_movie_stats(ratings_clean)
    movie_features = movie_features.merge(movie_stats, on="movieId", how="left")
    movie_features["avg_rating"] = movie_features["avg_rating"].fillna(0.0)
    movie_features["num_ratings"] = movie_features["num_ratings"].fillna(0).astype(int)

    return {
        "movies": movie_features,
        "ratings": ratings_clean,
        "links": raw["links"],
        "tags": raw["tags"],
    }


if __name__ == "__main__":
    data = load_and_clean()
    print("Movies:", data["movies"].shape)
    print("Ratings:", data["ratings"].shape)
    print(data["movies"][["movieId", "clean_title", "year", "genre_list", "avg_rating", "num_ratings"]].head())
