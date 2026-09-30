"""
notebooks/eda.py

Exploratory data analysis on the MovieLens ratings data. This is a plain
script (not a .ipynb) so it runs anywhere without a Jupyter kernel --
feel free to paste these cells into 01_EDA.ipynb if you want the notebook
format for your portfolio/GitHub.

Produces PNGs in notebooks/figures/:
    rating_distribution.png
    ratings_per_user.png
    ratings_per_movie.png
    genre_popularity.png
    avg_rating_by_genre.png

Run:
    python notebooks/eda.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preprocessing import load_and_clean  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def main():
    data = load_and_clean()
    movies, ratings = data["movies"], data["ratings"]

    print(f"Movies: {len(movies)}, Ratings: {len(ratings)}, Users: {ratings['userId'].nunique()}")

    # 1. Rating distribution
    plt.figure(figsize=(6, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Distribution of Star Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    savefig("rating_distribution.png")

    # 2. Ratings per user
    per_user = ratings.groupby("userId").size()
    plt.figure(figsize=(6, 4))
    per_user.plot(kind="hist", bins=40, color="#55A868")
    plt.title("Number of Ratings per User")
    plt.xlabel("Ratings given")
    plt.ylabel("Number of users")
    savefig("ratings_per_user.png")
    print(f"  median ratings/user: {per_user.median():.0f}, max: {per_user.max()}")

    # 3. Ratings per movie
    per_movie = ratings.groupby("movieId").size()
    plt.figure(figsize=(6, 4))
    per_movie.plot(kind="hist", bins=40, color="#C44E52")
    plt.title("Number of Ratings per Movie")
    plt.xlabel("Ratings received")
    plt.ylabel("Number of movies")
    savefig("ratings_per_movie.png")
    print(f"  median ratings/movie: {per_movie.median():.0f}, max: {per_movie.max()}")

    # 4. Genre popularity (by rating count)
    exploded = movies.explode("genre_list")
    exploded = exploded[exploded["genre_list"].notna() & (exploded["genre_list"] != "")]
    genre_counts = exploded.groupby("genre_list")["num_ratings"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    genre_counts.plot(kind="barh", color="#8172B2")
    plt.title("Total Ratings by Genre")
    plt.xlabel("Total ratings")
    plt.gca().invert_yaxis()
    savefig("genre_popularity.png")

    # 5. Average rating by genre (only genres with a reasonable sample size)
    genre_avg = (
        exploded[exploded["num_ratings"] >= 20]
        .groupby("genre_list")
        .apply(lambda g: (g["avg_rating"] * g["num_ratings"]).sum() / g["num_ratings"].sum())
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(8, 5))
    genre_avg.plot(kind="barh", color="#CCB974")
    plt.title("Average Rating by Genre (weighted, min 20 ratings)")
    plt.xlabel("Average rating")
    plt.gca().invert_yaxis()
    savefig("avg_rating_by_genre.png")

    print("\nEDA complete. Figures saved to notebooks/figures/")


if __name__ == "__main__":
    main()
