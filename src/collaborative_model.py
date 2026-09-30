"""
collaborative_model.py

Collaborative filtering via matrix factorization (truncated SVD over the
user-item ratings matrix). This asks "which users have similar taste?"
rather than "which movies look similar?" -- it can surface non-obvious
recommendations (e.g. two sci-fi fans who also both love a random indie
comedy) that a pure content model would never find.

Note on cold start: a brand-new user or movie with no ratings has no
signal here, which is why the hybrid recommender (src/recommender.py)
falls back to the content-based model for those cases.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class CollaborativeRecommender:
    def __init__(self, n_factors: int = 100):
        self.n_factors = n_factors
        self.user_ids = None
        self.movie_ids = None
        self.user_index = None
        self.movie_index = None
        self.user_means = None
        self.predicted_matrix = None  # dense, users x movies, reconstructed ratings

    def fit(self, ratings: pd.DataFrame):
        """
        ratings must have columns: userId, movieId, rating.
        Builds a sparse user-item matrix, mean-centers per user, and
        factorizes with truncated SVD to fill in the missing entries.
        """
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}

        rows = ratings["userId"].map(self.user_index)
        cols = ratings["movieId"].map(self.movie_index)
        sparse_matrix = csr_matrix(
            (ratings["rating"], (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        dense = sparse_matrix.toarray()
        # mean-center each user's ratings (ignoring the zeros / unrated cells)
        rated_mask = dense != 0
        sums = dense.sum(axis=1)
        counts = rated_mask.sum(axis=1)
        counts[counts == 0] = 1
        self.user_means = sums / counts
        centered = dense - self.user_means.reshape(-1, 1)
        centered[~rated_mask] = 0

        k = min(self.n_factors, min(centered.shape) - 1)
        U, sigma, Vt = svds(centered, k=k)
        sigma_diag = np.diag(sigma)
        reconstructed = U @ sigma_diag @ Vt + self.user_means.reshape(-1, 1)
        self.predicted_matrix = reconstructed
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float | None:
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return None
        u = self.user_index[user_id]
        m = self.movie_index[movie_id]
        return float(self.predicted_matrix[u, m])

    def recommend_for_user(self, user_id: int, rated_movie_ids: set, top_n: int = 10) -> list[tuple[int, float]]:
        """Top-N unrated movies for a known user, ranked by predicted rating."""
        if user_id not in self.user_index:
            return []
        u = self.user_index[user_id]
        preds = self.predicted_matrix[u]
        ranked_idx = np.argsort(-preds)
        results = []
        for idx in ranked_idx:
            mid = int(self.movie_ids[idx])
            if mid in rated_movie_ids:
                continue
            results.append((mid, float(preds[idx])))
            if len(results) >= top_n:
                break
        return results

    def save(self, path: str = None):
        path = path or os.path.join(MODELS_DIR, "collaborative_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "n_factors": self.n_factors,
                "user_ids": self.user_ids,
                "movie_ids": self.movie_ids,
                "user_index": self.user_index,
                "movie_index": self.movie_index,
                "user_means": self.user_means,
                "predicted_matrix": self.predicted_matrix,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(MODELS_DIR, "collaborative_model.pkl")
        payload = joblib.load(path)
        model = cls(n_factors=payload["n_factors"])
        model.user_ids = payload["user_ids"]
        model.movie_ids = payload["movie_ids"]
        model.user_index = payload["user_index"]
        model.movie_index = payload["movie_index"]
        model.user_means = payload["user_means"]
        model.predicted_matrix = payload["predicted_matrix"]
        return model


if __name__ == "__main__":
    from data_preprocessing import load_and_clean

    data = load_and_clean()
    model = CollaborativeRecommender(n_factors=50).fit(data["ratings"])
    saved_path = model.save()
    print(f"Saved collaborative model to {saved_path}")

    id_to_title = dict(zip(data["movies"]["movieId"], data["movies"]["clean_title"]))
    sample_user = int(data["ratings"]["userId"].iloc[0])
    rated = set(data["ratings"][data["ratings"]["userId"] == sample_user]["movieId"])
    print(f"\nTop picks for user {sample_user} (has rated {len(rated)} movies):")
    for mid, score in model.recommend_for_user(sample_user, rated, top_n=5):
        print(f"  {id_to_title.get(mid, mid):40s} predicted_rating={score:.2f}")
