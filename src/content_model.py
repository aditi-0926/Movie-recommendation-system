"""
content_model.py

Content-based filtering: represent each movie as a TF-IDF vector over its
genres + tags, then recommend movies most similar (cosine similarity) to
ones a user already likes. This works even for brand-new users/movies
with no rating history yet (solves the "cold start" problem that
collaborative filtering can't).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.movie_ids = None
        self.index_by_movie_id = None

    def fit(self, movies: pd.DataFrame):
        """
        movies must have columns: 'movieId' and 'content' (a text blob of
        genres + tags, see src/data_preprocessing.build_movie_features).
        """
        self.movie_ids = movies["movieId"].values
        self.index_by_movie_id = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.tfidf_matrix = self.vectorizer.fit_transform(movies["content"].fillna(""))
        return self

    def similar_to_movie(self, movie_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        """Return [(movieId, similarity_score), ...] most similar to the given movie."""
        if movie_id not in self.index_by_movie_id:
            return []
        idx = self.index_by_movie_id[movie_id]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        ranked = np.argsort(-sims)
        results = []
        for i in ranked:
            candidate_id = self.movie_ids[i]
            if candidate_id == movie_id:
                continue
            results.append((int(candidate_id), float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def recommend_for_profile(self, liked_movie_ids: list[int], top_n: int = 10,
                               exclude_ids: set | None = None) -> list[tuple[int, float]]:
        """
        Build a user "taste vector" by averaging the TF-IDF vectors of movies
        they've rated highly, then return the closest unseen movies to it.
        This is the standard way to do content-based recs for a user (rather
        than just for a single movie).
        """
        exclude_ids = exclude_ids or set()
        idxs = [self.index_by_movie_id[m] for m in liked_movie_ids if m in self.index_by_movie_id]
        if not idxs:
            return []
        profile_vector = self.tfidf_matrix[idxs].mean(axis=0)
        profile_vector = np.asarray(profile_vector)
        sims = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        ranked = np.argsort(-sims)
        results = []
        for i in ranked:
            candidate_id = int(self.movie_ids[i])
            if candidate_id in liked_movie_ids or candidate_id in exclude_ids:
                continue
            results.append((candidate_id, float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def save(self, path: str = None):
        path = path or os.path.join(MODELS_DIR, "content_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "movie_ids": self.movie_ids,
                "index_by_movie_id": self.index_by_movie_id,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(MODELS_DIR, "content_model.pkl")
        payload = joblib.load(path)
        model = cls()
        model.vectorizer = payload["vectorizer"]
        model.tfidf_matrix = payload["tfidf_matrix"]
        model.movie_ids = payload["movie_ids"]
        model.index_by_movie_id = payload["index_by_movie_id"]
        return model


if __name__ == "__main__":
    from data_preprocessing import load_and_clean

    data = load_and_clean()
    model = ContentBasedRecommender().fit(data["movies"])
    saved_path = model.save()
    print(f"Saved content model to {saved_path}")

    # Quick smoke test: movies similar to Toy Story (movieId=1)
    id_to_title = dict(zip(data["movies"]["movieId"], data["movies"]["clean_title"]))
    print("\nMovies similar to Toy Story:")
    for mid, score in model.similar_to_movie(1, top_n=5):
        print(f"  {id_to_title.get(mid, mid):40s} score={score:.3f}")
