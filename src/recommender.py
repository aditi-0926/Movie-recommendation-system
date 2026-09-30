"""
recommender.py

Hybrid recommender: blends collaborative filtering (what similar users
liked) with content-based filtering (what similar movies look like).

Strategy:
- Known user with enough ratings -> weighted blend of both scores.
- Cold-start user (new, or too few ratings) -> content-based only,
  seeded from whatever they've rated/liked so far, or global popularity
  if they have nothing at all.

Both component scores are min-max normalized to [0, 1] before blending,
since raw cosine similarity and raw predicted-rating are on different
scales.
"""

import os
import numpy as np
import pandas as pd

from content_model import ContentBasedRecommender
from collaborative_model import CollaborativeRecommender

MIN_RATINGS_FOR_CF = 5  # below this, we don't trust the collaborative signal


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRecommender:
    def __init__(self, content_model: ContentBasedRecommender,
                 collaborative_model: CollaborativeRecommender,
                 movies: pd.DataFrame,
                 content_weight: float = 0.4,
                 collaborative_weight: float = 0.6):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.movies = movies
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self._popularity_ranked = (
            movies.sort_values(["num_ratings", "avg_rating"], ascending=False)["movieId"].tolist()
        )

    def _popular_fallback(self, exclude_ids: set, top_n: int) -> list[tuple[int, float]]:
        picks = [m for m in self._popularity_ranked if m not in exclude_ids][:top_n]
        return [(m, 1.0) for m in picks]

    def recommend(self, user_id: int | None, rated_movie_ids: dict[int, float] | None = None,
                  top_n: int = 10) -> list[dict]:
        """
        rated_movie_ids: {movieId: rating} the user has already given, used both to
        exclude already-seen movies and (for content-based) to build a taste profile
        from the ones they rated >= 4.

        Returns a list of dicts: {movieId, title, score} sorted best-first.
        """
        rated_movie_ids = rated_movie_ids or {}
        liked_ids = [m for m, r in rated_movie_ids.items() if r >= 4.0]
        exclude_ids = set(rated_movie_ids.keys())

        is_known_cf_user = (
            user_id is not None
            and self.collaborative_model.user_index is not None
            and user_id in self.collaborative_model.user_index
            and len(rated_movie_ids) >= MIN_RATINGS_FOR_CF
        )

        content_scores = {}
        if liked_ids:
            content_scores = dict(
                self.content_model.recommend_for_profile(liked_ids, top_n=top_n * 5, exclude_ids=exclude_ids)
            )

        if not is_known_cf_user:
            # Cold start: content-only, or pure popularity if we have nothing to go on.
            if content_scores:
                ranked = sorted(content_scores.items(), key=lambda kv: -kv[1])[:top_n]
                results = ranked
            else:
                results = self._popular_fallback(exclude_ids, top_n)
            return self._attach_titles(results)

        collaborative_scores = dict(
            self.collaborative_model.recommend_for_user(user_id, exclude_ids, top_n=top_n * 5)
        )

        norm_content = _normalize(content_scores)
        norm_collab = _normalize(collaborative_scores)

        all_ids = set(norm_content) | set(norm_collab)
        blended = {}
        for mid in all_ids:
            c_score = norm_content.get(mid, 0.0)
            cf_score = norm_collab.get(mid, 0.0)
            blended[mid] = (
                self.content_weight * c_score + self.collaborative_weight * cf_score
            )

        ranked = sorted(blended.items(), key=lambda kv: -kv[1])[:top_n]
        return self._attach_titles(ranked)

    def _attach_titles(self, scored_ids: list[tuple[int, float]]) -> list[dict]:
        title_lookup = dict(zip(self.movies["movieId"], self.movies["clean_title"]))
        genre_lookup = dict(zip(self.movies["movieId"], self.movies["genre_str"]))
        return [
            {
                "movieId": mid,
                "title": title_lookup.get(mid, f"Movie {mid}"),
                "genres": genre_lookup.get(mid, ""),
                "score": round(score, 4),
            }
            for mid, score in scored_ids
        ]


def load_hybrid_recommender(models_dir: str = None) -> HybridRecommender:
    """Convenience loader: reads saved models + movie metadata from disk."""
    from data_preprocessing import load_and_clean

    content_model = ContentBasedRecommender.load(
        os.path.join(models_dir, "content_model.pkl") if models_dir else None
    )
    collaborative_model = CollaborativeRecommender.load(
        os.path.join(models_dir, "collaborative_model.pkl") if models_dir else None
    )
    data = load_and_clean()
    return HybridRecommender(content_model, collaborative_model, data["movies"])


if __name__ == "__main__":
    from data_preprocessing import load_and_clean

    data = load_and_clean()
    hybrid = load_hybrid_recommender()

    sample_user = int(data["ratings"]["userId"].iloc[0])
    user_ratings = dict(
        zip(
            data["ratings"][data["ratings"]["userId"] == sample_user]["movieId"],
            data["ratings"][data["ratings"]["userId"] == sample_user]["rating"],
        )
    )
    print(f"Hybrid recs for existing user {sample_user}:")
    for rec in hybrid.recommend(sample_user, user_ratings, top_n=5):
        print(f"  {rec['title']:40s} ({rec['genres']:30s}) score={rec['score']}")

    print("\nHybrid recs for a brand-new user who just rated Toy Story 5 stars (cold start):")
    for rec in hybrid.recommend(None, {1: 5.0}, top_n=5):
        print(f"  {rec['title']:40s} ({rec['genres']:30s}) score={rec['score']}")

    print("\nHybrid recs for a totally new user with zero ratings (popularity fallback):")
    for rec in hybrid.recommend(None, {}, top_n=5):
        print(f"  {rec['title']:40s} ({rec['genres']:30s}) score={rec['score']}")
