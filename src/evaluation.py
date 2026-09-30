"""
evaluation.py

Evaluates the recommendation models honestly using a held-out test split:

- RMSE / MAE: how close the collaborative model's predicted ratings are
  to actual ratings the user gave (rating-prediction accuracy).
- Precision@K / Recall@K / Hit Rate@K: of the top-K movies we'd recommend,
  how many did the user actually go on to rate highly (>= 4 stars) in the
  test set (ranking / top-N quality -- closer to what a real product cares
  about than raw RMSE).

Run directly:
    python src/evaluation.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_clean
from collaborative_model import CollaborativeRecommender

RELEVANCE_THRESHOLD = 4.0  # a test rating >= this counts as "relevant" for Precision/Recall@K


def rmse_mae(model: CollaborativeRecommender, test_ratings: pd.DataFrame) -> dict:
    errors = []
    for row in test_ratings.itertuples():
        pred = model.predict_rating(row.userId, row.movieId)
        if pred is None:
            continue
        errors.append(pred - row.rating)
    errors = np.array(errors)
    if len(errors) == 0:
        return {"rmse": None, "mae": None, "n_predictions": 0}
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return {"rmse": rmse, "mae": mae, "n_predictions": len(errors)}


def precision_recall_at_k(model: CollaborativeRecommender, train_ratings: pd.DataFrame,
                           test_ratings: pd.DataFrame, k: int = 10) -> dict:
    precisions, recalls, hits = [], [], []

    train_by_user = train_ratings.groupby("userId")["movieId"].apply(set)
    test_relevant_by_user = (
        test_ratings[test_ratings["rating"] >= RELEVANCE_THRESHOLD]
        .groupby("userId")["movieId"]
        .apply(set)
    )

    for user_id, relevant_movies in test_relevant_by_user.items():
        if user_id not in model.user_index or not relevant_movies:
            continue
        already_rated = train_by_user.get(user_id, set())
        recs = model.recommend_for_user(user_id, already_rated, top_n=k)
        recommended_ids = {mid for mid, _ in recs}
        if not recommended_ids:
            continue

        n_hits = len(recommended_ids & relevant_movies)
        precisions.append(n_hits / len(recommended_ids))
        recalls.append(n_hits / len(relevant_movies))
        hits.append(1 if n_hits > 0 else 0)

    if not precisions:
        return {"precision_at_k": None, "recall_at_k": None, "hit_rate_at_k": None, "n_users_evaluated": 0}

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "hit_rate_at_k": float(np.mean(hits)),
        "n_users_evaluated": len(precisions),
        "k": k,
    }


def run_full_evaluation(k: int = 10, test_size: float = 0.2, random_state: int = 42) -> dict:
    data = load_and_clean()
    ratings = data["ratings"]

    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=random_state, stratify=None
    )

    model = CollaborativeRecommender(n_factors=50).fit(train_ratings)

    rating_metrics = rmse_mae(model, test_ratings)
    ranking_metrics = precision_recall_at_k(model, train_ratings, test_ratings, k=k)

    return {"rating_prediction": rating_metrics, "top_n_ranking": ranking_metrics}


if __name__ == "__main__":
    results = run_full_evaluation(k=10)
    print("=== Rating Prediction (trained on 80% split, tested on held-out 20%) ===")
    rp = results["rating_prediction"]
    print(f"  RMSE: {rp['rmse']:.4f}" if rp["rmse"] is not None else "  RMSE: n/a")
    print(f"  MAE:  {rp['mae']:.4f}" if rp["mae"] is not None else "  MAE: n/a")
    print(f"  (evaluated on {rp['n_predictions']} held-out ratings)")

    print("\n=== Top-N Ranking Quality (K=10) ===")
    tn = results["top_n_ranking"]
    if tn["precision_at_k"] is not None:
        print(f"  Precision@10: {tn['precision_at_k']:.4f}")
        print(f"  Recall@10:    {tn['recall_at_k']:.4f}")
        print(f"  Hit Rate@10:  {tn['hit_rate_at_k']:.4f}")
        print(f"  (evaluated on {tn['n_users_evaluated']} users)")
    else:
        print("  Not enough overlapping users/ratings to evaluate.")
