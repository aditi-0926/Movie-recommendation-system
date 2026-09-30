"""
ratings.py -- CRUD for the `ratings` collection.

Schema:
{
  "_id": ObjectId,
  "user_id": str,     # stringified ObjectId of the user
  "movie_id": int,    # movieId, matches movies.movieId
  "rating": float,    # 0.5 - 5.0
  "timestamp": datetime
}
"""

from datetime import datetime, timezone

from mongodb import get_db


# ---------- CREATE / UPDATE (upsert -- rating a movie twice just updates it) ----------

def rate_movie(user_id: str, movie_id: int, rating: float) -> bool:
    result = get_db().ratings.update_one(
        {"user_id": user_id, "movie_id": movie_id},
        {"$set": {"rating": rating, "timestamp": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return result.upserted_id is not None or result.modified_count > 0


# ---------- READ ----------

def get_user_ratings(user_id: str) -> list[dict]:
    return list(get_db().ratings.find({"user_id": user_id}))


def get_user_ratings_dict(user_id: str) -> dict[int, float]:
    """Convenience shape for feeding straight into the recommender: {movieId: rating}."""
    return {r["movie_id"]: r["rating"] for r in get_user_ratings(user_id)}


def get_movie_ratings(movie_id: int) -> list[dict]:
    return list(get_db().ratings.find({"movie_id": movie_id}))


def get_rating(user_id: str, movie_id: int) -> dict | None:
    return get_db().ratings.find_one({"user_id": user_id, "movie_id": movie_id})


# ---------- UPDATE ----------
# (rate_movie above already handles update-in-place via upsert)


# ---------- DELETE ----------

def delete_rating(user_id: str, movie_id: int) -> bool:
    result = get_db().ratings.delete_one({"user_id": user_id, "movie_id": movie_id})
    return result.deleted_count > 0


if __name__ == "__main__":
    print(rate_movie("demo_user", 1, 4.5))
    print(get_user_ratings_dict("demo_user"))
    print(delete_rating("demo_user", 1))
