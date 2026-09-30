"""
watchlist.py -- CRUD for the `watchlist` collection.

Schema:
{
  "_id": ObjectId,
  "user_id": str,
  "movie_id": int,
  "added_at": datetime
}
"""

from datetime import datetime, timezone

from mongodb import get_db


# ---------- CREATE ----------

def add_to_watchlist(user_id: str, movie_id: int) -> bool:
    result = get_db().watchlist.update_one(
        {"user_id": user_id, "movie_id": movie_id},
        {"$setOnInsert": {"added_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return result.upserted_id is not None


# ---------- READ ----------

def get_watchlist(user_id: str) -> list[dict]:
    return list(get_db().watchlist.find({"user_id": user_id}).sort("added_at", -1))


def is_in_watchlist(user_id: str, movie_id: int) -> bool:
    return get_db().watchlist.find_one({"user_id": user_id, "movie_id": movie_id}) is not None


# ---------- DELETE ----------

def remove_from_watchlist(user_id: str, movie_id: int) -> bool:
    result = get_db().watchlist.delete_one({"user_id": user_id, "movie_id": movie_id})
    return result.deleted_count > 0


if __name__ == "__main__":
    print(add_to_watchlist("demo_user", 1))
    print(get_watchlist("demo_user"))
    print(remove_from_watchlist("demo_user", 1))
