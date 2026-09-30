"""
interactions.py -- CRUD for the `interactions` collection.

Lightweight event log for anything that isn't a star rating or a
watchlist add -- likes, dislikes, "not interested", clicks-through to a
movie page, etc. Useful later if you want to feed implicit-feedback
signals into the model, or just to show an "activity feed" in the UI.

Schema:
{
  "_id": ObjectId,
  "user_id": str,
  "movie_id": int,
  "action": str,       # "like" | "dislike" | "click" | "not_interested"
  "timestamp": datetime
}
"""

from datetime import datetime, timezone

from mongodb import get_db

VALID_ACTIONS = {"like", "dislike", "click", "not_interested"}


# ---------- CREATE ----------

def log_interaction(user_id: str, movie_id: int, action: str) -> str:
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}, got {action!r}")
    result = get_db().interactions.insert_one(
        {
            "user_id": user_id,
            "movie_id": movie_id,
            "action": action,
            "timestamp": datetime.now(timezone.utc),
        }
    )
    return str(result.inserted_id)


# ---------- READ ----------

def get_user_interactions(user_id: str, limit: int = 50) -> list[dict]:
    return list(
        get_db().interactions.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    )


def get_liked_movie_ids(user_id: str) -> list[int]:
    docs = get_db().interactions.find({"user_id": user_id, "action": "like"})
    return [d["movie_id"] for d in docs]


# ---------- DELETE ----------

def delete_interaction(interaction_id: str) -> bool:
    from bson import ObjectId

    result = get_db().interactions.delete_one({"_id": ObjectId(interaction_id)})
    return result.deleted_count > 0


if __name__ == "__main__":
    iid = log_interaction("demo_user", 1, "like")
    print("Logged:", iid)
    print("Recent:", get_user_interactions("demo_user"))
    print("Deleted:", delete_interaction(iid))
