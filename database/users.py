"""
users.py -- CRUD for the `users` collection.

Schema:
{
  "_id": ObjectId,
  "name": str,
  "email": str (unique),
  "password_hash": str,
  "age": int | None,
  "preferences": {"genres": [str, ...]},
  "created_at": datetime
}
"""

import hashlib
import os
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId

from mongodb import get_db


def _hash_password(password: str) -> str:
    """
    Salted SHA-256 hash. Fine for a portfolio project; if this ever became
    a real product, swap this for bcrypt/argon2 (passlib) instead.
    """
    salt = os.getenv("PASSWORD_SALT", "movie-rec-dev-salt")
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


# ---------- CREATE ----------

def create_user(name: str, email: str, password: str, age: int | None = None,
                 genres: list[str] | None = None) -> str:
    db = get_db()
    doc = {
        "name": name,
        "email": email.lower().strip(),
        "password_hash": _hash_password(password),
        "age": age,
        "preferences": {"genres": genres or []},
        "created_at": datetime.now(timezone.utc),
    }
    result = db.users.insert_one(doc)
    return str(result.inserted_id)


# ---------- READ ----------

def get_user_by_email(email: str) -> dict | None:
    return get_db().users.find_one({"email": email.lower().strip()})


def get_user_by_id(user_id: str) -> dict | None:
    try:
        return get_db().users.find_one({"_id": ObjectId(user_id)})
    except InvalidId:
        return None


def verify_login(email: str, password: str) -> dict | None:
    """Returns the user doc if email+password match, else None."""
    user = get_user_by_email(email)
    if user and user["password_hash"] == _hash_password(password):
        return user
    return None


def list_users(limit: int = 50) -> list[dict]:
    return list(get_db().users.find().limit(limit))


# ---------- UPDATE ----------

def update_preferences(user_id: str, genres: list[str]) -> bool:
    result = get_db().users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"preferences.genres": genres}},
    )
    return result.modified_count > 0


def update_profile(user_id: str, **fields) -> bool:
    """Generic partial update, e.g. update_profile(uid, name='New Name', age=25)."""
    allowed = {"name", "age"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    result = get_db().users.update_one({"_id": ObjectId(user_id)}, {"$set": updates})
    return result.modified_count > 0


# ---------- DELETE ----------

def delete_user(user_id: str) -> bool:
    """Deletes the user AND their ratings/watchlist/interactions (cascade)."""
    db = get_db()
    oid = ObjectId(user_id)
    db.ratings.delete_many({"user_id": user_id})
    db.watchlist.delete_many({"user_id": user_id})
    db.interactions.delete_many({"user_id": user_id})
    result = db.users.delete_one({"_id": oid})
    return result.deleted_count > 0


if __name__ == "__main__":
    # Tiny manual smoke test against your real Atlas cluster.
    uid = create_user("Test User", "test.user@example.com", "password123", age=25, genres=["Sci-Fi"])
    print("Created:", uid)
    print("Fetched:", get_user_by_email("test.user@example.com"))
    print("Login OK:", verify_login("test.user@example.com", "password123") is not None)
    print("Updated prefs:", update_preferences(uid, ["Sci-Fi", "Comedy"]))
    print("Deleted:", delete_user(uid))
