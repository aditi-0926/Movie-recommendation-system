"""
mongodb.py

Single shared MongoDB Atlas connection for the whole app. Every other
database/*.py module imports get_db() from here rather than opening its
own connection.

Requires a .env file (copy .env.example -> .env) with:
    MONGO_URI=mongodb+srv://...
    MONGO_DB_NAME=movie_recommender
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

def _get_config(key: str, default: str = None):
    # Prefer Streamlit Cloud secrets if available, fall back to .env/os.environ for local dev
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

MONGO_URI = _get_config("MONGO_URI")
MONGO_DB_NAME = _get_config("MONGO_DB_NAME", "movie_recommender")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    if not MONGO_URI or MONGO_URI.startswith("your_mongodb"):
        raise RuntimeError(
            "MONGO_URI is not set. Copy .env.example to .env and paste in your "
            "MongoDB Atlas connection string."
        )
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))


@lru_cache(maxsize=1)
def get_db():
    return get_client()[MONGO_DB_NAME]


def ping() -> bool:
    """Quick connectivity check -- call this once at app startup."""
    get_client().admin.command("ping")
    return True


def ensure_indexes():
    """
    Create the indexes the app relies on. Safe to call repeatedly --
    create_index is a no-op if the index already exists with the same spec.
    """
    db = get_db()
    db.users.create_index("email", unique=True)
    db.movies.create_index("movieId", unique=True)
    db.movies.create_index([("genres", 1)])
    db.movies.create_index([("title", "text")])
    db.ratings.create_index([("user_id", 1), ("movie_id", 1)], unique=True)
    db.ratings.create_index("movie_id")
    db.watchlist.create_index([("user_id", 1), ("movie_id", 1)], unique=True)
    db.interactions.create_index([("user_id", 1), ("timestamp", -1)])


if __name__ == "__main__":
    print(f"Connecting to database '{MONGO_DB_NAME}'...")
    ping()
    print("Connected successfully.")
    ensure_indexes()
    print("Indexes ensured:")
    for name in ["users", "movies", "ratings", "watchlist", "interactions"]:
        print(f"  {name}: {list(get_db()[name].index_information().keys())}")
