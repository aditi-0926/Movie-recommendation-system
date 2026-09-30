"""
movies.py -- CRUD for the `movies` collection.

Schema:
{
  "_id": ObjectId,
  "movieId": int,        # original MovieLens id, used as the join key everywhere
  "title": str,
  "clean_title": str,
  "year": int | None,
  "genres": [str, ...],
  "avg_rating": float,
  "num_ratings": int
}
"""

import pandas as pd
from pymongo import UpdateOne

from mongodb import get_db


# ---------- CREATE ----------

def add_movie(movie_id: int, title: str, genres: list[str], year: int | None = None) -> str:
    db = get_db()
    doc = {
        "movieId": movie_id,
        "title": title,
        "clean_title": title,
        "year": year,
        "genres": genres,
        "avg_rating": 0.0,
        "num_ratings": 0,
    }
    result = db.movies.insert_one(doc)
    return str(result.inserted_id)


def bulk_load_movies(movies_df: pd.DataFrame) -> int:
    """
    Upsert every row of the cleaned movies DataFrame (see
    src/data_preprocessing.py) into MongoDB in one batched call. Used by
    scripts/load_data_to_mongo.py to seed the database from MovieLens.
    """
    db = get_db()
    operations = []
    for row in movies_df.itertuples():
        operations.append(
            UpdateOne(
                {"movieId": row.movieId},
                {
                    "$set": {
                        "movieId": row.movieId,
                        "title": row.title,
                        "clean_title": row.clean_title,
                        "year": row.year,
                        "genres": row.genre_list,
                        "avg_rating": round(float(row.avg_rating), 3),
                        "num_ratings": int(row.num_ratings),
                    }
                },
                upsert=True,
            )
        )
    if not operations:
        return 0
    result = db.movies.bulk_write(operations)
    return result.upserted_count + result.modified_count


# ---------- READ ----------

def get_movie(movie_id: int) -> dict | None:
    return get_db().movies.find_one({"movieId": movie_id})


def search_movies(query: str, limit: int = 20) -> list[dict]:
    """Simple case-insensitive title search (requires the text index from mongodb.ensure_indexes)."""
    db = get_db()
    return list(db.movies.find({"title": {"$regex": query, "$options": "i"}}).limit(limit))


def get_movies_by_genre(genre: str, limit: int = 20) -> list[dict]:
    return list(get_db().movies.find({"genres": genre}).limit(limit))


def get_top_rated(limit: int = 20, min_ratings: int = 20) -> list[dict]:
    return list(
        get_db()
        .movies.find({"num_ratings": {"$gte": min_ratings}})
        .sort([("avg_rating", -1)])
        .limit(limit)
    )


# ---------- UPDATE ----------

def update_movie_stats(movie_id: int, avg_rating: float, num_ratings: int) -> bool:
    result = get_db().movies.update_one(
        {"movieId": movie_id},
        {"$set": {"avg_rating": round(avg_rating, 3), "num_ratings": num_ratings}},
    )
    return result.modified_count > 0


# ---------- DELETE ----------

def delete_movie(movie_id: int) -> bool:
    result = get_db().movies.delete_one({"movieId": movie_id})
    return result.deleted_count > 0


if __name__ == "__main__":
    print("Top rated movies currently in DB:")
    for m in get_top_rated(limit=5):
        print(f"  {m['title']:40s} avg={m['avg_rating']} n={m['num_ratings']}")
