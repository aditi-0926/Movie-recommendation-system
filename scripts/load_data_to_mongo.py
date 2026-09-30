"""
scripts/load_data_to_mongo.py

One-time (or re-runnable) seed script: reads the MovieLens CSVs, cleans
them, and upserts every movie into MongoDB Atlas. Run this once after you
set up your Atlas cluster and .env file.

Usage:
    python scripts/load_data_to_mongo.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "database"))

from data_preprocessing import load_and_clean  # noqa: E402
from mongodb import ping, ensure_indexes  # noqa: E402
from movies import bulk_load_movies  # noqa: E402


def main():
    print("Connecting to MongoDB Atlas...")
    ping()
    print("Connected. Ensuring indexes...")
    ensure_indexes()

    print("Loading and cleaning MovieLens data...")
    data = load_and_clean()
    movies = data["movies"]
    print(f"  {len(movies)} movies ready to upsert.")

    print("Upserting movies into MongoDB...")
    count = bulk_load_movies(movies)
    print(f"Done. {count} documents inserted/updated in the 'movies' collection.")


if __name__ == "__main__":
    main()
