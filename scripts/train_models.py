"""
scripts/train_models.py

Trains the content-based and collaborative models on the MovieLens data
and saves both to models/*.pkl, ready for the Streamlit app to load.

Usage:
    python scripts/train_models.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import load_and_clean  # noqa: E402
from content_model import ContentBasedRecommender  # noqa: E402
from collaborative_model import CollaborativeRecommender  # noqa: E402


def main():
    print("Loading and cleaning data...")
    data = load_and_clean()

    print("Training content-based model (TF-IDF + cosine similarity)...")
    content_model = ContentBasedRecommender().fit(data["movies"])
    path = content_model.save()
    print(f"  Saved -> {path}")

    print("Training collaborative filtering model (SVD matrix factorization)...")
    collaborative_model = CollaborativeRecommender(n_factors=50).fit(data["ratings"])
    path = collaborative_model.save()
    print(f"  Saved -> {path}")

    print("\nBoth models trained and saved. Run the app with:")
    print("  streamlit run app/app.py")


if __name__ == "__main__":
    main()
