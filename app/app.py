"""
app/app.py

Streamlit front-end for the Personalized Movie Recommendation & User
Preference Analytics Platform.

Run with:
    streamlit run app/app.py

Requires:
    - A configured .env (MONGO_URI, MONGO_DB_NAME) -- see .env.example
    - Trained models in models/ -- run scripts/train_models.py first
    - Movies seeded into Mongo -- run scripts/load_data_to_mongo.py first
"""

import os
import sys

import pandas as pd
import streamlit as st

# Make src/ and database/ importable regardless of where streamlit is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "database"))

from mongodb import ping  # noqa: E402
import users as users_db  # noqa: E402
import movies as movies_db  # noqa: E402
import ratings as ratings_db  # noqa: E402
import watchlist as watchlist_db  # noqa: E402
import interactions as interactions_db  # noqa: E402
from recommender import load_hybrid_recommender  # noqa: E402

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to MongoDB Atlas...")
def get_connection_status():
    try:
        ping()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


@st.cache_resource(show_spinner="Loading recommendation models...")
def get_recommender():
    return load_hybrid_recommender()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def render_auth_screen():
    st.title("🎬 Movie Recommender")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = users_db.verify_login(email, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            age = st.number_input("Age", min_value=13, max_value=100, value=25)
            genres = st.multiselect(
                "Favorite genres (optional)",
                ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
                 "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance",
                 "Sci-Fi", "Thriller", "War"],
            )
            submitted = st.form_submit_button("Create account")
            if submitted:
                if not name or not email or not password:
                    st.error("Name, email, and password are all required.")
                elif users_db.get_user_by_email(email):
                    st.error("An account with that email already exists.")
                else:
                    uid = users_db.create_user(name, email, password, age=int(age), genres=genres)
                    st.success("Account created! Please log in.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard(user):
    st.header(f"Welcome back, {user['name']} 👋")
    user_id = str(user["_id"])
    user_ratings = ratings_db.get_user_ratings_dict(user_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("Movies Rated", len(user_ratings))
    avg = sum(user_ratings.values()) / len(user_ratings) if user_ratings else 0
    col2.metric("Average Rating", f"{avg:.1f} ⭐" if user_ratings else "—")
    top_genre = _top_genre_for_user(user_ratings)
    col3.metric("Favorite Genre", top_genre or "—")

    st.subheader("🎬 Recommended For You")
    recommender = get_recommender()
    recs = recommender.recommend(user_id=None, rated_movie_ids=user_ratings, top_n=10)
    _render_movie_grid(recs, user_id, show_match=True)

    st.subheader("Your Genre Preferences")
    _render_genre_chart(user_ratings)


def page_browse(user):
    st.header("🔍 Browse Movies")
    user_id = str(user["_id"])

    col1, col2 = st.columns([3, 1])
    query = col1.text_input("Search by title")
    genre = col2.selectbox(
        "Filter by genre",
        ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
         "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
         "Thriller", "War"],
    )

    if query:
        results = movies_db.search_movies(query, limit=24)
    elif genre != "All":
        results = movies_db.get_movies_by_genre(genre, limit=24)
    else:
        results = movies_db.get_top_rated(limit=24)

    movies_as_recs = [
        {"movieId": m["movieId"], "title": m["title"], "genres": " ".join(m.get("genres", [])), "score": m.get("avg_rating", 0)}
        for m in results
    ]
    _render_movie_grid(movies_as_recs, user_id, show_match=False)


def page_watchlist(user):
    st.header("📌 My Watchlist")
    user_id = str(user["_id"])
    watchlist_entries = watchlist_db.get_watchlist(user_id)

    if not watchlist_entries:
        st.info("Your watchlist is empty. Add movies from the Browse or Dashboard tabs.")
        return

    for entry in watchlist_entries:
        movie = movies_db.get_movie(entry["movie_id"])
        if not movie:
            continue
        col1, col2, col3 = st.columns([4, 2, 1])
        col1.write(f"**{movie['title']}**")
        col2.write(" · ".join(movie.get("genres", [])))
        if col3.button("Remove", key=f"remove_{entry['movie_id']}"):
            watchlist_db.remove_from_watchlist(user_id, entry["movie_id"])
            st.rerun()


def page_analytics(user):
    st.header("📊 Your Analytics")
    user_id = str(user["_id"])
    user_ratings = ratings_db.get_user_ratings_dict(user_id)

    if not user_ratings:
        st.info("Rate a few movies to unlock your analytics.")
        return

    ratings_series = pd.Series(list(user_ratings.values()))
    st.subheader("Rating Distribution")
    counts = ratings_series.value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Genre Breakdown")
    _render_genre_chart(user_ratings)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_movie_grid(recs: list[dict], user_id: str, show_match: bool):
    if not recs:
        st.write("No movies to show.")
        return

    cols = st.columns(3)
    for i, rec in enumerate(recs):
        col = cols[i % 3]
        with col.container(border=True):
            st.markdown(f"**{rec['title']}**")
            st.caption(rec.get("genres", ""))
            if show_match:
                match_pct = int(min(rec["score"], 1.0) * 100) if rec["score"] <= 1 else int(rec["score"])
                st.progress(min(rec["score"], 1.0) if rec["score"] <= 1 else 1.0, text=f"{match_pct}% match")

            existing_rating = ratings_db.get_rating(user_id, rec["movieId"])
            current = int(existing_rating["rating"]) if existing_rating else 0
            new_rating = st.select_slider(
                "Your rating", options=[0, 1, 2, 3, 4, 5], value=current,
                key=f"rate_{rec['movieId']}_{i}",
            )
            if new_rating != current and new_rating > 0:
                ratings_db.rate_movie(user_id, rec["movieId"], float(new_rating))
                st.toast(f"Rated {rec['title']} {new_rating}⭐")

            in_watchlist = watchlist_db.is_in_watchlist(user_id, rec["movieId"])
            label = "✅ In Watchlist" if in_watchlist else "❤️ Add to Watchlist"
            if st.button(label, key=f"watch_{rec['movieId']}_{i}", disabled=in_watchlist):
                watchlist_db.add_to_watchlist(user_id, rec["movieId"])
                interactions_db.log_interaction(user_id, rec["movieId"], "like")
                st.rerun()


def _top_genre_for_user(user_ratings: dict[int, float]) -> str | None:
    if not user_ratings:
        return None
    genre_scores: dict[str, float] = {}
    for movie_id, rating in user_ratings.items():
        movie = movies_db.get_movie(movie_id)
        if not movie:
            continue
        for g in movie.get("genres", []):
            genre_scores[g] = genre_scores.get(g, 0) + rating
    if not genre_scores:
        return None
    return max(genre_scores, key=genre_scores.get)


def _render_genre_chart(user_ratings: dict[int, float]):
    if not user_ratings:
        st.write("No ratings yet.")
        return
    genre_scores: dict[str, float] = {}
    for movie_id, rating in user_ratings.items():
        movie = movies_db.get_movie(movie_id)
        if not movie:
            continue
        for g in movie.get("genres", []):
            genre_scores[g] = genre_scores.get(g, 0) + rating
    if not genre_scores:
        st.write("No genre data yet.")
        return
    df = pd.Series(genre_scores).sort_values(ascending=False)
    st.bar_chart(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    connected, error = get_connection_status()
    if not connected:
        st.error(
            "Could not connect to MongoDB Atlas. Copy `.env.example` to `.env`, "
            "fill in your connection string, then restart the app.\n\n"
            f"Details: {error}"
        )
        st.stop()

    if "user" not in st.session_state:
        render_auth_screen()
        return

    user = users_db.get_user_by_id(str(st.session_state.user["_id"]))
    if not user:
        del st.session_state["user"]
        st.rerun()

    with st.sidebar:
        st.title("🎬 Movie Recommender")
        st.write(f"Signed in as **{user['name']}**")
        page = st.radio("Navigate", ["Dashboard", "Browse", "Watchlist", "Analytics"])
        if st.button("Log out"):
            del st.session_state["user"]
            st.rerun()

    if page == "Dashboard":
        page_dashboard(user)
    elif page == "Browse":
        page_browse(user)
    elif page == "Watchlist":
        page_watchlist(user)
    elif page == "Analytics":
        page_analytics(user)


if __name__ == "__main__":
    main()
