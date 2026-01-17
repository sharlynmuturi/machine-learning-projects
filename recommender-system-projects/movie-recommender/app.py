import streamlit as st
import pickle
import pandas as pd
import requests
import os

st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

st.title("Movie Recommendation System")
st.write("Search for a movie and get similar recommendations!")

# Load data
BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_data():
    movies_path = os.path.join(BASE_DIR, "movie_list.pkl")
    similarity_path = os.path.join(BASE_DIR, "similarity_list.pkl")
    
    movies = pickle.load(open(movies_path, "rb"))
    similarity = pickle.load(open(similarity_path, "rb"))
    return movies, similarity

movies, similarity = load_data()

# Fetch poster from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()

    poster_path = data.get("poster_path")
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image"

# Recommendation logic
def recommend(movie_title, n=5):
    index = movies[movies["title"].str.lower() == movie_title.lower()].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n + 1]

    recommendations = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        poster = fetch_poster(movie_id)

        recommendations.append((title, poster))

    return recommendations

# Search box UI
search_query = st.text_input("Search for a movie", placeholder="Type a movie title...")

if search_query:
    matches = movies[movies["title"].str.contains(search_query, case=False)]

    if len(matches) == 0:
        st.warning("No movies found.")
    else:
        selected_movie = st.selectbox("Select a movie", matches["title"].values)

        num_recs = st.slider("Number of recommendations", 3, 10, 5)

        if st.button("Recommend"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommend(selected_movie, num_recs)

            st.subheader("Recommended Movies")

            cols = st.columns(num_recs)
            for col, (title, poster) in zip(cols, recommendations):
                with col:
                    st.image(poster)
                    st.caption(title)

st.markdown("---")
