import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

st.title("Semantic Book Recommender")
st.write("Discover books based on description, category, and emotional tone.")

BASE_DIR = Path(__file__).parent

EXCEL_PATH = BASE_DIR / "books_with_emotions.csv"

# Loading book data
@st.cache_data
def load_books():
    books = pd.read_csv(EXCEL_PATH)

    books["large_thumbnail"] = np.where(
        books["thumbnail"].isna(),
        "cover-not-found.jpg",
        books["thumbnail"] + "&fife=w800"
    )

    return books

books = load_books()

# Recommendation logic
@st.cache_data
def retrieve_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16
):

    if query:
    
        stop_words = {"a","the","and","about","story","of","in","to","with"}
    
        keywords = [w for w in query.lower().split() if w not in stop_words]
    
        pattern = "|".join(keywords)
    
        recs = books[
            books["description"].str.contains(pattern, case=False, na=False)
        ].head(initial_top_k)
    
    else:
        recs = books.head(initial_top_k)

    if category != "All":
        recs = recs[recs["simple_categories"] == category]

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        recs = recs.sort_values(tone_map[tone], ascending=False)

    return recs.head(final_top_k)

# Filters
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

st.markdown("### Search for a Book")

col1, col2, col3 = st.columns([3,2,2])

with col1:
    query = st.text_input(
        "Book description",
        placeholder="e.g., A story about grief and healing"
    )

with col2:
    category = st.selectbox("Category", categories)

with col3:
    tone = st.selectbox("Emotional tone", tones)

st.write("")

search_col1, search_col2, search_col3 = st.columns([1,1,1])
with search_col2:
    search_button = st.button("Find Recommendations")


# Display Recommendations
if search_button:
    
    recommendations = retrieve_recommendations(query, category, tone)

    st.markdown("### Recommended Books")

    books_per_row = 4

    for i in range(0, len(recommendations), books_per_row):

        row_books = recommendations.iloc[i:i+books_per_row]
        cols = st.columns(books_per_row)

        for col, (_, row) in zip(cols, row_books.iterrows()):

            with col:

                image = row["large_thumbnail"]

                if isinstance(image, str) and image.startswith("http"):
                    st.image(image, use_container_width=True)
                else:
                    st.image("cover-not-found.jpg", use_container_width=True)

                # Title
                st.markdown(f"**{row['title']}**")

                # Authors formatting
                authors_split = str(row["authors"]).split(";")

                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors_str = str(row["authors"])

                st.caption(f"by {authors_str}")

                # Description
                description = row["description"]
                truncated = " ".join(description.split()[:30]) + "..."
                st.caption(truncated)

            st.write("")
