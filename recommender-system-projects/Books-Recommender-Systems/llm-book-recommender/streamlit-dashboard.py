import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

st.title("Semantic Book Recommender")

# Load data
@st.cache_data
def load_books():
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )

    return books

books = load_books()

# Load vector DB
@st.cache_resource
def load_db():
    # Load raw documents
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Persist the database if it doesn't exist
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="chroma_db")

    # Load from persisted directory (this is fast on subsequent runs)
    db_books = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())

    return db_books

db_books = load_db()

# Recommendation function
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
):

    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
    ]

    book_recs = books[
        books["isbn13"].isin(books_list)
    ].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values("joy", ascending=False)

    elif tone == "Surprising":
        book_recs = book_recs.sort_values("surprise", ascending=False)

    elif tone == "Angry":
        book_recs = book_recs.sort_values("anger", ascending=False)

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values("fear", ascending=False)

    elif tone == "Sad":
        book_recs = book_recs.sort_values("sadness", ascending=False)

    return book_recs.head(final_top_k)

# UI Controls
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

query = st.text_input("Describe the book you want", placeholder="e.g., A story about forgiveness")

category = st.selectbox("Select category", categories)

tone = st.selectbox("Select emotional tone", tones)


# Search button
if st.button("Find Recommendations"):

    recommendations = retrieve_semantic_recommendations(query, category, tone)

    st.subheader("Recommended Books")

    cols = st.columns(4)

    for i, (_, row) in enumerate(recommendations.iterrows()):

        with cols[i % 4]:

            description = row["description"]
            truncated = " ".join(description.split()[:30]) + "..."

            st.image(row["large_thumbnail"])

            st.markdown(f"**{row['title']}**")

            st.caption(truncated)
