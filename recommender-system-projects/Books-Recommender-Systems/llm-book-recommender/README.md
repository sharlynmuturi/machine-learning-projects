# Semantic Book Recommender with LLMs

This project uses Large Language Models (LLMs) to build a **semantic book recommendation system**.
Instead of relying on simple keyword matching, the system understands the **meaning of natural language queries** and recommends books that are semantically similar.

For example, a user could search for:

> "A story about someone seeking revenge"

and the system will return books with similar themes.

The project combines **text processing, vector search, zero-shot classification, sentiment analysis, and a simple web interface**.

---

# Project Components

The project is organized into five main stages:

## 1. Data Cleaning and Exploration
Notebook: `data-exploration.ipynb`

- Loads the [book dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata?resource=download) 
- Cleans and preprocesses text data
- Performs exploratory analysis of book descriptions

---

## 2. Semantic Vector Search
Notebook: `vector-search.ipynb`

- Converts book descriptions into **vector embeddings**
- Stores embeddings in a **vector database**
- Allows similarity search using natural language queries

Example query:
"A book about a person seeking revenge"


The system retrieves books with **similar semantic meaning**.

---

## 3. Text Classification (Zero-Shot Learning)
Notebook: `text-classification.ipynb`

Uses LLMs for **zero-shot classification** to categorize books as:

- Fiction
- Nonfiction

This creates a **filterable category facet** for the recommender system.

---

## 4. Sentiment and Emotion Analysis
Notebook: `sentiment-analysis.ipynb`

Extracts emotional signals from book descriptions such as:

- Joy
- Sadness
- Fear
- Surprise
- Anger

These emotional scores allow users to **sort books by tone**, for example:

- suspenseful
- joyful
- sad

---

## 5. Recommendation Web Application
Files: `streamlit-dashboard.py`, `gradio-dashboard.py`, `app.py`

A simple web application where users can:

- Enter natural language queries
- Filter by **fiction/nonfiction**
- Sort results by **emotional tone**
- View recommended books with descriptions and covers

---

# Requirements

The project was developed with **Python 3.11**.

Main dependencies include:

- kagglehub
- pandas
- matplotlib
- seaborn
- python-dotenv
- langchain-community
- langchain-opencv
- langchain-chroma
- transformers
- gradio
- notebook
- ipywidgets


# Environment Setup
To build the vector database you will need an OpenAI API key. The project loads environment variables using python-dotenv.

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

# Environment Setup
## 1. Install dependencies

```bash
pip install -r requirements.txt

```

## 2. Run the notebooks in order:


data-exploration.ipynb
vector-search.ipynb
text-classification.ipynb
sentiment-analysis.ipynb


## 3. Launch the web application

**Gradio interface**

```bash
python gradio-dashboard.py

```

**Streamlit semantic search**

```bash
streamlit run streamlit-dashboard.py

```

**Streamlit offline version**
  
This version of the application runs without using the OpenAI API. Normally, semantic search relies on embedding models hosted by OpenAI, which require an API key and internet access. 
This version performs local keyword-based search on the book descriptions so the recommender works entirely on your machine without calling external AI services.

```bash
streamlit run app.py

```