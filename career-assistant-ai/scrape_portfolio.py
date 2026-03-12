import requests
from bs4 import BeautifulSoup
import pandas as pd

# Portfolio pages to scrape
urls = [
    "https://sharlynmuturi.github.io/",
    "https://sharlynmuturi.github.io/generic.html",
    "https://sharlynmuturi.github.io/elements.html"
]


# Technology keywords used for automatic detection
tech_keywords = [
"python", "r", "sql", "javascript", "html", "css",
"django", "flask", "streamlit",
"pytorch", "tensorflow",
"llm", "langchain", "huggingface", "openai", "chromadb",
"prophet",
"spark", "databricks",
"mysql", "postgresql", "sql server", "ssms", "access", "mongodb",
"tableau", "power bi",
"excel", "a/b testing",
"MLflow", "DVC"
]


# Function to detect technologies in descriptions
import re

def detect_stack(text):
    text_lower = text.lower()
    found = []
    for tech in tech_keywords:
        # Only match whole words (word boundaries) to fix the r
        if re.search(rf"\b{re.escape(tech)}\b", text_lower):
            found.append(tech)
    return ", ".join(sorted(found))
    
projects = []

# Loop through each portfolio page
for url in urls:

    print(f"Scraping: {url}")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    
    # Extract project titles and descriptions
    for project in soup.find_all("h2"):

        title = project.get_text(strip=True)

        description_tag = project.find_next("p")
        description = description_tag.get_text(strip=True) if description_tag else ""

        tech_stack = detect_stack(description)

        # Try to locate GitHub link near project
        github_link = ""
        for a in project.find_all_next("a", href=True, limit=5):
            if "github.com" in a["href"]:
                github_link = a["href"]
                break

        projects.append({
            "project_name": title,
            "description": description,
            "tech_stack": tech_stack,
            "link": github_link,
            "source_page": url
        })


# Convert to DataFrame
df = pd.DataFrame(projects)

# Remove duplicates if the same project appears on multiple pages
df = df.drop_duplicates(subset=["project_name"])


# Create combined text column for embeddings to improve vector search later
df["all_text"] = (
    df["project_name"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["tech_stack"].fillna("")
)


# Save dataset
df.to_csv("portfolio.csv", index=False)

print("\nCSV created successfully\n")
print(df)