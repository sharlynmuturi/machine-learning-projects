# Fake News Detector

This project combines multiple datasets (LIAR, ISOT, FakeNewsNet) and **scraped real news articles** to classify news as **Fake** or **Real** using a **TF-IDF + SMOTE + Machine Learning pipeline**. It also includes a **Streamlit web app** for interactive predictions and optional dataset exploration.

| Dataset        | Description                             |
|----------------|-----------------------------------------|
| [LIAR](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)           | Short political statements with labels |
| [ISOT Fake News](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset) | Real vs Fake articles, clean separation |
| [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)    | Headline-only CSVs (Politifact, GossipCop) |
| Scraped News (Reuters, BBC) | Latest real-world news articles collected via RSS feeds and parsed with Newspaper3k |

---

## Project Workflow

1. **Data ingestion (`src/components/data_ingestion.py`)**:  
   - Combine LIAR, ISOT, and FakeNewsNet datasets.  
   - Scrape real news articles from trusted sources (Reuters, BBC) using **RSS feeds** and extract full article text with **Newspaper3k**.  

2. **Data transformation (`src/components/data_transformation.py`)**:  
   - Clean and normalize text.  
   - Vectorize using **TF-IDF**.  
   - Handle class imbalance with **SMOTE**.  

3. **Model training (`src/components/model_trainer.py`)**:  
   - Train machine learning models such as Logistic Regression, Random Forest, or XGBoost.  

4. **Prediction pipeline (`src/pipeline/predict_pipeline.py`)**:  
   - Unified pipeline to transform input text and make predictions on new articles.  

5. **Streamlit App**:  
   - Web interface for users to input text.  
   - Shows **prediction (Fake/Real) + probability**.  
   - Optional dataset statistics dashboard.  

---

## Notes

- Scraped news is **dynamically collected via RSS**, so your dataset can include **up-to-date real news articles**.  
- For reproducibility, all scraped articles are combined with curated datasets in a unified CSV (`data/processed/fake_news_full.csv`).  
- The ML pipeline supports adding more sources or expanding scraping to other trusted outlets in the future.  
