# Fake News Detector

This project combines multiple datasets (LIAR, ISOT, and FakeNewsNet) and uses a **TF-IDF + SMOTE + Machine Learning pipeline** to classify news as **Fake** or **Real**. It also includes a **Streamlit web app** for interactive predictions and optional dataset exploration.

| Dataset        | Description                             |
|----------------|-----------------------------------------|
| [LIAR](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)           | Short political statements with labels |
| [ISOT Fake News](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset) | Real vs Fake articles, clean separation |
| [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)    | Headline-only CSVs (Politifact, GossipCop) |

The project uses a unified ML pipeline:

1. **Data ingestion (src/components/data_ingestion.py)**: Combine LIAR, ISOT, and FakeNewsNet datasets.  
2. **Data transformation (src/components/data_transformation.py)**: Clean text, vectorize using TF-IDF and handle class imbalance using SMOTE.  
3. **Model training (src/components/model_trainer.py)**: Train a machine learning model (Logistic Regression / Random Forest / XGBoost).  
4. **Prediction pipeline (src/pipeline/predict_pipeline.py)**: Unified pipeline for transforming input text and making predictions.  
5. **Streamlit App**: Web interface for user text input and real-time predictions.