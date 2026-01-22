import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import Predictor
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Fake News Detector")

# Loading trained pipeline
@st.cache_resource
def load_model():
    try:
        BASE_DIR = Path(__file__).resolve().parent
        model_path = BASE_DIR / "artifacts" / "fake_news_pipeline.pkl"
        predictor = Predictor(model_path)
        return predictor
    except Exception as e:
        st.error("Failed to load the pipeline.")
        logger.error(e)
        return None

predictor = load_model()

# Main input box
st.subheader("Enter the news article text for prediction:")
user_input = st.text_area("Paste article text here:", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    elif predictor is None:
        st.error("Pipeline not loaded. Cannot predict.")
    else:
        try:
            prediction = predictor.predict(user_input)[0]
            probability = predictor.predict_proba(user_input)[0]

            label = "Real" if prediction == 1 else "Fake"
            prob_real = probability[1]
            prob_fake = probability[0]

            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"**Probability (Real):** {prob_real:.2f}")
            st.markdown(f"**Probability (Fake):** {prob_fake:.2f}")

        except Exception as e:
            st.error("Prediction failed.")
            logger.error(e)
