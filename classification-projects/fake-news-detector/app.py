import streamlit as st
from src.pipeline.predict_pipeline import Predictor
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detector")
st.caption("Paste a news headline or article text below to check if it's likely Fake or Real.")

@st.cache_resource
def load_model():
    try:
        BASE_DIR = Path(__file__).resolve().parent
        model_path = BASE_DIR / "artifacts" / "fake_news_pipeline.pkl"
        return Predictor(model_path)
    except Exception as e:
        logger.error(e)
        return None

predictor = load_model()


user_input = st.text_area(
    "Article Text",
    height=260,
    placeholder="Paste the news article here..."
)

i1, i2 = st.columns(2)

with i1:
    st.info(
        "**What the Model Analyzes**\n\n"
        "- Writing style & tone\n"
        "- Structural language cues\n"
        "- Common misinformation markers"
    )

with i2:
    st.info(
        "**Limitations**\n\n"
        "- Predictions are probabilistic\n"
        "- May struggle with satire or new topics"
    )

st.markdown("---")
predict = st.button("Predict")

if predict:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif predictor is None:
        st.error("Model not loaded.")
    else:
        try:
            text = [user_input]
            prediction = predictor.predict(text)[0]
            probability = predictor.predict_proba(text)[0]

            label = "Real" if prediction == 1 else "Fake"
            prob_real = probability[1]
            prob_fake = probability[0]

            st.subheader("Prediction Result")
            st.write(f"### **{label}**")

            c1, c2 = st.columns(2)
            c1.metric("Probability (Real)", f"{prob_real:.2%}")
            c2.metric("Probability (Fake)", f"{prob_fake:.2%}")

        except Exception as e:
            st.error("Prediction failed.")
            logger.error(e)
