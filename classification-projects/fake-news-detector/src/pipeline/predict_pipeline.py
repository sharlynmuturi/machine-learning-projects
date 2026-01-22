from src.components.model_trainer import ModelTrainer
from src.utils import load_object
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

class Predictor:
    def __init__(self, pipeline_path="artifacts/fake_news_pipeline.pkl"):
        try:
            logger.info(f"Loading pipeline from {pipeline_path}")
            self.pipeline = load_object(pipeline_path)
        except Exception as e:
            logger.error("Failed to load pipeline")
            raise CustomException(e)

    def predict(self, texts):
        """
        Predict fake/real for a list of texts
        Returns: list of 0/1 labels
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            preds = self.pipeline.predict(texts)
            return preds
        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e)

    def predict_proba(self, texts):
        """
        Predict probability for fake/real
        Returns: list of [prob_fake, prob_real]
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            probs = self.pipeline.predict_proba(texts)
            return probs
        except Exception as e:
            logger.error("Prediction probability failed")
            raise CustomException(e)


if __name__ == "__main__":
    predictor = Predictor()
    sample_texts = [
        "Breaking news: Local man wins lottery after buying ticket online!",
        "This news article is completely fabricated for testing."
    ]

    predictions = predictor.predict(sample_texts)
    probabilities = predictor.predict_proba(sample_texts)

    for text, pred, prob in zip(sample_texts, predictions, probabilities):
        label = "Real" if pred == 1 else "Fake"
        logger.info(f"Text: {text}\nPrediction: {label}, Probability: {prob}\n")