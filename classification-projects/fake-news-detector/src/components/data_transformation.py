from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import re

from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Simple text cleaning: lowercasing, remove URLs, non-alphabetic characters, extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DataTransformation:
    def __init__(self, model_type="logistic"):
        """
        Initialize the pipeline
        """
        self.model_type = model_type
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            )
        else:
            raise ValueError("Currently only logistic regression is supported")

        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english")),
                ("smote", SMOTE(random_state=42)),
                ("classifier", self.model)
            ]
        )

    def initiate_data_transformation(self, csv_path: str, test_size=0.2, random_state=42):
        """
        Reads CSV, splits train/test, fits pipeline on training data
        Returns fitted pipeline and test split
        """
        try:
            logger.info("Starting data transformation")

            df = pd.read_csv(csv_path)
            df = df.dropna(subset=["text"])
            df["text"] = df["text"].apply(clean_text)

            X = df["text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )

            # Fit pipeline on training data
            logger.info("Fitting pipeline (TF-IDF + SMOTE + Logistic Regression)")
            self.pipeline.fit(X_train, y_train)

            # Evaluate on test data
            y_pred = self.pipeline.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"F1 Score on test set: {f1:.4f}")
            logger.info("\n" + classification_report(y_test, y_pred))

            # Save the full pipeline
            save_object("artifacts/pipeline.pkl", self.pipeline)

            return self.pipeline, X_test, y_test

        except Exception as e:
            logger.error("Error during data transformation pipeline")
            raise CustomException(e)
