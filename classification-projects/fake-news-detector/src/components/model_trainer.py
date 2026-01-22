from sklearn.metrics import classification_report, f1_score, accuracy_score
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object, load_object

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles model evaluation and saving.
    Works with a pre-built pipeline that includes vectorizer, SMOTE, and classifier.
    """

    def __init__(self, pipeline=None):
        """
        pre-built pipeline (TF-IDF + model)
        """
        self.pipeline = pipeline

    def evaluate_pipeline(self, X_test, y_test):
        """
        Evaluates the pipeline on test data
        """
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline not provided to ModelTrainer")

            logger.info("Starting model evaluation")

            y_pred = self.pipeline.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Accuracy on test set: {accuracy:.4f}")
            logger.info(f"F1 Score on test set: {f1:.4f}")
            logger.info("\n" + classification_report(y_test, y_pred))

            return f1, accuracy, y_pred

        except Exception as e:
            logger.error("Error during model evaluation")
            raise CustomException(e)

    def save_pipeline(self, file_path="artifacts/fake_news_pipeline.pkl"):
        """
        Saves the trained pipeline
        """
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline not provided to ModelTrainer")

            save_object(file_path, self.pipeline)
            logger.info(f"Pipeline saved successfully at: {file_path}")

        except Exception as e:
            logger.error("Error saving the pipeline")
            raise CustomException(e)

    def load_pipeline(self, file_path="artifacts/fake_news_pipeline.pkl"):
        """
        Load a saved pipeline
        """
        try:
            self.pipeline = load_object(file_path)
            logger.info(f"Pipeline loaded successfully from: {file_path}")
            return self.pipeline
        except Exception as e:
            logger.error("Error loading the pipeline")
            raise CustomException(e)