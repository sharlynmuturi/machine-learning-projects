from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def run_training_pipeline():
    try:
        logger.info("=== Starting full training pipeline ===")

        # Step 1: Data ingestion
        ingestion = DataIngestion()
        csv_path = ingestion.initiate_data_ingestion()

        # Step 2: Data transformation & pipeline fitting
        transformer = DataTransformation()
        pipeline, X_test, y_test = transformer.initiate_data_transformation(csv_path)

        # Step 3: Evaluate & save pipeline
        trainer = ModelTrainer(pipeline=pipeline)
        f1, acc, _ = trainer.evaluate_pipeline(X_test, y_test)
        trainer.save_pipeline("artifacts/fake_news_pipeline.pkl")

        logger.info(f"Training pipeline completed. Test F1: {f1:.4f}, Accuracy: {acc:.4f}")

    except Exception as e:
        logger.error("Training pipeline failed")
        raise CustomException(e)


if __name__ == "__main__":
    run_training_pipeline()