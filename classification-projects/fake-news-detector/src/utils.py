import os
import pickle
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logger.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logger.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e)


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Directory ensured: {path}")
