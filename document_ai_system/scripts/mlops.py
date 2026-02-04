"""
MLOps utilities, tracking:
- OCR output
- extraction quality
- model versions
"""

import mlflow


def log_extraction(file_name, tokens, fields):
    mlflow.set_experiment("document_ai_extraction")

    with mlflow.start_run():
        mlflow.log_param("file", file_name)
        mlflow.log_param("num_tokens", len(tokens))

        for k, v in fields.items():
            mlflow.log_param(k, v or "None")

        with open("ocr_text.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))

        mlflow.log_artifact("ocr_text.txt")
