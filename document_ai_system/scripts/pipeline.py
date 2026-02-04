"""
End-to-End Document AI Pipeline

- OCR + Regex bootstrapping
- LayoutLM inference
- Hybrid confidence-based selection
- SQLite persistence
- Multi-sheet Excel export for business review
"""

import os
import cv2
from pathlib import Path
import pandas as pd

from ocr_layout import pdf_to_images, run_tesseract, process_ocr
from extraction import extract_fields, generate_bio_labels
from mlops import log_extraction
from db_storage import init_db, create_document, save_fields
from modeling import infer_layoutlm, layoutlm_model, tokenizer

init_db()

EXCEL_OUTPUT = Path("data/processed/processed_invoices.xlsx")
excel_data = {}  # Collect all field records for multi-sheet Excel


# Helper Functions
def compute_field_confidence(field_value, tokens, token_confidences):
    """Average OCR confidence for field tokens"""
    if not field_value:
        return 0.0
    parts = field_value.split()
    scores = [token_confidences[i] for i, t in enumerate(tokens) if t in parts]
    return sum(scores) / len(scores) if scores else 0.5


def append_to_excel_data(document_id, filename, page_num, final_fields, confidences, sources):
    """Accumulate data per field for Excel multi-sheet output"""
    for field, value in final_fields.items():
        if field not in excel_data:
            excel_data[field] = []
        excel_data[field].append({
            "document_id": document_id,
            "filename": filename,
            "page": page_num,
            "field_value": value,
            "confidence": confidences.get(field, 0.0),
            "source": sources.get(field, "unknown")
        })


def write_excel():
    """Write all fields into separate sheets in Excel"""
    if not excel_data:
        return
    EXCEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(EXCEL_OUTPUT, engine="xlsxwriter") as writer:
        for field, records in excel_data.items():
            df = pd.DataFrame(records)
            # Excel sheet name max 31 chars
            df.to_excel(writer, sheet_name=field[:31], index=False)



# Main Pipeline
def process_document(path):
    """
    Process a single PDF or image document end-to-end:
    OCR - Regex - LayoutLM - Hybrid - DB - Excel - MLOps
    """
    filename = os.path.basename(path)
    document_id = create_document(filename)  # ONE document ID for all pages
    pages = []

    images = pdf_to_images(path) if path.lower().endswith(".pdf") else [cv2.imread(path)]

    for i, image in enumerate(images, start=1):
        # OCR
        words, w, h = run_tesseract(image)
        processed = process_ocr(words, w, h)
        tokens = [w["text"] for w in processed]
        bboxes = [w["bbox"] for w in processed]
        token_conf = [w["confidence"] for w in processed]

        # Regex bootstrapped extraction
        text = " ".join(tokens)
        fields = extract_fields(text)

        # BIO labels (optional, for analysis/training)
        labels = generate_bio_labels(tokens, fields)

        # LayoutLM inference
        model_fields = infer_layoutlm(layoutlm_model, tokens, bboxes)

        # Hybrid selection based on confidence
        final_fields, confidences, sources = {}, {}, {}
        for field in fields.keys():
            regex_value = fields.get(field)
            regex_conf = compute_field_confidence(regex_value, tokens, token_conf)
            model_value = model_fields.get(field)

            if regex_conf >= 0.9 and regex_value:
                final_fields[field] = regex_value
                confidences[field] = regex_conf
                sources[field] = "regex"
            elif model_value:
                final_fields[field] = model_value
                confidences[field] = 0.7
                sources[field] = "layoutlm"
            else:
                final_fields[field] = regex_value or ""
                confidences[field] = regex_conf if regex_value else 0.0
                sources[field] = "regex" if regex_value else "none"


        # Save to SQLite
        save_fields(document_id, final_fields, confidences, sources)

        # Accumulate for Excel
        append_to_excel_data(document_id, filename, i, final_fields, confidences, sources)

        # MLOps logging
        log_extraction(filename, tokens, fields)

        pages.append({
            "page": i,
            "tokens": tokens,
            "bboxes": bboxes,
            "labels": labels,
            "fields": final_fields,
            "confidences": confidences,
            "sources": sources
        })

    # Write Excel after all pages
    write_excel()

    return {
        "document_id": document_id,
        "file": filename,
        "pages": pages
    }
