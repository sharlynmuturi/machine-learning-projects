# Invoice Data Extraction (Document AI)

## Overview

This project demonstrates an end-to-end **Document AI pipeline** applied to invoice PDFs. The system is designed to extract key fields from invoices, such as invoice number, date, total amount and payment terms—using a combination of **rule-based extraction** and **machine learning**.

The goal is to showcase how AI can automate document processing, reduce manual entry and improve operational efficiency.

## Data Source

The sample invoice PDFs used in this project are a subset sourced from:

[Sample-Pdf-invoices by femstac](https://github.com/femstac/Sample-Pdf-invoices)


## Key Concepts Applied

### OCR (Optical Character Recognition)
- Uses **Tesseract OCR** to convert PDF pages or images into machine-readable text.
- Captures both **tokens** (words) and **bounding boxes** (spatial locations on the page).
- Provides a **confidence score** per token to indicate OCR reliability.

### Rule-Based Extraction (Regex)
- High-precision extraction for known patterns like:
  - `invoice_number`
  - `date`
  - `total_amount`
  - `terms`

### BIO Labeling for Weak Supervision
- Converts regex-extracted values into **BIO token labels** for training.
  - `B-` (Begin) indicates the first token of a field.
  - `I-` (Inside) indicates continuation tokens.
  - `O` (Outside) for all other tokens.
- Enables **weakly supervised learning** to bootstrap a LayoutLM model without manual annotation.

### LayoutLM Model Fine-Tuning
- **LayoutLM** is a transformer-based model that understands text **and layout**.
- Fine-tuned using OCR + regex labels to improve:
  - Robustness to diverse invoice formats
  - Recall for fields missed by regex
- Model outputs token-level predictions reconstructed into structured fields.

### Hybrid Confidence-Based Selection
- Combines regex and LayoutLM predictions:
  - Regex fields with high trust override model outputs.
  - Otherwise, LayoutLM provides fallback extraction.
- Confidence scores drive **highlighting for human review**:
  - Low-confidence fields are flagged

### Data Persistence and Review
- Extracted fields are saved to:
  - **SQLite database** 
  - **Excel workbook**

- Enables **human-in-the-loop validation**, where low-confidence fields can be reviewed.

### MLOps Logging
- Every extraction event is logged for:
  - Reproducibility
  - Monitoring extraction performance
  - Improving model retraining cycles

## Business Value

- **Automation:** Eliminates manual data entry from invoices
- **Accuracy & Trust:** Combines rule-based precision with ML robustness
- **Transparency:** Confidence scores and review flags ensure human oversight
- **Scalability:** Works across multiple pages and varied invoice layouts
