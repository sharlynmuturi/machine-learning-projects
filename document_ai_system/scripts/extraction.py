"""
Regex + BIO bootstrapping - Rule-based extraction & label bootstrapping

Rules give high precision, models learn recall & robustness.
"""

import re


PATTERNS = {
    "invoice_number": r"(?i)(?:invoice\s*#?\s*|order\s*id\s*[:\-]?\s*)([A-Z0-9\-]+)",
    "date": r"(?i)date[:\s]*([A-Za-z]{3,}\s\d{1,2}\s\d{4})",
    "total_amount": r"(?i)(?:total|balance due)[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
    "terms": r"(?i)terms[:\s]*([^\n]+)"
}


def extract_fields(text):
    """
    Extract high-precision fields using regex.
    (Used for baseline extraction and auto-labeling training data)
    """
    results = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            results[field] = match.group(1).strip()
        else:
            results[field] = None
    return results


def generate_bio_labels(tokens, extracted_fields):
    """
    Convert extracted values into BIO token labels.
    This is weak supervision (Not perfect, but enough to train LayoutLM)
    """
    labels = ["O"] * len(tokens)

    for field, value in extracted_fields.items():
        if not value:
            continue

        value_tokens = value.split()
        for i in range(len(tokens)):
            if tokens[i:i+len(value_tokens)] == value_tokens:
                labels[i] = f"B-{field.upper()}"
                for j in range(1, len(value_tokens)):
                    labels[i+j] = f"I-{field.upper()}"

    return labels
