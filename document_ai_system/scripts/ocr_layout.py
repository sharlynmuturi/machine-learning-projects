"""
OCR & Layout utilities

Responsible for converting documents into:
- tokens
- bounding boxes (LayoutLM format)
- OCR confidence
"""

import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path


def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF into page-wise images.
    LayoutLM operates page by page.
    """
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]


def run_tesseract(image, conf_threshold=40):
    """
    Run OCR and return word-level tokens with bounding boxes.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    h, w, _ = image.shape
    words = []

    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i])
        if text and conf >= conf_threshold:
            x, y, bw, bh = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            words.append({
                "text": text,
                "bbox": [x, y, x + bw, y + bh],
                "confidence": conf / 100
            })

    return words, w, h


def normalize_bbox(bbox, width, height):
    """
    Convert absolute pixels to LayoutLM 0-1000 scale
    """
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]


def process_ocr(words, width, height):
    """
    Final OCR output fed to models:
    - token
    - normalized bbox
    - confidence
    """
    return [{
        "text": w["text"],
        "bbox": normalize_bbox(w["bbox"], width, height),
        "confidence": w["confidence"]
    } for w in words]
