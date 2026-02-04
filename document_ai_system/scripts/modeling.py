"""
Fine-Tuned LayoutLM Inference Module
- Loads trained LayoutLM model
- Token / bbox alignment
- Field reconstruction
- Confidence scoring
"""

from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
import torch
import torch.nn.functional as F


# Label definitions
LABELS = {
    "B-INVOICE_NUMBER": 0,
    "I-INVOICE_NUMBER": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-TOTAL_AMOUNT": 4,
    "I-TOTAL_AMOUNT": 5,
    "B-TERMS": 6,
    "I-TERMS": 7,
    "O": 8
}

ID2LABEL = {v: k for k, v in LABELS.items()}
LABEL2ID = LABELS


# Load fine-tuned model
MODEL_PATH = "layoutlm_invoice"

tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)

layoutlm_model = LayoutLMForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

layoutlm_model.eval()



# Inference
def infer_layoutlm(model, tokens, bboxes):
    """
    Run LayoutLM inference and reconstruct fields with confidence and returns:
    {
        "invoice_number": {"value": "...", "confidence": 0.87},
        "total_amount":  {"value": "...", "confidence": 0.92}
    }
    """

    encoding = tokenizer(
        tokens,
        boxes=bboxes,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits[0]          
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)

    word_ids = encoding.word_ids()

    entities = {}
    current_field = None
    current_tokens = []
    current_confidences = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        label_id = predictions[idx].item()
        label = model.config.id2label[label_id]
        token = tokens[word_id]
        confidence = probs[idx][label_id].item()

        if label.startswith("B-"):
            if current_field:
                entities[current_field] = {
                    "value": " ".join(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)
                }

            current_field = label[2:].lower()
            current_tokens = [token]
            current_confidences = [confidence]

        elif label.startswith("I-") and current_field == label[2:].lower():
            current_tokens.append(token)
            current_confidences.append(confidence)

        else:
            if current_field:
                entities[current_field] = {
                    "value": " ".join(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)
                }
                current_field = None
                current_tokens = []
                current_confidences = []

    # Flush last entity
    if current_field:
        entities[current_field] = {
            "value": " ".join(current_tokens),
            "confidence": sum(current_confidences) / len(current_confidences)
        }

    return entities
