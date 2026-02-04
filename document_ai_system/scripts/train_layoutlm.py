"""
End-to-End LayoutLM Fine-Tuning on Invoices

- Load OCR + regex-bootstrapped documents
- Prepare dataset with tokens, bboxes and BIO labels
- Tokenize and align labels for LayoutLM
- Fine-tune LayoutLM
- Save model & tokenizer for inference
"""

import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


DATA_DIR = Path("data/processed")  # Where OCR + regex results are stored
MODEL_OUTPUT_DIR = Path("layoutlm_invoice")  # Where fine-tuned model is to be saved
NUM_EPOCHS = 3
BATCH_SIZE = 2


# Defining labels
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

# Loading bootstrapped OCR + Regex data
# Expecting a Python object - list of dicts per document, each with "pages" - "tokens", "bboxes", "labels"
import pickle
with open(DATA_DIR / "all_results.pkl", "rb") as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} documents")


# Building HF Dataset
def build_dataset(results):
    """Flatten document pages into a list for HF Dataset"""
    rows = []
    for doc in results:
        for page in doc["pages"]:
            rows.append({
                "tokens": page["tokens"],
                "bboxes": page["bboxes"],
                "labels": page["labels"]
            })
    return Dataset.from_list(rows)

dataset = build_dataset(all_results)
print(f"Dataset contains {len(dataset)} pages")


# Tokenize & align labels
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=512)


def tokenize_and_align_labels(example):
    encoding = tokenizer(
        example["tokens"],
        boxes=example["bboxes"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    word_ids = encoding.word_ids()

    labels = []
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        else:
            labels.append(LABELS[example["labels"][word_id]])

    # bbox is ALREADY padded by tokenizer
    encoding["labels"] = labels

    # Remove offsets (Trainer doesn't need them)
    encoding.pop("offset_mapping")

    return encoding



tokenized_dataset = dataset.map(tokenize_and_align_labels)

# Initialize LayoutLM
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABELS
)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=10,
    report_to="none"
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


# Train
print("Starting LayoutLM fine-tuning...")
trainer.train()
print("Training complete!")

# Save model & tokenizer
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"Fine-tuned LayoutLM saved to {MODEL_OUTPUT_DIR}")
