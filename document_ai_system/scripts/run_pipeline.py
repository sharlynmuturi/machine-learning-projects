from pathlib import Path
import pickle
from pipeline import process_document

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]

all_results = []

for file_path in DATA_DIR.iterdir():
    if file_path.suffix.lower() in VALID_EXTS:
        print(f"Processing {file_path.name} ...")
        result = process_document(str(file_path))
        all_results.append(result)

# Saving pickle for training
with open(PROCESSED_DIR / "all_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("All documents processed! Pickle saved.")
print(f"Total documents: {len(all_results)}")
