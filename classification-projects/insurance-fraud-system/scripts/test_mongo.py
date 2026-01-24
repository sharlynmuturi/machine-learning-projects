import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from db.mongo import policyholders_col

# Inserting test document
policyholder = {
    "_id": "PH001",
    "name": "John Mwangi",
    "phone": "0712345678",
    "email": "john@example.com",
    "risk_profile": "normal"
}

if not policyholders_col.find_one({"_id": "PH001"}):
    policyholders_col.insert_one(policyholder)

result = policyholders_col.find_one({"_id": "PH001"})
print("MongoDB test document:")
print(result)
